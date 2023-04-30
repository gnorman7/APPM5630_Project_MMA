import torch
import cooper
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import logging
from MMA import gcmmasub,subsolv,kktcheck,asymp,concheck,raaupdate


def gcmma_train(problem, epochs, maxinnerit, logger=None, device='cpu', track_param_history=False, logger_freq=1):
    if logger is None:
        logger = logging.getLogger(__name__)

    def get_loss_and_constraints(params_as_vec, compute_grad=False):
        """ Compute the loss as a 1 x 1 tensor and the constraints as a m x 1 vector. For both, compute the derivatives if compute_grad is True. The gradient of loss is n x 1, and the jacobian of the constraints is m x n. For now, these are numpy arrays, but they should be torch tensors for performance."""
        # Set the parameters, convert to pytorch if numpy
        if type(params_as_vec) == np.ndarray:
            params_as_vec = torch.tensor(params_as_vec,dtype=torch.float32)
        torch.nn.utils.vector_to_parameters(params_as_vec, problem.model.parameters())
    
        if compute_grad:
            problem.model.zero_grad()
        
        state = problem.closure()

        # Loss and gradient
        f0_val = np.zeros((1,1))
        f0_val[0,0] = state.loss.item()
        if compute_grad:
            state.loss.backward()
            df0dx = np.zeros((problem.n,1))
            i = 0
            for p in problem.model.parameters():
                if p.requires_grad==False or p.grad==None:
                    for p_elem in p:
                        i += 1
                else:
                    p_grad_np = p.grad.cpu().double().numpy().flatten()
                    for p_elem in p_grad_np:
                        df0dx[i] = p_elem
                        i += 1
                    # p.grad.zero_()
            problem.model.zero_grad()

        # Constraints and jacobian
        f_val = np.zeros((problem.m,1))
        f_val = state.ineq_defect.cpu().detach().numpy()
        if compute_grad:
            eee = torch.zeros((problem.m,1),dtype=torch.float32,device=device)
            dfdx = np.zeros((problem.m,problem.n))
            for j in range(problem.m):
                eee[j] = 1.0
                state.ineq_defect.backward(eee,retain_graph=True)
                i = 0
                for p in problem.model.parameters():
                    if p.requires_grad==False or p.grad==None:
                        # Added this new bit of code. Possible error computing gradients if only some parameters are trainable.
                        for p_elem in p_grad_np:
                            i += 1
                        # p.grad.zero_()
                        # p.grad = None
                    else:
                        p_grad_np = p.grad.cpu().clone().numpy().flatten()
                        for p_elem in p_grad_np:
                            dfdx[j,i] = p_elem
                            i += 1
                # problem.model.zero_
                eee[j] = 0.0
                problem.model.zero_grad()

        if compute_grad:
            return (f0_val, f_val, df0dx, dfdx)
        return (f0_val, f_val)


    def iteration_update(outeriter, f0val, fval, df0dx=None, dfdx=None, kktnorm=None, innerit=None):
        max_constraint_violation = np.max(fval)
        total_constraint_violation = np.sum(fval[fval>0])
        out_string = f'Iter: {outeriter}, obj: {f0val.item():.7e}, max constr: {max_constraint_violation:.4e}, total viol:{total_constraint_violation:.4e}'
        if kktnorm is not None:
            out_string += f', kkt: {kktnorm:.4e}'
        if innerit is not None:
            out_string += f', innerit: {innerit}'
        logger.info(out_string)


    # m : number of constraints, n : number of parameters
    m = problem.m
    n = problem.n
    # Lower and upper parameters
    lb = problem.lb
    ub = problem.ub

    logger.info(f'Found Model: {problem.model}')

    # Metrics to store
    f0s = np.zeros((epochs,1))
    fvals = np.zeros((epochs,m))
    kktnorms = np.zeros((epochs,1))
    grad_norms = np.zeros((epochs,1))
    jacobian_norms = np.zeros((epochs,m))
    inner_iterations = np.zeros((epochs,1))

    def store_metrics(outeriter, f0val, fval, df0dx, dfdx, kktnorm=None, innerit=None):
        f0s[outeriter] = f0val
        fvals[outeriter,:] = fval.flatten()
        kktnorms[outeriter] = kktnorm
        grad_norms[outeriter] = np.linalg.norm(df0dx)
        jacobian_norms[outeriter,:] = np.linalg.norm(dfdx,axis=1)
        inner_iterations[outeriter] = innerit

    epochs -= 1
    epsimin = 0.00001
    eeen = np.ones((n,1))
    eeem = np.ones((m,1))
    zerom = np.zeros((m,1))
    xval = torch.nn.utils.parameters_to_vector(problem.model.parameters()).cpu().detach().unsqueeze(dim=1).float().numpy()
    if track_param_history:
        xhist = []
        xhist.append(xval)
    xold1 = xval.copy()
    xold2 = xval.copy()
    # Increasing to give faster convergence / changes... haven't paid any attention to this...
    xmax = ub*eeen
    xmin = lb*eeen
    low = xmin.copy()
    upp = xmax.copy()
    c = 1000*eeem
    d = eeem.copy()
    a0 = 1
    a = zerom.copy()
    raa0 = 0.001
    raa = 0.001*eeem
    raa0eps = 0.0001
    raaeps = 0.0001*eeem
    maxoutit = epochs
    maxinnerit = maxinnerit
    kkttol = 0		
    outeriter = 0

    # Calculate values and gradients of the objective and constraints
    if outeriter == 0:
        (f0val,fval,df0dx,dfdx) = get_loss_and_constraints(xval,compute_grad=True)
        iteration_update(outeriter, f0val, fval, df0dx, dfdx)
        store_metrics(outeriter, f0val, fval, df0dx, dfdx)

        innerit = 0

    # The iterations starts
    kktnorm = kkttol+10

    while (kktnorm > kkttol) and (outeriter < maxoutit):
        outeriter += 1
        # The parameters low, upp, raa0 and raa are calculated:
        low,upp,raa0,raa= \
            asymp(outeriter,n,xval,xold1,xold2,xmin,xmax,low,upp,raa0,raa,raa0eps,raaeps,df0dx,dfdx)
        
        # The MMA subproblem is solved at the point xval:
        xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,f0app,fapp= \
            gcmmasub(m,n,None,epsimin,xval,xmin,xmax,low,upp,raa0,raa,f0val,df0dx,fval,dfdx,a0,a,c,d)
        # 1

        # The user should now calculate function values (no gradients) of the objective- and constraint
        # functions at the point xmma ( = the optimal solution of the subproblem).
        (f0valnew,fvalnew) = get_loss_and_constraints(xmma)
        
        # It is checked if the approximations are conservative:
        conserv = concheck(m,epsimin,f0app,f0valnew,fapp,fvalnew)

        # While the approximations are non-conservative (conserv=0), repeated inner iterations are made:
        innerit = 0
        if conserv == 0:
            while conserv == 0 and innerit < maxinnerit:
                innerit += 1
                # New values on the parameters raa0 and raa are calculated:
                raa0,raa = raaupdate(xmma,xval,xmin,xmax,low,upp,f0valnew,fvalnew,f0app,fapp,raa0, \
                    raa,raa0eps,raaeps,epsimin)
                # The GCMMA subproblem is solved with these new raa0 and raa:
                xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,f0app,fapp = gcmmasub(m,n,None,epsimin,xval,xmin, \
                    xmax,low,upp,raa0,raa,f0val,df0dx,fval,dfdx,a0,a,c,d)
                # The user should now calculate function values (no gradients) of the objective- and 
                # constraint functions at the point xmma ( = the optimal solution of the subproblem).
                (f0valnew,fvalnew) = get_loss_and_constraints(xmma)
                
                # It is checked if the approximations have become conservative:
                conserv = concheck(m,epsimin,f0app,f0valnew,fapp,fvalnew)
        # Some vectors are updated:
        xold2 = xold1.copy()
        xold1 = xval.copy()
        xval = xmma.copy()
        if track_param_history:
            xhist.append(xval)
        
        # Re-calculate function values and gradients of the objective and constraints functions
        (f0val,fval,df0dx,dfdx) = get_loss_and_constraints(xval,compute_grad=True)

        # The residual vector of the KKT conditions is calculated
        residu,kktnorm,residumax = \
            kktcheck(m,n,xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,xmin,xmax,df0dx,fval,dfdx,a0,a,c,d)
        
        if ((outeriter >= logger_freq and outeriter % logger_freq == 0) or outeriter == maxoutit-1):
            iteration_update(outeriter, f0val, fval, df0dx=None, dfdx=None, kktnorm=kktnorm, innerit=innerit)
        store_metrics(outeriter, f0val, fval, df0dx, dfdx, kktnorm, innerit)

    metrics = {'f0s': f0s, 'fvals': fvals, 'kktnorms': kktnorms, 'grad_norms': grad_norms, 'jacobian_norms':jacobian_norms, 'inner_iterations': inner_iterations}
    if track_param_history:
        return metrics, xhist
    else:
        return metrics

