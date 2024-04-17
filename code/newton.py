#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solve a system of non-linear equations on a MDG,
using Newton's method and the AD-framework in PorePy

@author: uw
"""

import numpy as np
#import scipy.sparse as sps
import scipy.sparse.linalg as spla
#import porepy as pp
#import update_param

def clip_variable(x, inds, min_val, max_val):
    """
    Helper method to cut the values of a target variable.
    Intended use is the concentration variable.
    
    inds is (often) given by equation_system.dof_of([target_name]) 

    """
    #inds = equation_system.dofs_of([target_name])
    x[inds] = np.clip(x[inds], a_min=min_val, a_max=max_val)

    # dof_ind = np.cumsum(np.hstack((0, dof_manager.full_dof)))
    # for key, val in dof_manager.block_dof.items():
    #     if key[1] == target_name:
    #         inds = slice(dof_ind[val], dof_ind[val + 1])
    #         x[inds] = np.clip(x[inds], a_min=min_val, a_max=max_val)
    #     # end if
    # # end key,val-loop

    return x


def backtrack(
        equation, 
        grad_f, 
        p_k, 
        x_k,
        f_0, 
        var_keys=None, eq_keys=None
        ):
    """
    Compute a stp size, using Armijo interpolation backtracing
    """
    # Initialize variables
    c_1 = 1e-4
    alpha = 1.0  # initial step size
    maxiter = 10
    min_tol = 1e-4
    flag = 0
    dot = grad_f.dot(p_k)
    
    # r = equation.assemble_subrhs(
    #       variables=var_keys,
    #       equations=eq_keys
    #       )
    # f_0 = 0.5 * r.dot(r)
    
   
    # The function to be minimised
    def phi(alpha):
        
        equation.set_variable_values(
            values=x_k + alpha * p_k, 
            variables=var_keys,
            to_iterate=True, 
            additive=False
            )
        
        F_new = equation.assemble_subrhs(
            equations=eq_keys,
            variables=var_keys
            )
        
        phi_step = 0.5 * F_new.dot(F_new)
        return phi_step

    # Variables
    alpha_prev = 1.0  # Initial step size
    phi_old = phi(alpha)
    phi_new = phi_old.copy()

    for i in range(maxiter):

        # If the first Wolfe condition is satisfied, we stop
        f_new = phi_new.copy()
        if f_new < f_0 + alpha * c_1 * dot:
            break
        # end if

        # Upper and lower bounds
        u = 0.5 * alpha
        l = 0.1 * alpha

        # Compute the new step size
        if i == 0:  # remember that we have not updated the iteration index yet,
            # hence we use the index one lower than what we expect

            # The new step size
            denominator = 2 * (phi_old - f_0 - dot)
            alpha_temp = -dot / denominator
           
        else:
            # The matrix-vector multiplication.
            mat = np.array(
                [
                    [1 / alpha**2, -1 / alpha**2],
                    [-alpha_prev / alpha**2, alpha / alpha_prev**2],
                ]
            )

            vec = np.array(
                [
                    phi_new - f_0 - dot * alpha,
                    phi_old - f_0 - dot * alpha_prev
                    ]
            )

            a, b = (1 / denominator) * np.matmul(mat, vec)

            if np.abs(a) < 1e-3:  # cubic interpolation becomes quadratic interpolation
                alpha_temp = -dot / (2 * b)
            else:
                alpha_temp = (-b + np.sqrt(np.abs(b**2 - 3 * a * dot))) / (3 * a)
            # end if-else
        # end if-else

        # Check if the new step size is to big
        # From a safty point of view, this helps if alpha_temp is inf.
        alpha_temp = min(alpha_temp, u)

        # Update the values, while ensuring that step size is not too small
        alpha_prev = alpha
        phi_old = phi_new
        alpha = max(alpha_temp, l)

        phi_new = phi(alpha)

        # Check if norm(alpha*p_k) is small. Stop if yes.
        # In such a case we might expect convergence
        if np.linalg.norm(alpha * p_k) < min_tol:
            break
        # end if

    # end i-loop

    return flag

def newton_gb(equations,  
              solution_strategy="fully_coupled", 
              var_keys_list = None,
              eq_keys_list = None,
              ):
    """Newton's method applied to an equtions class, pulling values 
    and state from the mdg

    Parameters
    ----------
        equations: class keeping track of all the equations
        
        equation system (pp.ad.Equation_system): 
        
        solve_stategy (str): which solution strategy should be used. Either 
            'fully_coupled' or 'sequential'. Defults to fully coupled.
            
        var_keys_list (list): The equation name of which equations 
            should be solved.            
        
        eq_keys_list (list): A list of the equations to be solved for.
    
    Returns
    ----------
        conv (bool): Wether Newton converged within a 
            tolerance and maximum number of iterations
        i (int): The number of iterations used
        flag (bool): Check if something went horribly wrong in the Newton iteration
        
    """
    
    # Use 'assemble_subsystem' to get the desired equations.
    # With var_keys and eq_keys set to None, we get the
    # same equations as using 'assemble'
    J,resid = equations.equation_system.assemble_subsystem(
        variables=var_keys_list,
        equations=eq_keys_list
        )

    norm_orig = np.linalg.norm(resid)
    
    conv = False
    i = 0
    maxit = 50

    flag = 0
    
    while conv is False and i < maxit:

        # Compute the search direction
        grad_f = J.T.dot(-resid)

        dx = spla.spsolve(J, resid, use_umfpack=False)

        # Solution from prevous iteration step
        x_prev = equations.equation_system.get_variable_values(
            variables=var_keys_list, 
            from_iterate=True
            )
        f_0 = 0.5 * resid.dot(resid)

        # Step size
        flag = backtrack(equations.equation_system, 
                          grad_f, dx, x_prev, f_0,
                          var_keys=var_keys_list, 
                          eq_keys=eq_keys_list)

        # # New solution
        x_new = equations.equation_system.get_variable_values(
            variables=var_keys_list,
            from_iterate=True
            )
        
        if np.any(np.isnan(x_new)):
            flag = 1
        # end if

        if flag == 1:
            break
        # end if

        # x_new = clip_variable(
        #     x_new.copy(), equation_system.dofs_of(["log_X"]), 
        #     min_val, max_val
        #     )

        equations.equation_system.set_variable_values(
            values=x_new.copy(), 
            variables=var_keys_list,
            to_state=False, 
            to_iterate=True, 
            additive=False
        )

        # --------------------- #

        # Increase number of steps
        i += 1
        
        # Assemble the rediescretised equations
        J, resid = equations.equation_system.assemble_subsystem(
            equations=eq_keys_list,
            variables=var_keys_list
            )
        
        # Measure the error
        norm_now = np.linalg.norm(resid)
        err_dist = np.linalg.norm(dx)
        
        # Stop if converged.
        if (
            norm_now < 1e-9 * norm_orig
            or norm_now < 1e-9 # The extra criteria are for safty
            or err_dist < 1e-9 * np.linalg.norm(x_new)
        ):
            conv = True
        # end if

    # end while

    # Print some information
    if solution_strategy == "fully_coupled":
       
        if flag == 0:
            print(f"Number of Newton iterations {i}")
            print(f"Residual reduction {norm_now / norm_orig}")
        elif flag == 1:
            print("Warning: NaN detected")
        # end if
    # end if
    
    # Return the number of Newton iterations;
    # we use it to adjust the time step
   
    return conv, i, flag

# end Newton