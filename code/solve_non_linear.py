"""
Solve a non-linear equation 
"""

import numpy as np
import porepy as pp
import scipy.sparse as sps
from newton import newton_gb
import update_param 
import solve_sequential


def solve(rt_equations, solution_strategy="fully_coupled"):
    """
    Solve the non-linear equations, and possibly adjust the time step
    """
    
    # Check the solution strategy
    if (
            solution_strategy != "fully_coupled" and
            solution_strategy != "sequential"
        ):
        raise ValueError("Unknown solution strategy")
    # end if
    
    # Solve
    if solution_strategy == "fully_coupled":
        conv, newton_iter, flag = newton_gb(rt_equations.pde_solver,
                                            eq_keys_list=rt_equations.pde_solver.eq_names,
                                            var_keys_list=rt_equations.pde_solver.ad_vars
                                            )
    else:  # Solve sequentially
        # Note that newton_iter for the sequential iteration
        # is the number of sequential iteration
        conv, newton_iter, flag = solve_sequential.solve(rt_equations)
    # end if

    # Temporal information for the solver.
    mdg = rt_equations.pde_solver.equation_system.mdg
    grid = mdg.subdomains(dim=mdg.dim_max())[0]
    
    data = mdg.subdomain_data(grid)
    data_transport = data[pp.PARAMETERS]["transport"]
    dt = data_transport["time_step"]
    current_time = data_transport["current_time"]

    # If the solver converged, distribute the solution to the
    # next step and use as initial guess at the next time step.
    # Otherwise, repeat the current time step with a smaller time step
    if conv is True and flag == 0: # If Newton converged, both conditions are satisfied
        current_time += dt
        data[pp.PARAMETERS]["previous_newton_iteration"][
            "Number_of_Newton_iterations"
        ].append(newton_iter)
        data[pp.PARAMETERS]["previous_time_step"]["time_step"].append(dt)
        data_transport["current_time"] = current_time

        x = rt_equations.pde_solver.equation_system.get_variable_values(
            from_iterate=True         
        )
        
        rt_equations.pde_solver.equation_system.set_variable_values(
            values=x.copy(), 
            to_state=True, 
            to_iterate=False, 
            additive=False
        )
        
        # If successfully solved, update the permeability
        update_perm = data_transport.get("update_perm", False)
        if update_perm:
            
            # Update Darcy dictionaly
            update_param.update_darcy(rt_equations.pde_solver.equation_system)
            
            if solution_strategy == "sequential" :
                update_param.update_from_reaktoro(
                    rt_equations.chemical_solver, 
                    rt_equations.pde_solver.equation_system.mdg
                    )
            elif solution_strategy == "fully_coupled":
                update_param.update_concentrations(
                    rt_equations.pde_solver.equation_system.mdg,
                    rt_equations.pde_solver.equation_system,
                    )
                update_param.update_using_porepy(mdg) 
            # end if-else
        # end if
        
    elif conv is False or flag == 1:
        x = rt_equations.pde_solver.equation_system.get_variable_values(
            variables=rt_equations.pde_solver.ad_vars,
            from_iterate=False
        )
        rt_equations.pde_solver.equation_system.set_variable_values(
            variables=rt_equations.pde_solver.ad_vars,
            values=x.copy(), 
            to_state=False, 
            to_iterate=True, 
            additive=False
        )
    # end if

    # Adjust time step, if necessary, while making sure
    # the time step is not too big nor too small
    if flag == 0:
        if newton_iter < 3:  # Few steps, increase the time step
        
            if data_transport["constant_time_step"] is True:        
                data_transport["time_step"] = dt
            else:
                data_transport["time_step"] = np.minimum(dt*2 , 1 * pp.YEAR)
        elif newton_iter > 8 :
            # Used many iterations (or the solver didnt converge
            # in maximum number of iterations),
            # decreace the time step
            data_transport["time_step"] = np.maximum(dt / 10, 1e-10)
    else:  # flag == 1
        data_transport["time_step"] = np.maximum(dt / 8, 1e-10)
    # end if-elif

    # Check if the current time step will make the next current_time
    # larger than the final time point
    if flag == 0 and current_time + dt > data_transport["final_time"]:
        dt = data_transport["final_time"] - current_time
        data_transport["time_step"] = dt
    # end if

    data_transport["current_time"] = current_time

    return conv


def solve_eqs(rt_equations, solution_strategy="fully_coupled"):
    """Solve the non-linear equations in the equations class

    Parameters
    ----------
        equations: class that keeps track of the equation construction
        equation_system (pp.ad.Equation_system): A `PorePy` equation system

    """
    conv = solve(rt_equations, solution_strategy)
    
    # Return chemical value is converged.
    
    if conv:
        #If fully coupled, then we dont use Reaktoro,
        # i.e we return in a "different" way
        if solution_strategy == "fully_coupled":
            update_param.update_concentrations(
                rt_equations.pde_solver.equation_system.mdg,
                rt_equations.pde_solver.equation_system,
                )
        
        else:
            rt_equations.chemical_solver.set_state_values(
                rt_equations.pde_solver.equation_system.mdg
            )
    # end if
    
    return conv 
# end function

