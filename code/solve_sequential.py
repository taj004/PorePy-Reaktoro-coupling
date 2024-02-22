"""Solve the reactive transport processes in a sequential sense,

"""
import newton
import numpy as np

def solve(rt_equations):

    conv_sequential = False
    sequential_iteration = 0
    max_sequential_iterations = 10
    
    # To check cell-wise convergence¨
    rt_equations.chemical_solver.conv = np.array(
        [False] * rt_equations.pde_solver.equation_system.mdg.num_subdomain_cells() 
        )
        
    while conv_sequential is False and sequential_iteration < max_sequential_iterations:
        
        # ---- Solve the transport processes 

        conv_transport, i, flag = newton.newton_gb(
            rt_equations.pde_solver,
            solution_strategy="sequential",
            var_keys_list= rt_equations.pde_solver.ad_vars,
            eq_keys_list=rt_equations.pde_solver.eq_names,
        )
        
        # ---- Solve the chemical processes 
    
        # Get solid partition of the elements
        solid_elements = rt_equations.chemical_solver.element_values(
            mdg=rt_equations.pde_solver.equation_system.mdg,
            inds=rt_equations.chemical_solver.solid_indices,
        )

        # Transported aqueous element values
        fluid_elements = rt_equations.pde_solver.equation_system.get_variable_values(
            variables=["aq_amount"], from_iterate=True
        )

        # Update total amount of elements
        rt_equations.pde_solver.equation_system.set_variable_values(
            values=solid_elements + fluid_elements,
            variables=["total_amount"],
            to_state=False,
            to_iterate=True,
            additive=False,
        )

        # Code: call on Reaktoto via the PorePy-Reaktoro interface,
        # to solve the problem chemical problem
        conv_react = rt_equations.chemical_solver.solver_preparation_and_solve(
            mdg=rt_equations.pde_solver.equation_system.mdg,
        )

        # Check convergence
        react_has_not_conv = False in conv_react
        
        # ------------------- #
        
        # # Check convergence
        sequential_iteration += 1
        print(f"SI nr {sequential_iteration}")
        if conv_transport and react_has_not_conv is False:
            conv_sequential = True
            # equations.chemical_solver.set_state_values(equation_system.mdg)
        else:
            # Use the updated solid elements to update fluid elements
            rt_equations.pde_solver.get_solute_transport_CC_eqs(
                rt_equations.pde_solver.equation_system,
                iterate=True,
                U_solid=rt_equations.chemical_solver.element_values(
                    mdg=rt_equations.pde_solver.equation_system.mdg, 
                    inds=rt_equations.chemical_solver.solid_indices
                )
            )   
        # end if

    # end while

    print(f"Number of sequential iterations: {sequential_iteration}")

    return conv_sequential, sequential_iteration, flag

