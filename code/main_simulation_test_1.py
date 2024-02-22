#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 09:25:51 2023

Main script
@author: uw
"""

#%% Import the neseccary packages
import numpy as np
import porepy as pp
import reaktoro.reaktoro4py as rt
import matplotlib.pyplot as plt

import reactive_transport
import equations
from solve_non_linear import solve_eqs
import update_param
import porepy_reaktoro_interface

import os

#%% Initialise variables related to the chemistry and the domain

# Define mdg
sd = pp.CartGrid(nx=[int(100)], physdims=[1])
sd.compute_geometry()
mdg = pp.MixedDimensionalGrid()
mdg.add_subdomains(sd)
domain = {"xmin": 0, "xmax": np.max(sd.face_centers[0])}  # domain

# Keywords
mass_kw = "mass"
chemistry_kw = "chemistry"
transport_kw = "transport"
flow_kw = "flow"

pressure = "pressure"
tot_var = "total_amount"
aq_var = "aq_amount"
temperature = "temperature"

#%% Initial chemistry

aq_species = [
    "CO2(aq)",
    "Cl-",
    "H2O2(aq)",
    "MgCO3(aq)",
    "CO3-2",
    "ClO-",
    "HCO3-",
    "MgCl+",
    "Ca(HCO3)+",
    "ClO2-",
    "HCl(aq)",
    "MgOH+",
    "Ca+2",
    "ClO3-",
    "HClO(aq)",
    "Na+",
    "CaCO3(aq)",
    "ClO4-",
    "HClO2(aq)",
    "NaCl(aq)",
    "CaCl+",
    "H+",
    "HO2-",
    "NaOH(aq)",
    "CaCl2(aq)",
    "H2(aq)",
    "Mg(HCO3)+",
    "O2(aq)",
    "CaOH+",
    "H2O(aq)",
    "Mg+2",
    "OH-",

]

mineral_species = ["Quartz", "Calcite", "Dolomite"]

# Database
db = rt.SupcrtDatabase("supcrt98")

aq_phase = rt.AqueousPhase(aq_species)
aq_phase.setActivityModel(
    rt.chain(   
    rt.ActivityModelPitzer(),
    rt.ActivityModelDrummond("CO2(aq)")
    )
    
    )

mineral_phase = rt.MineralPhases(mineral_species)
mineral_phase.setActivityModel(rt.ActivityModelIdealSolution(rt.StateOfMatter.Solid))

chem_system = rt.ChemicalSystem(db, aq_phase, mineral_phase)

chemical_solver = porepy_reaktoro_interface.ChemicalSolver(
    chemical_system=chem_system, 
    kinetic=False,
    use_ODML=False
)

# Solver specifications
options = rt.EquilibriumOptions()
options.optima.maxiters = 40
options.optima.convergence.requires_at_least_one_iteration=False
chemical_solver.solver.setOptions(options)

# Initial (guess) values for chemical values [mol]
chemical_solver.state.set("H2O(aq)" , 1.0, "kg")
chemical_solver.state.set("NaCl(aq)", 0.7, "mol")
chemical_solver.state.set("Calcite" , 10.0, "mol")
chemical_solver.state.set("Quartz"  , 10.0, "mol")  

#%% Set pressure, temperature, and initialise chemical states

p = 100
t = pp.CELSIUS_to_KELVIN(60.0)
chemical_solver.state.pressure(p, "bar")
chemical_solver.state.temperature(t, "K")

init_conv = chemical_solver.solve_chemistry()

# Porosity
porosity = 0.1 

# Scale aqueous phase
chemical_solver.scale_aqueous_phase_in_state(porosity)

# Scale the minerals
chemical_solver.state.scalePhaseVolume("Quartz",  0.882, "m3")
chemical_solver.state.scalePhaseVolume("Calcite", 0.018, "m3")

# For lagging mineral
chemical_solver.state.scalePhaseVolume("Calcite", 0.018, "m3")

# Multiple states for RT simulations
chemical_solver.multiple_states(mdg.num_subdomain_cells())

init_c = chemical_solver.state.componentAmounts().asarray()

# We also need the aqueous parts and mineral parts separated
a_species = chemical_solver.state.speciesAmounts().asarray()[
    chemical_solver.fluid_indices
]

init_aq = chemical_solver.chem_system.formulaMatrix()[
    :, chemical_solver.fluid_indices
].dot(a_species)

#%% Boundary chemistry

bc_chemistry = porepy_reaktoro_interface.ChemicalSolver(chemical_system=chem_system)

bc_chemistry.state.pressure(p, "bar")
bc_chemistry.state.temperature(t, "K")

bc_chemistry.state.set("H2O(aq)"  , 1.0, "kg")
bc_chemistry.state.set("NaCl(aq)" , 0.9, "mol")
bc_chemistry.state.set("MgCl+"     , 0.05, "mol")
# Note that MgCl+ is used insted of MgCl2,
# following https://reaktoro.org/tutorials/miscellaneous/defining-materials.html
bc_chemistry.state.set("CaCl2(aq)", 0.01, "mol")
bc_chemistry.state.set("CO2(aq)"  , 0.75, "mol")

bc_conv = bc_chemistry.solve_chemistry()
bc_chemistry.state.scalePhaseVolume("AqueousPhase", porosity, "m3")
bc_c = bc_chemistry.state.componentAmounts().asarray()

bc_fluid = bc_chemistry.element_values(inds=bc_chemistry.fluid_indices)

# %% Loop over the mdg, and set initial and default data

for sd, d in mdg.subdomains(return_data=True):
    
    # Initialize the primary variable dictionaries
    d[pp.PRIMARY_VARIABLES] = {
        tot_var: {"cells": chemical_solver.num_components},
        aq_var: {"cells": chemical_solver.num_components},
    }

    # Initialize a state
    pp.set_state(d)

    unity = np.ones(sd.num_cells)

    # ---- Boundary conditions----------------------------- #

    bound_faces = sd.tags["domain_boundary_faces"].nonzero()[0]

    bound_faces_centers = sd.face_centers[:, bound_faces]

    # Neumann faces
    neu_faces = np.array(["neu"] * bound_faces.size)
    # Define the inflow and outflow in the domain
    inflow = bound_faces_centers[0, :] < domain["xmin"] + 1e-4
    outflow = bound_faces_centers[0, :] > domain["xmax"] - 1e-4

    # The boundary conditions for transport. Transport here means
    # transport of solutes
    labels_for_transport = neu_faces.copy()
    #labels_for_transport[inflow] = "dir"
    labels_for_transport[outflow] = "dir"
    bound_for_transport = pp.BoundaryCondition(
        sd, faces=bound_faces, cond=labels_for_transport
    )

    # Set bc values. Note, at each face we have num_aq_components
    expanded_left = pp.fvutils.expand_indices_nd(
        bound_faces[inflow], chemical_solver.num_components
    )

    expanded_right = pp.fvutils.expand_indices_nd(
        bound_faces[outflow], chemical_solver.num_components
    )
    # Ny = bound_faces[inflow].size
    bc_values_for_transport = np.zeros(sd.num_faces * chemical_solver.num_components)

    bc_values_for_transport[expanded_left] = np.tile(-bc_c/(7 *pp.DAY)*porosity, bound_faces[inflow].size)
    bc_values_for_transport[expanded_right] = np.tile(
        init_c, bound_faces[outflow].size
    )

    # ---- Set the values in dictionaries ---- #

    mass_data = {
        "porosity": porosity * unity,
        "mass_weight": unity.copy(),
        "num_components": chemical_solver.num_components,
    }

    transport_data = {
        "bc_values": bc_values_for_transport,
        "bc": bound_for_transport,
        "num_components": chemical_solver.num_components,
        "darcy_flux": 1/(7*pp.DAY) * sd.face_areas,
    }

    # Set the parameters in the dictionary
    # Treat this different to make sure parameter dictionary is constructed
    d = pp.initialize_data(sd, d, mass_kw, mass_data)

    #d[pp.PARAMETERS][flow_kw] = {} #flow_data
    d[pp.PARAMETERS][transport_kw] = transport_data
    d[pp.DISCRETIZATION_MATRICES][transport_kw] = {}

    d[pp.PARAMETERS][chemical_solver.chemical_name] = {}
    
    d[pp.PARAMETERS][transport_kw].update(
        {
            "time_step": 1800,  # s,
            "current_time": 0,
            "final_time": 50 * pp.DAY,
            "constant_time_step": True,
            "update_perm": False
        }
    )
    d[pp.PARAMETERS]["grid_params"] = {}
    d[pp.PARAMETERS]["previous_time_step"] = {"time_step": []}
    d[pp.PARAMETERS]["previous_newton_iteration"] = {
        "Number_of_Newton_iterations": []
    }

    # ---- Set state values----------------------------- #

    # Cellwise inital concentration values, replicated versions 

    total_amount = np.tile(init_c, sd.num_cells)
    aq_amount = np.tile(init_aq, sd.num_cells)

    d[pp.STATE].update(
        {
            pressure: p * pp.BAR * unity,
            tot_var: total_amount.copy(),
            aq_var: aq_amount,
            temperature: t * unity,

            pp.ITERATE: {
                pressure: p * pp.BAR * unity,
                tot_var: total_amount.copy(),
                aq_var: aq_amount.copy(), 
                temperature: t * unity,  
            },
        }
    )
# end sd,d-loop

#%% Aspects for the calculations

# Equation system
equation_system = pp.ad.EquationSystem(mdg)

# Chemical names
chemical_solver.set_species_in_PorePy_dict_from_state(mdg)

# and the initial equations
eqs = equations.EquationConstruction(equation_system=equation_system)

# To keep track of the RT eqs
rt_eqs = reactive_transport.ReactiveTransport(pde_solver=eqs, 
                                              chemical_solver=chemical_solver)

#%% Prepere for exporting

# The data
sd = mdg.subdomains(dim=1)[0]
data = mdg.subdomain_data(sd)

current_time = data[pp.PARAMETERS]["transport"]["current_time"]
final_time = data[pp.PARAMETERS]["transport"]["final_time"]

# Make folder
folder_name = "pictures/simulation_1/"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
# end if

#%%  Plot sinlge plots 

# def plot_1d_single(title_time):
    
#     # Check that the grid is 1D
#     assert sd.dim == 1, print(f"Grid has dimension {sd.dim}, but expected grid has dimension 1")
    
#     # The x-axis    
#     x = sd.cell_centers[0][0:50] #
#     #x=np.linspace(0, 0.5, num=int(sd.num_cells/2))

#     # Plot conc
#     plt.Figure(figsize=(25,25))
#     conc_name = ["CO2(aq)", "Ca+2", "HCO3-", "OH-", "H+", "Mg(HCO3)+"]
#     colour_name = ["b", "purple" ,"violet", "r", "g", "k"]
#     label_name = ["CO$_{2}$(aq)", "Ca$^{2+}$", "HCO$_{3}^{-}$", 
#                   "OH$^{-}$", "H$^{+}$", "Mg(CO$_{3}$)$^{+}$" ]
#     for i in range(len(conc_name)):

#         # conc = d[pp.STATE][conc_name[i]][0:x.size]

#         conc = np.zeros(int(sd.num_cells/2))
#         for j in range(int(sd.num_cells/2)):
#             aqprops = rt.AqueousProps(chemical_solver.states[j])
#             conc[j] = aqprops.speciesMolality(conc_name[i])
#         # end j-loop
        
#         #plt.plot(x, np.log2(conc), color=colour_name[i], label=conc_name[i] )
#         plt.semilogy(x, conc, "-o", color=colour_name[i], label=label_name[i])
#         plt.ylim((pow(10, -11), pow(10, 1)))
#     # end i-loop
    
#     plt.legend(loc="upper right", fontsize=17)
#     plt.yticks(fontsize=20)
#     plt.xticks(fontsize=20)
#     plt.xlabel("x [m]", fontsize=20)
#     plt.ylabel("Concentration [molal]", fontsize=20)
#     plt.title(title_time, fontsize=25)
    
#     title_time_list = title_time.split() 
#     save_name =title_time_list[0]+"_"+title_time_list[1]+ "_corrected_activity_modified_flux.pdf" # "HFK_activity.pdf"

#     # plt.savefig(folder_name+"conc_"+save_name, 
#     #             bbox_inches="tight", 
#     #             pad_inches=0.1)
#     plt.show()

#     # plot mineral
#     plt.Figure(figsize=(25,25))
    
#     conc_name = ["Calcite", "Dolomite"]
#     colour_name = ["darkblue", "darkred"]
    
#     min_vol = np.zeros(x.size)
#     tot_volume = update_param.calulate_mineral_volume(chemical_solver, mdg)

#     for i in range(len(conc_name)):
#         for j in range(int(sd.num_cells/2)):

#             props = chemical_solver.states[j].props()
            
#             # # Calculate the total volume in the indivdual cell
#             # total_volume = 0

#             # for phase in chemical_solver.chem_system.phases():

#             #     if phase.name() != "AqueousPhase":
#             #         total_volume += props.phaseProps(phase.name()).volume().val()
#             #         print(phase.name())
#             #         print(total_volume)
#             #     # end if
#             # # end phase-loop
#             #breakpoint()
#             # Calculate the mineral volumes, in the indivdual cell
            
#             if tot_volume[j] < 1e-6:
#                 min_vol[j]=0
#             else:
#                 min_vol[j] = props.phaseProps(conc_name[i]).volume().val() * 100 / tot_volume[j]
#             # end if-else
            
#         # end j-loop
#         plt.plot(x, min_vol, "-o", color=colour_name[i], label=conc_name[i])
        
#     # end i-loop

#     plt.legend(loc="center right", fontsize=17)
#     plt.yticks(fontsize=20)
#     plt.xticks(fontsize=20)
#     plt.xlabel("x [m]", fontsize=20)
#     plt.ylabel("Mineral volume [%]", fontsize=20)
#     plt.title(title_time, fontsize=25)
    
#     # plt.savefig(folder_name+"minerals_"+save_name, 
#     #             bbox_inches="tight", 
#     #             pad_inches=0.1)
    
#     plt.show() 
# # end function

#%% Plot as a 2x3 figure (done for the paper)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18,10))
plt.subplots_adjust(wspace=0.4, hspace=0.3)
axes = axes.ravel()

def plot_1d_several(title_time, index):
    # index = 0,1,2
    
    # Check that the grid is 1D
    assert sd.dim == 1, print(f"Grid has dimension {sd.dim}, but expected grid has dimension 1")
    
    # The x-axis    
    x = sd.cell_centers[0][0:50] 
    #x=np.linspace(0, 0.5, num=int(sd.num_cells/2))
   
    # x-ticks
    xt = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    
    # Plot conc
    ax = axes[index]
    conc_name = ["CO2(aq)", "Ca+2", "HCO3-", "OH-", "H+", "Mg(HCO3)+"]
    colour_name = ["b", "purple" ,"violet", "r", "g", "k"]
    label_name = ["CO$_{2}$(aq)", "Ca$^{2+}$", "HCO$_{3}^{-}$", 
                  "OH$^{-}$", "H$^{+}$", "Mg(CO$_{3}$)$^{+}$" ]
    for i in range(len(conc_name)):

        conc = np.zeros(int(sd.num_cells/2))
        for j in range(int(sd.num_cells/2)):
            aqprops = rt.AqueousProps(chemical_solver.states[j])
            conc[j] = aqprops.speciesMolality(conc_name[i])
        # end j-loop
        
        ax.semilogy(x, conc, "-o", color=colour_name[i], label=label_name[i])
        ax.set_ylim((pow(10, -11), pow(10, 1)))
    # end i-loop
    
    if index == 2:
        ax.legend(loc="upper right", fontsize=17, bbox_to_anchor=(1.7, 1.0))
    # else
    
    ax.tick_params(axis="y", labelsize=20)
    ax.set_xticks(ticks=xt, labels=xt, fontsize=20)
    ax.set_xlabel("x [m]", fontsize=20)
    if index == 0:
        ax.set_ylabel("Concentration [molal]", fontsize=20)
    # end if
    ax.set_title(title_time, fontsize=25)
    
    # plot mineral    
    conc_name = ["Calcite", "Dolomite"]
    colour_name = ["darkblue", "darkred"]
    
    min_vol = np.zeros(x.size)
    tot_volume = update_param.calulate_mineral_volume(chemical_solver, mdg)
    
    ax = axes[index+3]
    for i in range(len(conc_name)):
        for j in range(int(sd.num_cells/2)):

            props = chemical_solver.states[j].props()
            
            # # Calculate the total volume in the indivdual cell
            
            if tot_volume[j] < 1e-6:
                min_vol[j]=0
            else:
                min_vol[j] = props.phaseProps(conc_name[i]).volume().val() * 100 / tot_volume[j]
            # end if-else
        # end j-loop
        ax.plot(x, min_vol, "-o", color=colour_name[i], label=conc_name[i])    
    # end i-loop
    
    if index == 2:
        ax.legend(loc="center right", fontsize=17, bbox_to_anchor=(1.7, 0.81))
    # else
    
    ax.tick_params(axis="both", labelsize=20)
    ax.set_xlabel("x [m]", fontsize=20)
    ax.set_xticks(ticks=xt, labels=xt, fontsize=20)
    
    if index == 0: 
        ax.set_ylabel("Mineral volume [%]", fontsize=20)
    # end if
    
# end function

#%% Time loop
store_time=[30 * pp.MINUTE, 10*pp.HOUR, 50 * pp.DAY]
title_time = ["30 Minutes", "10 Hours", "50 Days"]
j=0

while current_time < final_time : 

    print(f"Current time {current_time}")

    # Initialise equations
    rt_eqs.pde_solver.get_solute_transport_CC_eqs(
        equation_system, 
        iterate=False,
        U_solid = rt_eqs.chemical_solver.element_values(
            mdg=rt_eqs.pde_solver.equation_system.mdg, 
            inds=rt_eqs.chemical_solver.solid_indices
            ),
        )

    conv = solve_eqs(rt_eqs, solution_strategy="sequential")
    current_time = data[pp.PARAMETERS]["transport"]["current_time"]
    
    if conv and current_time in store_time: 
        # Export values
        plot_1d_several(title_time=title_time[j], index=j)
        j+=1
    # end if
# end time-loop

print(f"Current time {current_time}")

# Save the 2x3 formated figure
plt.savefig(
    folder_name+"conc_mineral_equil_in_one.pdf", 
    bbox_inches="tight", 
    pad_inches=0.1
)
plt.show() 
