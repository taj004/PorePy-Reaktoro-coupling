#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 13:37:37 2023

@author: uw
    
Based on 2D simulation of Todaka et al 2004: Transport simulation in Onikobe 
  
"""
# %% Import the neseccary packages
import numpy as np
import porepy as pp
import reaktoro.reaktoro4py as rt

import equations
from solve_non_linear import solve_eqs
import update_param
import os
import constant_params
import reactive_transport
import rt_utills

import porepy_reaktoro_interface

# %% Initialise variables related to the domain and the chemistry 

# Start with the domain

# Define mdg

#%% Structured 
frac = np.array([
    [100, 600],
    [700, 700]    
    ]) # fracture from 100<x<600 and fixed at y=700

frac2 = np.array([
    [700, 1000],
    [800,  800]
    ])

frac3 = np.array([
    [800,  800],
    [500, 1000]
    ])

mdg = pp.meshing.cart_grid(fracs=[frac, frac2, frac3], 
                          nx=[50, 50],
                          physdims=[1500, 1500])


sd = mdg.subdomains(dim=mdg.dim_max())[0]
domain = {"xmin": 0, "xmax": np.max(sd.face_centers[0]),
          "ymin": 0, "ymax": np.max(sd.face_centers[1]),
          } # domain

#%% Unstructured

# def benchmark_2d_case_3(size: pp.number = 1) -> list[pp.LineFracture]:
#     """Return a list of fractures as used in case 3 of the 2d benchmark study by
#     Flemisch et al. 2018.

#     Parameters:
#         size: The side length of the domain.

#     Returns:
#         List of fractures.

#     """
#     points = [
#         np.array([[0.0500, 0.2200], [0.4160, 0.0624]]),
#         np.array([[0.0500, 0.2500], [0.2750, 0.1350]]),
#         np.array([[0.1500, 0.4500], [0.6300, 0.0900]]),
#         np.array([[0.1500, 0.4000], [0.9167, 0.5000]]),
#         np.array([[0.6500, 0.849723], [0.8333, 0.167625]]),
#         np.array([[0.7000, 0.849723], [0.2350, 0.167625]]),
#         np.array([[0.6000, 0.8500], [0.3800, 0.2675]]),
#         np.array([[0.3500, 0.8000], [0.9714, 0.7143]]),
#         np.array([[0.7500, 0.9500], [0.9574, 0.8155]]),
#         np.array([[0.1500, 0.4000], [0.8363, 0.9727]]),
#     ]
#     fractures = [pp.LineFracture(pts * size) for pts in points]
#     return fractures

# domain = {"xmin": 0, "xmax": 1500, "ymin": 0, "ymax": 1500}

# mesh_args = {
#     "mesh_size_bound": 40.0,
#     "mesh_size_min":   40.0,
#     "mesh_size_frac":  40.0
#     } 

# f_list = benchmark_2d_case_3(size=1500)
# network = pp.FractureNetwork2d(
#     fractures=f_list,
#     domain=pp.Domain(domain)
#     )
# mdg = network.mesh(mesh_args)

#%% Keywords
mass_kw = "mass"
chemistry_kw = "chemistry"
transport_kw = "transport"
flow_kw = "flow"

pressure = "pressure"
tot_var = "total_amount"
aq_var = "aq_amount"
temperature = "temperature"
tracer = "passive_tracer"

mortar_pressure = "mortar_pressure"
mortar_transport = "mortar_transport"
mortar_temperature_convection = "mortar_temperature_convection"
mortar_temperature_conduction = "mortar_temperature_conduction"
mortar_tracer = "mortar_tracer"

aq_species=rt.AqueousPhase().setActivityModel(rt.ActivityModelHKF())

mineral_species = [
    # Primary minerals
    "Quartz",
    "K-Feldspar",
    "Albite", 
    "Anorthite",
    "Diopside",
    "Hedenbergite",
    "Enstatite",
    "Ferrosilite",
    "Magnetite",
    
    # Secondary species
    "Clinochlore,14A",
    "Daphnite,14A",
    "Kaolinite",
    "Pyrophyllite",
    "Laumontite",
    "Wairakite",
    "Prehnite",
    "Clinozoisite",
    "Epidote",    
    "Pyrite",
    "Sphalerite",
    "Galena",
    "Calcite",
    "Anhydrite",
]

db = rt.SupcrtDatabase("supcrt16")

mineral_phase = rt.MineralPhases(mineral_species)

# The Palandri-Kharaka parameters
params = rt.Params.local("PalandriKharaka_customised_by_SIB.yaml")

chem_system = rt.ChemicalSystem(
    db,
    aq_species,
    mineral_phase,
    
    #--------------------------#

    rt.MineralReaction("Quartz").setRateModel(
        rt.ReactionRateModelPalandriKharaka(params)
        ),
    rt.MineralReaction("K-Feldspar").setRateModel(
        rt.ReactionRateModelPalandriKharaka(params)
        ),
    rt.MineralReaction("Albite").setRateModel(
        rt.ReactionRateModelPalandriKharaka(params)
        ),
    rt.MineralReaction("Anorthite").setRateModel(
        rt.ReactionRateModelPalandriKharaka(params)
        ),
    rt.MineralReaction("Diopside").setRateModel(
        rt.ReactionRateModelPalandriKharaka(params)
        ),
    rt.MineralReaction("Hedenbergite").setRateModel(
        rt.ReactionRateModelPalandriKharaka(params)
        ),
    rt.MineralReaction("Enstatite").setRateModel(
        rt.ReactionRateModelPalandriKharaka(params)
        ),
    rt.MineralReaction("Ferrosilite").setRateModel(
        rt.ReactionRateModelPalandriKharaka(params)
        ),
    rt.MineralReaction("Magnetite").setRateModel(
        rt.ReactionRateModelPalandriKharaka(params)
        ),
    rt.MineralReaction("Clinochlore,14A").setRateModel(
        rt.ReactionRateModelPalandriKharaka(params)
        ),
    rt.MineralReaction("Daphnite,14A").setRateModel(
        rt.ReactionRateModelPalandriKharaka(params)
        ),
    rt.MineralReaction("Kaolinite").setRateModel(
        rt.ReactionRateModelPalandriKharaka(params)
        ),
    rt.MineralReaction("Pyrophyllite").setRateModel(
        rt.ReactionRateModelPalandriKharaka(params)
        ),
    rt.MineralReaction("Laumontite").setRateModel(
        rt.ReactionRateModelPalandriKharaka(params)
        ),
    rt.MineralReaction("Wairakite").setRateModel(
        rt.ReactionRateModelPalandriKharaka(params)
        ),
    rt.MineralReaction("Prehnite").setRateModel(
        rt.ReactionRateModelPalandriKharaka(params)
        ),
    rt.MineralReaction("Clinozoisite").setRateModel(
        rt.ReactionRateModelPalandriKharaka(params)
        ),
    rt.MineralReaction("Epidote").setRateModel(
        rt.ReactionRateModelPalandriKharaka(params)
        ),
    rt.MineralReaction("Pyrite").setRateModel(
        rt.ReactionRateModelPalandriKharaka(params)
        ),
    rt.MineralReaction("Sphalerite").setRateModel(
        rt.ReactionRateModelPalandriKharaka(params)
        ),
    rt.MineralReaction("Galena").setRateModel(
        rt.ReactionRateModelPalandriKharaka(params)
        ),
    #--------------------------#
     
    rt.MineralSurface("Quartz",           14.77, "cm2/g"),
    rt.MineralSurface("K-Feldspar",        4.95, "cm2/g"),
    rt.MineralSurface("Albite",           22.32, "cm2/g"),
    rt.MineralSurface("Anorthite",         0.01, "cm2/g"),
    rt.MineralSurface("Diopside",          1.49, "cm2/g"),
    rt.MineralSurface("Hedenbergite",      0.98, "cm2/g"),
    rt.MineralSurface("Enstatite",         7.26, "cm2/g"),
    rt.MineralSurface("Ferrosilite",       4.93, "cm2/g"),
    rt.MineralSurface("Magnetite",         1.76, "cm2/g"),
    
    rt.MineralSurface("Clinochlore,14A", 100.00, "cm2/g"),
    rt.MineralSurface("Daphnite,14A",    100.00, "cm2/g"),
    rt.MineralSurface("Kaolinite",       100.00, "cm2/g"),
    rt.MineralSurface("Pyrophyllite",    100.00, "cm2/g"),
    
    rt.MineralSurface("Laumontite",       10.00, "cm2/g"),
    rt.MineralSurface("Wairakite",        10.00, "cm2/g"),
    rt.MineralSurface("Prehnite",         10.00, "cm2/g"),
    rt.MineralSurface("Clinozoisite",     10.00, "cm2/g"),
    rt.MineralSurface("Epidote",          10.00, "cm2/g"),
    rt.MineralSurface("Pyrite",            0.001, "cm2/g"),
    rt.MineralSurface("Sphalerite",        0.001, "cm2/g"),
    rt.MineralSurface("Galena",            0.001, "cm2/g"),

)

chemical_solver = porepy_reaktoro_interface.ChemicalSolver(
    chemical_system=chem_system, kinetic=True, use_ODML=False
)

#%% Initial chemistry
h2o = 550
chemical_solver.state.set("H2O(aq)", h2o, "mol")

chemical_solver.state.set("H+",       2.745e-6 , "mol") 
chemical_solver.state.set("SO4-2",    6.500e-3 , "mol")
chemical_solver.state.set("HCO3-",    1.372e-2 , "mol")
chemical_solver.state.set("SiO2(aq)", 8.498e-3 , "mol")
chemical_solver.state.set("Al+3",     5.656e-6 , "mol")
chemical_solver.state.set("Ca+2",     3.098e-3 , "mol")
chemical_solver.state.set("Mg+2",     5.992e-6 , "mol")
chemical_solver.state.set("Fe+2",     1.491e-6 , "mol")
chemical_solver.state.set("K+",       3.814e-3 , "mol")
chemical_solver.state.set("Na+",      3.470e-2 , "mol")
chemical_solver.state.set("Zn+2",     1.061e-7 , "mol")
chemical_solver.state.set("Pb+2",     1.00e-12 , "mol")
chemical_solver.state.set("O2(aq)",   1.724e-40, "mol") 

init_pressure = constant_params.ref_p()
init_temp = constant_params.ref_temp()
chemical_solver.state.pressure(init_pressure, "Pa")
chemical_solver.state.temperature(init_temp, "K")

init_chem_solv=chemical_solver.solve_chemistry(dt=1000) 

# Scale the mineral volume
chemical_solver.state.scalePhaseVolume("Quartz",       0.1477, "m3")
chemical_solver.state.scalePhaseVolume("K-Feldspar",   0.0495, "m3")
chemical_solver.state.scalePhaseVolume("Albite",       0.2232, "m3")
chemical_solver.state.scalePhaseVolume("Anorthite",    0.3145, "m3")
chemical_solver.state.scalePhaseVolume("Diopside",     0.0149, "m3")
chemical_solver.state.scalePhaseVolume("Hedenbergite", 0.0098, "m3")
chemical_solver.state.scalePhaseVolume("Enstatite",    0.0736, "m3")
chemical_solver.state.scalePhaseVolume("Ferrosilite",  0.0436, "m3")
chemical_solver.state.scalePhaseVolume("Magnetite",    0.0176, "m3")

# For possible extra lag; aleardy scaled minerals are unaffected
chemical_solver.state.scalePhaseVolume("Magnetite",    0.0176, "m3")

#%% Calulate initial (uniform) porosity and aperture
x = 0
minerals = []
surface_area_list = []
sd = mdg.subdomains(dim=1)[0]

for phase in chemical_solver.chem_system.phases():
    if phase.name() != "AqueousPhase":
        props = chemical_solver.state.props()
        minerals.append( props.phaseProps(phase.name()).volume().val() ) 
    # end if
# end phase-loop

mineral_sum = np.sum([e for e in minerals if e>1e-5])

# The porosity
porosity = 1.0 - mineral_sum / (1.0) 

# Scale aqueous phase
chemical_solver.scale_aqueous_phase_in_state(porosity)

# The aperture
aperture = constant_params.open_aperture() * porosity

#%% Setup for kinetic solver

options = rt.KineticsOptions()
options.optima.convergence.tolerance = 1e-6
options.optima.convergence.requires_at_least_one_iteration=False
options.optima.maxiters = 40

chemical_solver.solver.setOptions(options)

#%% Preparation for RT simulations

# Some total conc stuff we need
init_c = chemical_solver.state.componentAmounts().asarray()
init_precipitated = chemical_solver.state.speciesAmounts().asarray()[
    chemical_solver.solid_indices
]

a_species = chemical_solver.state.speciesAmounts().asarray()[
    chemical_solver.fluid_indices
]

init_aq = chemical_solver.chem_system.formulaMatrix()[
    :, chemical_solver.fluid_indices
].dot(a_species)

init_mineral = chemical_solver.chem_system.formulaMatrix()[
    :, chemical_solver.solid_indices
    ].dot(init_precipitated)

is_close = np.abs(init_c - (init_mineral + init_aq)) < 1e-5

# Multiple states for RT simulations
chemical_solver.multiple_states(mdg.num_subdomain_cells())

#%% The chemistry at the boundary
bc_chemistry = porepy_reaktoro_interface.ChemicalSolver(
    chemical_system=chem_system, kinetic=True, use_ODML=False
)

bc_temp = init_temp - 40
bc_pressure = 6 * constant_params.ref_p()
bc_chemistry.state.pressure(bc_pressure, "Pa")
bc_chemistry.state.temperature(bc_temp, "K")

bc_chemistry.state.set("H2O(aq)"  , h2o, "mol")

bc_chemistry.state.set("H+",       3.863e-4, "mol") 
bc_chemistry.state.set("SO4-2",    1.150e-2, "mol")
bc_chemistry.state.set("HCO3-",    1.782e-2, "mol")
bc_chemistry.state.set("SiO2(aq)", 8.428e-3, "mol")
bc_chemistry.state.set("Al+3",     9.577e-6, "mol")
bc_chemistry.state.set("Ca+2",     1.127e-2, "mol")
bc_chemistry.state.set("Mg+2",     1.143e-3, "mol")
bc_chemistry.state.set("Fe+2",     2.214e-3, "mol")
bc_chemistry.state.set("K+",       7.412e-3, "mol")
bc_chemistry.state.set("Na+",      6.228e-2, "mol")
bc_chemistry.state.set("Zn+2",     3.376e-5, "mol")
bc_chemistry.state.set("Pb+2",     5.899e-6, "mol")
bc_chemistry.state.set("O2(aq)",   2.344e-40, "mol") 

restriction = rt.EquilibriumRestrictions(bc_chemistry.chem_system)
for i in range(len(mineral_species)):
    restriction.cannotReact(mineral_species[i])

bc_solver = rt.EquilibriumSolver(bc_chemistry.chem_system)
bb = bc_solver.solve(bc_chemistry.state, restrictions=restriction)

bc_chemistry.scale_aqueous_phase_in_state(porosity)
bc_c = bc_chemistry.state.componentAmounts().asarray()

#%% Loop over the mdg, and set initial and default data

for sd, d in mdg.subdomains(return_data=True):
 
    # ---- Initialisation
    
    # Initialize the primary variable dictionaries
    d[pp.PRIMARY_VARIABLES] = {
        pressure: {"cells": 1},
        tot_var: {"cells": chemical_solver.num_components},
        aq_var: {"cells": chemical_solver.num_components},
        temperature: {"cells": 1},
        tracer: {"cells": 1}
    }

    # Initialize a state
    pp.set_state(d)

    unity = np.ones(sd.num_cells)

    # --------------------------------- #
    
    # ---- Grid related
    
    specific_volume = np.power(aperture, mdg.dim_max()-sd.dim)
    
    # permeability
    if sd.dim == mdg.dim_max():    
        Kxx = 2e-14 * unity.copy() 
    else:
        Kxx = np.power(aperture,2) * unity / 12
    # end if-else 
    
    perm = pp.SecondOrderTensor( 
        kxx= Kxx * specific_volume / constant_params.dynamic_viscosity() 
        )
    
    # --------------------------------- #

    # ---- Boundary conditions
    bound_faces = sd.tags["domain_boundary_faces"].nonzero()[0]

    if bound_faces.size != 0:

        bound_faces_centers = sd.face_centers[:, bound_faces]

        # Neumann faces
        neu_faces = np.array(["neu"] * bound_faces.size)
        # The global boundaries 
        left = bound_faces_centers[0, :] < domain["xmin"] + 1e-4
        right = bound_faces_centers[0, :] > domain["xmax"] - 1e-4
        top = bound_faces_centers[1, :] > domain["ymax"] - 1e-4
        bottom = bound_faces_centers[1, :] < domain["ymin"] + 1e-4
              
        # The BC labels for the flow problem
        labels_for_flow = neu_faces.copy()
        labels_for_flow[right] = "dir"
        labels_for_flow[left] = "dir"
        
        bound_for_flow = pp.BoundaryCondition(
            sd, faces=bound_faces, cond=labels_for_flow
            )
        
        # Set the BC values for flow
        bc_values_for_flow = np.zeros(sd.num_faces)

        bc_values_for_flow[bound_faces[right]] = init_pressure
        bc_values_for_flow[bound_faces[left]] = bc_pressure
        
        # The boundary conditions for solute transport. 
        labels_for_transport = neu_faces.copy()
        labels_for_transport[right] = "dir"
        labels_for_transport[left] = "dir"
        bound_for_transport = pp.BoundaryCondition(
            sd, faces=bound_faces, cond=labels_for_transport
        )

        # Set bc values. Note, at each face we have num_aq_components
        expanded_right = pp.fvutils.expand_indices_nd(
            bound_faces[right], 
            chemical_solver.num_components
        )
        
        expanded_left = pp.fvutils.expand_indices_nd(
            bound_faces[left], 
            chemical_solver.num_components
        )
        
        bc_values_for_transport = np.zeros(sd.num_faces * chemical_solver.num_components)

        bc_values_for_transport[expanded_right] = np.tile(
            init_c, bound_faces[right].size
            )
   
        bc_values_for_transport[expanded_left] = np.tile(
            bc_c, bound_faces[left].size
        )

        # Boundary conditions for temperature
        labels_for_temp = neu_faces.copy()
        labels_for_temp[np.logical_or(left, right)] = "dir"
        
        bound_for_temp = pp.BoundaryCondition(
            sd, faces=bound_faces, cond=labels_for_temp
        )

        # The bc values
        bc_values_for_temp = np.zeros(sd.num_faces)
        bc_values_for_temp[bound_faces[left]] = bc_temp
        bc_values_for_temp[bound_faces[right]] = init_temp


        # Boundary conditions for the passive tracer
        labels_for_tracer = neu_faces.copy()
        labels_for_tracer[left] = "dir"
        labels_for_tracer[right] = "dir"
        bound_for_tracer = pp.BoundaryCondition(
            sd, faces=bound_faces, cond=labels_for_tracer
        )

        # The values
        bc_values_for_tracer = np.zeros(sd.num_faces)
        bc_values_for_tracer[bound_faces[left]] = 1.0 
        
    else: # Internal boundaries
            
        bc_values_for_flow = np.zeros(sd.num_faces)
        bound_for_flow = pp.BoundaryCondition(sd) 
        
        bc_values_for_temp = np.zeros(sd.num_faces)
        bound_for_temp = pp.BoundaryCondition(sd) 
        
        bc_values_for_transport = np.zeros(chemical_solver.num_components * sd.num_faces)
        bound_for_transport = pp.BoundaryCondition(sd) 
        
        bc_values_for_tracer = np.zeros(sd.num_faces)
        bound_for_tracer = pp.BoundaryCondition(sd)       
    # end if
    
    # ---- Some extra stuff
    # Initial guess for Darcy flux
    init_darcy_flux = np.zeros(sd.num_faces)

    # Heat capacity and conduction
    solid_density = 2750
    heat_capacity =  (
        porosity * constant_params.specific_heat_capacity_fluid() * update_param.rho() +
        (1-porosity) * constant_params.specific_heat_capacity_solid() * solid_density
        
        )  
    conduction = (
        porosity * constant_params.fluid_conduction() +
        (1-porosity) * constant_params.solid_conduction()
        ) 
    # --------------------------------- #

    # ---- Set the values in dictionaries
    mass_data = {
        "porosity": porosity.copy() * unity,
        "aperture": aperture * unity,
        "specific_volume": specific_volume * unity,
        "mass_weight": specific_volume * unity,
        "num_components": chemical_solver.num_components,
    }

    flow_data = {
        "mass_weight": porosity * specific_volume * unity,
        "bc_values": bc_values_for_flow,
        "bc": bound_for_flow,
        "permeability": Kxx * specific_volume * unity/ constant_params.dynamic_viscosity(),
        "second_order_tensor": perm,
        "darcy_flux": np.abs(init_darcy_flux),
    }

    transport_data = {
        "bc_values": bc_values_for_transport,
        "bc": bound_for_transport,
        "num_components": chemical_solver.num_components,
        "darcy_flux": init_darcy_flux,
    }

    temp_data = {
        "mass_weight": heat_capacity * specific_volume * unity,
        "solid_density": solid_density,
        "bc_values": bc_values_for_temp,
        "bc": bound_for_temp,
        "darcy_flux": init_darcy_flux,
        "second_order_tensor": pp.SecondOrderTensor(
            conduction * specific_volume * unity
            ),
    }

    passive_tracer_data = {
        "bc_values": bc_values_for_tracer,
        "bc": bound_for_tracer,
        "darcy_flux": init_darcy_flux,
        "mass_weight": porosity * specific_volume * unity,
    }

    # The Initial values also serve as reference values
    reference_data = {
        "porosity": porosity.copy(),
        "permeability": Kxx.copy(),
        "mass_weight": porosity.copy(),
    }

    # --------------------------------- #

    # Set the parameters in the dictionary
    # Treat this different to make sure parameter dictionary is constructed
    d = pp.initialize_data(sd, d, mass_kw, mass_data)

    d[pp.PARAMETERS][flow_kw] = flow_data
    d[pp.DISCRETIZATION_MATRICES][flow_kw] = {}
    d[pp.PARAMETERS][transport_kw] = transport_data
    d[pp.DISCRETIZATION_MATRICES][transport_kw] = {}
    d[pp.PARAMETERS][temperature] = temp_data
    d[pp.DISCRETIZATION_MATRICES][temperature] = {}
    d[pp.PARAMETERS][tracer] = passive_tracer_data
    d[pp.DISCRETIZATION_MATRICES][tracer] = {}

    d[pp.PARAMETERS][chemical_solver.chemical_name] = {}

    # The reference values
    d[pp.PARAMETERS]["reference"] = reference_data

    # Set some data only in the highest dimension, in order to avoid techical issues later on
    if sd.dim == mdg.dim_max():

        d[pp.PARAMETERS][transport_kw].update(
            {
                "time_step": 1000,  # initial time step, in s,
                "current_time": 0,
                "final_time": 200 * pp.YEAR,
                "constant_time_step": False,
                "update_perm": True
            }
        )
        d[pp.PARAMETERS]["grid_params"] = {}
        d[pp.PARAMETERS]["previous_time_step"] = {"time_step": []}
        d[pp.PARAMETERS]["previous_newton_iteration"] = {
            "Number_of_Newton_iterations": [],
            "AD_full_flux" : pp.ad.DenseArray(init_darcy_flux),
            "AD_edge_flux": pp.ad.DenseArray(np.ones(mdg.num_interface_cells()))
        }
    # end if

    # --------------------------------- #

    # ---- Set state values

    # First for the concentrations, replicated versions of the equilibrium.
    total_amount = np.tile(init_c, sd.num_cells)
    aq_amount = np.tile(init_aq, sd.num_cells)


    d[pp.STATE].update(
        {
            "dimension": sd.dim * unity,
            pressure: init_pressure * unity, # use 0 for incompressible case
            tot_var: total_amount.copy(),
            aq_var: aq_amount.copy(),
            temperature: init_temp * unity,
            tracer: 0 * unity ,
    
            pp.ITERATE: {
                pressure: init_pressure * unity ,
                tot_var: total_amount.copy(),
                aq_var: aq_amount.copy(),
                temperature: init_temp * unity, 
                tracer: 0 * unity ,
            },
        }
    )
        
# end sd,d-loop

#%% The interfaces
for e,d in mdg.interfaces(return_data=True):
  
    # Set state
    pp.set_state(d)
    pp.set_iterate(d)
    
    # Primary variables
    d[pp.PRIMARY_VARIABLES] = {
        mortar_pressure: {"cells": 1},
        mortar_transport: {"cells": chemical_solver.num_components},
        mortar_temperature_convection: {"cells": 1},
        mortar_temperature_conduction: {"cells": 1},
        mortar_tracer: {"cells": 1}
        }
    
    unity = np.ones(e.num_cells)
    
    edge_init_c = np.tile(init_aq, e.num_cells)
    
    # Initial values
    vals = {
        mortar_pressure: 0.0 * unity,
        mortar_temperature_convection: 0.0 * unity,
        mortar_temperature_conduction: 0.0 * unity,
        mortar_transport: 0.0 * edge_init_c,
        mortar_tracer: 0.0 * unity 
        }
    
    # Set to state
    d[pp.STATE].update(vals)
    d[pp.STATE][pp.ITERATE].update(vals)
    
    # Set the parameter dictionary
    flow_params = {"darcy_flux": 0.0 * unity,
                   "normal_diffusivity": 1e3 * unity}
    
    transport_params = {"darcy_flux": 0.0 * unity,
                        "num_components": chemical_solver.num_components}
    
    temp_params = {"darcy_flux": 0.0 * unity,
                  "normal_diffusivity": constant_params.fluid_conduction() * unity / aperture}
    
    tracer_params = {"darcy_flux": 0.0 * unity}
    
    pp.initialize_data(grid=e, data=d, keyword="flow", specified_parameters=flow_params)
       
     
    d[pp.PARAMETERS][transport_kw] = transport_params
    d[pp.DISCRETIZATION_MATRICES][transport_kw] = {}
       
    d[pp.PARAMETERS][temperature] = temp_params
    d[pp.DISCRETIZATION_MATRICES][temperature] = {}
    
    d[pp.PARAMETERS][tracer] = tracer_params
    d[pp.DISCRETIZATION_MATRICES][tracer] = {}
    
# end interface-loop

# Set the conductive normal fluxes
update_param.update_interface(mdg)

#%% The data in the highest dimension
sd = mdg.subdomains(dim=mdg.dim_max())[0]
data = mdg.subdomain_data(sd)

# set initial chemical values in pp.parameters
chemical_solver.set_species_in_PorePy_dict_from_state(mdg)

#%% Aspects for the transport calculations

# Equation system
equation_system = pp.ad.EquationSystem(mdg)

# and the initial equations
eqs = equations.EquationConstruction(equation_system=equation_system)

rt_eqs = reactive_transport.ReactiveTransport(
    pde_solver=eqs, chemical_solver=chemical_solver
    )

#%% Prepere for exporting

# Make folder

if hasattr(sd, "cart_dims"):
    folder_name = "pictures/simulation_4_structured/"
else:
    folder_name = "pictures/simulation_4_unstructured/"
# end if

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

export = pp.Exporter(mdg, file_name="vars", folder_name=folder_name)

fields = [s.name() for s in chemical_solver.chem_system.species()]
fields.append("passive_tracer")
fields.append("pressure") 
fields.append("temperature")
fields.append("porosity")
fields.append("aperture")
fields.append("ratio_perm")

store_time = np.array([20, 200, 201]) * pp.YEAR # The last one is for safty 
    
#%% 
current_time = data[pp.PARAMETERS]["transport"]["current_time"]
time_step = data[pp.PARAMETERS]["transport"]["time_step"]
final_time = data[pp.PARAMETERS]["transport"]["final_time"]
j=0

#%%
while current_time < final_time :
    
    if current_time < pp.DAY:
        print(f"Current time is second {current_time}")
    elif current_time >= 0 and current_time < pp.YEAR:
        print(f"Current time is day {current_time/pp.DAY}")
    else:
        print(f"Current time is year {current_time/pp.YEAR}")
    # end if-else
    
    # Initialise equations
    eqs.get_flow_eqs(equation_system, iterate=False)

    eqs.get_solute_transport_CC_eqs(
        equation_system, 
        iterate=False,
        U_solid=rt_eqs.chemical_solver.element_values(
            mdg=rt_eqs.pde_solver.equation_system.mdg, 
            inds=rt_eqs.chemical_solver.solid_indices
            )
        )
    
    eqs.get_temperature_eqs(equation_system, iterate=False) 

    eqs.get_tracer_eqs(equation_system, iterate=False)
    
    conv = solve_eqs(rt_eqs, solution_strategy="sequential")
    current_time = data[pp.PARAMETERS]["transport"]["current_time"]
   
    # export
    if conv and np.abs(current_time-store_time[j]) < 0.9 * pp.YEAR :
        export.write_vtu(data=fields, time_step=current_time) 
        rt_utills.save_mdg(mdg, folder_name=folder_name)
        j+=1
# end time-loop

if current_time < pp.DAY:
    print(f"Current time is second {current_time}")
elif current_time >= 0 and current_time < pp.YEAR:
    print(f"Current time is day {current_time/pp.DAY}")
else:
    print(f"Current time is year {current_time/pp.YEAR}")
# end if

# Final export
export.write_vtu(data=fields, time_step=current_time/pp.YEAR) 
rt_utills.save_mdg(mdg, folder_name=folder_name)
