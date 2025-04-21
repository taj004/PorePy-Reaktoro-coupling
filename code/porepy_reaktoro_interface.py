"""A Porepy and Reaktoro interface

   After preforming a Newton iteration, sequential iteration 
   or advancing forward in time with PorePy, feed the information 
   (pressure, temperature and element) here, to calculate the 
   chemical state with Reaktoro and return values to PorePy
            
Note: 
    
    Reaktoro always K and Bar, while we might using something else.
    The easiest is to always use K and Pa, 
    and let Reaktoro work on converting the units.

      
"""

from typing import Optional, Union

import numpy as np
import porepy as pp
import reaktoro.reaktoro4py as rt

def species_indices(chemical_system):
    """Get the various number indices for species the different phases

    Since I have not found a way to get all fluid and solid
    indices for a particlular system, I made this.

    """
    
    indices = []
    new_phase_number = 0

    # Loop over the phases to get the indices, in a global sense
    for phase in chemical_system.phases():

        # Loop over names and store the indices
        for name in phase.species():
            indices.append(phase.species().index(name.name()) + new_phase_number)
        # end loop

        # Update to new phase
        new_phase_number += phase.species().size()
    # end loop

    # Next, splitt according to aqueous, gas and mineral phases    
    solid_indices = [] # Treat the solid (mineral) species different,
                       # since Reaktoro considers each solid as a singe phase
    
    # Prepare a dictionary to return
    dict_to_return = {}

    phase_ind = 0
    for phase in chemical_system.phases():
        part = slice(phase_ind, phase_ind + phase.species().size())
        if phase.name() == "AqueousPhase":
            dict_to_return.update({phase.name(): indices[part] })
        elif phase.name() == "GaseousPhase":
            dict_to_return.update({phase.name(): indices[part] })
        else:
            solid_indices.append(indices[part][0]) 
            # use [0], since Reaktoro considers each mineral as a single phase
        # if-else

        # Update the phase
        phase_ind += phase.species().size()
    # end phase-loop
    
    if len(solid_indices) > 0:
        dict_to_return.update({"SolidPhase": solid_indices}) 
    # end if
     
    return dict_to_return


class ChemicalSolver:
    """
    The PorePy-Reaktoro class
    
    
    Parameters:

                chemical system: The system of the chemical problem investigated

                kinetic (optional): ``default=None``

                    if kinetic reactions are considered, and thus the
                    kinetic solver should be used

                use_ODML (optional): ``default=None``

                    if the on-demand machine learning solver should be used
                    
    """
    
    def __init__(
        self,
        chemical_system: rt.ChemicalSystem,
        kinetic: bool = False,
        use_ODML: bool = False,
    ):

        # The chemical system
        self.chem_system = chemical_system

        # Choose chemical solver
        # NOTE: The smart solver should be used with care, as the usage is currently (30.3.23)
        # not described in the Reaktoro's tutorials

        # The machine learning strategy solver
        if use_ODML:
            if kinetic:
                raise ValueError("ODML solver is to be implemetent")
            else:
                self.solver = rt.SmartEquilibriumSolver(self.chem_system)
            # end if
        # end if

        # Conventional solver
        else:
            if kinetic:
                self.solver = rt.KineticsSolver(self.chem_system)
            else:  # equilibrium
                self.solver = rt.EquilibriumSolver(self.chem_system)
            # end kinetic-if-else
        # end ODML-if-else

        # Chemical state
        self.state = rt.ChemicalState(self.chem_system)

        self.chemical_name = "chemical_variables"
        
        self.conv = np.array([0])        
        
        # Number of various aspects
        self.num_species = self.state.speciesAmounts().asarray().size
        self.num_components = self.state.componentAmounts().asarray().size
        self.num_elements = self.state.elementAmounts().asarray().size
        
        # The chemical elements are C, H, etc,
        # and the components are elements pluss charge
        assert self.num_elements == self.num_components - 1
    
        # Get number of chemical species, according to phases
        species_inds = species_indices(self.chem_system)
        
        for key in species_inds:
            if key == "AqueousPhase":
                self.fluid_indices = species_inds["AqueousPhase"]
            if key == "GaseousPhase":
                self.gas_indices = species_inds["GaseousPhase"]
            if key == "SolidPhase" : 
                self.solid_indices = species_inds["SolidPhase"]
            # end ifs
        
          
    # end init
        
    # ---- From PorePy to Reaktoro
    
    def set_species_in_states_from_PorePy_dict(self, mdg: pp.MixedDimensionalGrid):
        """ Set values in `Reaktoro` `STATES` with values from PP 
        
        The main usage of this function is if a simulation is paused at some time in 
        reactive transport simulation and a restart is to be done
        """
        
        # Global indexation
        grid_index = 0
        for sd, d in mdg.subdomains(return_data=True):

            for index in range(sd.num_cells):
                mesh_point_index = index + grid_index
                for species in self.chem_system.species():
                    self.states[mesh_point_index].set(
                        species.name(), 
                        d[pp.PARAMETERS][self.chemical_name][species.name()][index],
                        "mol"
                        )
                # end species-loop
            # end index-loop

            # For next subdomain
            grid_index += sd.num_cells
        # end mdg-loop
    
    def _extract_values(
            self,
            data: dict,
            key: str,
            from_iterate: bool = False,
            inds: Optional[Union[int, np.ndarray]] = None 
            )-> np.ndarray: 
        """Get `key` values from the `PorePy` state dictionary 'data'

        Parameters:

            data:

                A ``PorePy`` dictionary

            key:

                String representing the values

            from_values (optional): ``default=False``

                If the values are from ``pp.ITERATE`` or ``pp.STATE`` (default)

            inds (optional): ``default=None``

                Particular indices for the problem

        """
        if from_iterate:
            val = data[pp.STATE][pp.ITERATE][key]
        else:
            val = data[pp.STATE][key]
        # end if-else

        # If a small subset is desired
        if inds is not None:
            val = val[inds]
        # end if
        
        return val

    def multiple_states(self, num_points: int):
        """Compute 'num_points' replica of `state` for RT modelling

        Parameters:

            num_points: Number of replica of ``STATE`` to be created

        """

        self.states = [self.state.clone() for _ in range(num_points)]

    # ---- Use Reakoto 

    def scale_aqueous_phase_in_state(self, porosity):
        """Scale the (single) aqueous phase in a state by porosity"""
        self.state.scalePhaseVolume("AqueousPhase", porosity, "m3")
        
    def scale_aqueous_phases_in_states(self, mdg: pp.MixedDimensionalGrid):
        """ Scale aqueous phases by the local grid cell porosity 
        
        Mostly used to re-scale the aqueous phases in states in a RT simulation
        """
        
        # Global index
        grid_index = 0
        
        for sd, d in mdg.subdomains(return_data=True):
            
            phi = d[pp.STATE][pp.ITERATE]["porosity"] 
            
            for index in range(sd.num_cells):
                mesh_point_index = index + grid_index
                
                # The scaling
                self.states[mesh_point_index].scalePhaseVolume(
                    "AqueousPhase", phi[index], "m3"
                    ) 
            # end index-loop
        # end index-loop
        
    def scale_mineral_phases_in_states(self):
        """Scale the mineral phases in the states by their 
        current mineral volume value
        """
        
        # Loop over the states
        for i in range(len(self.states)):
            
            # State nr.i
            state = self.states[i]
            
            # Loop over the phases
            for phase in self.chem_system.phases():
                
                if phase.name != "AqueousPhase" and phase.name() != "GaseousPhase":
                    # Mineral volume value
                    mineral_value = state.props().phaseProps(phase.name()).volume().val()
                    
                    # Rescale the phase value
                    state.scalePhaseVolume(phase.name(), mineral_value, "m3")
                    
                    # It can happen that the scaling is not performed by
                    # the line above, for whatever reason.
                    # This is to ensure that the mineral phase is indeed scaled;
                    # If the phase is scaled by the above, this line don't do anything 
                    state.scalePhaseVolume(phase.name(), mineral_value, "m3")
                # end if
            # end phase-loop
        # end i-loop
                    
    def solve_chemistry(
        self,
        cond: Optional[rt.EquilibriumConditions] = None,
        mesh_point_index: Optional[int] = None,
        dt: Optional[float] = None,
    ):
        """Solve equilibrium problem, given a total element U


        Parameters:
            cond (optional): ``default=None``

                Specifically conditions for the chemical reactions

            mesh_point_index (optional): ``default=None``

                The index of a mesh point

            dt (Optional): ``default=None``

        Returns:
            conv: If the equilibrium solver converged

        """
       
        # The if-tree:
        #    i) check which solver  
        #   ii) mesh point stuff below
        #  iii) cond
        
        # Equilibrium solver
        if isinstance(self.solver, 
                      (rt.SmartEquilibriumSolver, rt.EquilibriumSolver)):
            
            if mesh_point_index is None:
                if cond is None:                        
                    res = self.solver.solve(self.state)
                else:
                    res = self.solver.solve(self.state, cond)
                # end cond-if-else
            else:
                if cond is None:
                    res = self.solver.solve(self.states[mesh_point_index])
                else:                        
                    res = self.solver.solve(self.states[mesh_point_index], cond)
                # end cond-if-else
            # end mesh_point if-else
            
        elif isinstance(self.solver, rt.SmartKineticsSolver):
            raise ValueError("To be implenented")
    
        # Kinetic solver
        elif isinstance(self.solver, rt.KineticsSolver):
            # Check that the time step is not nonesense
            if dt is None:
                raise ValueError("Proper time step is required for kinetic calculations")
            # end if

            if mesh_point_index is None:
                if cond is None:
                    res = self.solver.solve(self.state, dt)
                else:
                    res = self.solver.solve(self.state, dt, cond)
                # end cond-if-else
            else:
                if cond is None:
                    res = self.solver.solve(self.states[mesh_point_index], dt)
                else:
                    res = self.solver.solve(self.states[mesh_point_index], dt, cond)
                # end cond-if-else
            # end mesh_point-if-else
        
        # Return whether or not the solver converged 
        conv = res.succeeded()
        return conv

    def set_chemical_values(
        self,
        total_amount: np.ndarray,
    ):
        """Set the chemical values for chemical calculations, via the 
        Reaktoro EquilibirumConditions given a total amount U
     
        Parameters:
            U: The total amount of elements

        Returns:
            cond: Problem conditions

        """
       
        cond = rt.EquilibriumConditions(self.chem_system)
        cond.setInitialComponentAmounts(total_amount)

        return cond

    def _set_cond(
        self,
        data,
        cond,
        subdomain_index: int,
        mesh_point_index: Optional[int]=None,
        set_pressure: bool = False,
        set_temperature: bool = False,
        from_iterate: bool = False,
    ):
        """Conditions (i.e. temperature and pressure) for the problem
        The values from the ``Porepy STATE/ITERATE`` dictionary

        Parameters:
            data: A ``PorePy`` dictionary
            
            cond: a ``Reaktoro EquilibriumConditions``
                
            subdomain_index (int): Cell index of subdomain i

            mesh_point_index (optional): ``default=False``

                The index of the mesh point (typically in a RT simulation)

            set_pressure (optional): ``default=False``

                Set the pressure for the equilibrium calculations

            set_temperature (optional): ``default=False``

                Set the temperature for the equilibrium calculations

            from_iterate (optional): ``default=False``

                If the pressure, temperature and chemical values are
                from ``pp.STATE`` or ``pp.ITERATE``

        """

        if mesh_point_index is None:
            if set_pressure:
                pressure = self._extract_values(
                    data, "pressure", from_iterate=from_iterate, inds=subdomain_index
                )
                self.state.setPressure(pressure, "Pa")
            if set_temperature:
                temperature = self._extract_values(
                    data, "temperature", from_iterate=from_iterate, inds=subdomain_index
                )
                self.state.setTemperature(temperature, "K")
            # end many ifs

        else:
            if set_pressure:
                pressure = self._extract_values(
                    data, "pressure", from_iterate=from_iterate, inds=subdomain_index
                )
                self.states[mesh_point_index].setPressure(pressure, "Pa")
                cond.pressure(pressure, "Pa")
            if set_temperature:
                temperature = self._extract_values(
                    data, "temperature", from_iterate=from_iterate, inds=subdomain_index
                )
                self.states[mesh_point_index].setTemperature(temperature, "K")
                cond.temperature(temperature, "K")
            # end many ifs
        # end if-else
        
        return cond
    
    def return_element(self, 
                       data: dict, 
                       mesh_point_index: Optional[int] = None, 
                       subdomain_index: Optional[int] = None):
        """Return the calculated chemical values.

        Parameters:
            d: A ``PorePy`` dictionary

            mesh_point_index (optional): ``default=None``
                           an `ìnt``  represents the cell index of subdomain nr i

            subdomain_index (optional): ``default=None``
                           an `ìnt``  represents the cell index of subdomain nr i

        """

        # Loop over the chemical names
        for species in self.chem_system.species():

            if mesh_point_index is None:
                species_val = self.state.speciesAmount(species.name()).val()
                data[pp.PARAMETERS][self.chemical_name][species.name()] = species_val
            else:
                species_val = (
                    self.states[mesh_point_index].speciesAmount(species.name()).val()
                )

                data[pp.PARAMETERS][self.chemical_name][
                    species.name()
                    ][subdomain_index] = species_val
            # end if-else
        # end s-loop
        
    def solver_preparation_and_solve(self, mdg: pp.MixedDimensionalGrid):
        """Prepare to solve the chemical reactions, and solve"""

        # Get the advective time step
        dt_adv = mdg.subdomain_data(
            mdg.subdomains(dim=mdg.dim_max())[0]
            )[pp.PARAMETERS]["transport"]["time_step"]
        
        # Cellwise reactive time step
        dt_reac = np.ones(mdg.num_subdomain_cells())

        # Equilibrium solver, simply solve
        if isinstance(self.solver, 
                      (rt.EquilibriumSolver, rt.SmartEquilibriumSolver)
                      ):

            self.cellwise_solving(
                mdg,
                get_pressure=True,
                get_temperature=True,
                from_iterate=True,
                dt=dt_reac
                )

        # Kinetic problem, a loop where the reactive time step
        # decreases if not converged
        else:
            
            iter_step = 0 # Number of kinetic steps
            conv_all_cells = False
            
            # Set an initial reactive time step
          
            global_index = 0
            for sd, d in mdg.subdomains(return_data=True):
                                              
                for index in range(sd.num_cells):
                    mesh_point_index = index + global_index  
                    dt_reac[mesh_point_index] = self._initial_reactive_dt(
                        dt_adv=dt_adv
                        )
                # end index loop
                global_index += sd.num_cells
            # end mdg-loop
            
            while conv_all_cells is False:
                self.cellwise_solving(
                    mdg,
                    get_pressure=True,
                    get_temperature=True,
                    from_iterate=True,
                    dt=dt_reac,    
                    )
                
                # Update iteration counter
                iter_step += 1

                # Redo cells where the solver did not converge
                # with a smaller reactive time step
                if False in self.conv:
                    # Get the indices that should be adjusted 
                    all_inds = np.arange(0, dt_reac.size)
                    where_conv = np.where(self.conv)[0]
                    where_is_false = np.setdiff1d(all_inds, where_conv)
                    
                    dt_reac[where_is_false] *= 1e-3
                else:    
                    conv_all_cells = True
                # end if-else

                # Stop if too many iterations have been used
                if iter_step == 1:
                    break
            # end while

        # end if-else

        return self.conv

    def _initial_reactive_dt(
            self, 
            dt_adv
            ):
        """Calculate an inital reactive time step
        """
        
        dt_reak = np.clip(a=dt_adv, a_min=500, a_max=pp.DAY*3)

        return dt_reak 

    def _check_for_nan(self, data: dict, mesh_point_index=None, subdomain_index=None):
        """if the chemical solver do not converge,
        the returned state variables might be nan.
        
        Note that this functionallity is (mostly) used in reactive transport
        simulations. Therefore, mesh_point_index and subdomain_index are required.
        If either are provided, an error is raised 
                
        """

        if mesh_point_index is None:
            raise ValueError("The mesh point index must be provided")
        # end if
        
        if subdomain_index is None:
            raise ValueError("Subdomain index must be provided")
        # end if

        # Loop over the chemical names
        for species in self.chem_system.species():
            
            species_val = self.states[mesh_point_index].speciesAmount(species.name()).val()
            if np.isnan(species_val):
              
                self.states[mesh_point_index].set(
                    species.name(),
                    data[
                        pp.PARAMETERS
                        ][self.chemical_name][species.name()][subdomain_index],
                    "mol",
                ) 
                # Note that the value from `d` has unit mol/m3, but Reaktoro might
                # interpret it as mol or mol/kg.
                # This can possibly cause something funny in the calculations
            # end if

        # end s-loop

        # Check pressure and temperature
        pressure = self.states[mesh_point_index].pressure().val()
        temperature = self.states[mesh_point_index].temperature().val()
        
        if np.isnan(pressure):
            self.states[mesh_point_index].pressure(
                data[pp.STATE][pp.ITERATE]["pressure"][subdomain_index], "Pa"
            )
        # end if
        if np.isnan(temperature):
            self.states[mesh_point_index].temperature(
                data[pp.STATE][pp.ITERATE]["temperature"][subdomain_index], "K"
            )
        # end if
        
    def cellwise_solving(
        self,
        mdg: pp.MixedDimensionalGrid,
        get_pressure: bool = True,
        get_temperature: bool = True,
        from_iterate: bool = True,
        dt: Optional[Union[float, int]] = None,
    ):

        """Loop over the mixed-dimensinal grid,
        solve the chemical reaction problem cell indices which has not converged
        and return the values

        Parameters:
            mdg: A ``PorePy`` mixed-dimensional grid

            get_pressure (optional): ``default=False``

                if pressure should be used

            get_temperaturee (optional): ``default=False``

                 if temperature should be used

            from_iterate (optional): ``default=False``
                if the above values are extracted for ``pp.STATE`` or ``pp.ITERATE``

        """

        # Raise an error if the intension is to calculate on a mixed-dimensional grid,
        # but with only one state
        if hasattr(self, "states") is False:
            raise ValueError("Did you forget to use multiple_states?")
        # end if
        
        # For just equilibrium reactions, the input reactive time step 
        # can be just the advective time step, as float. To make the code as 
        # `general' as possible, make an array
        if dt is None:
            dt=1.0
        # end if
        if not isinstance(dt, np.ndarray):
            dt *= np.ones(len(self.states))
        # end if
        
        # For a global indexing for both PorePy and Reaktoro
        grid_cell = 0
        
        # Loop over mixed-dim grid
        for sd, d in mdg.subdomains(return_data=True):
            

            # Solve cell-wise
            for index in range(sd.num_cells):
                inds = slice(
                    index * self.num_components, (index + 1) * self.num_components
                )

                U = self._extract_values(
                    d, key="total_amount", from_iterate=from_iterate, inds=inds
                )
                                
                mesh_point_index = index + grid_cell
               
                # Solve if not already converged
                if not self.conv[mesh_point_index]:
                   
                    cond = self.set_chemical_values(U)

                    # Set pressure and temperature
                    cond = self._set_cond(
                        data=d,
                        cond=cond,
                        subdomain_index=index,
                        mesh_point_index=mesh_point_index,
                        set_pressure=get_pressure,
                        set_temperature=get_temperature,
                        from_iterate=from_iterate,
                    )
                    
                    self.conv[mesh_point_index] = self.solve_chemistry(
                        cond=cond, 
                        mesh_point_index=mesh_point_index, 
                        dt=dt[mesh_point_index] 
                    )
                    
                elif self.conv[mesh_point_index]:
                    continue
                else :
                    msg = (
                        "You have converged and not converged simultaneously. "
                        "Well done!"
                        )
                    raise ValueError(msg)
                # end if-else
                
                # Check for nan or return if converged
                if not self.conv[mesh_point_index]:
                    self._check_for_nan(
                        d, mesh_point_index=mesh_point_index, subdomain_index=index
                    )
                else:
                    self.return_element(
                        d, mesh_point_index=mesh_point_index, subdomain_index=index
                    )
            # end index-loop

            # Update for next subdomain
            grid_cell += sd.num_cells
        
        # end mdg-loop

    # ---- From Reaktoro to PorePy

    def set_species_in_PorePy_dict_from_state(self, mdg: pp.MixedDimensionalGrid):
        """Set the species names and values in PorePy parameter dictionary.

        Parameters:
            mdg: A ``PorePy`` mixed-dimensional computational grid

        """
        
        for sd, d in mdg.subdomains(return_data=True):
            for species in self.chem_system.species():
                if species.name not in d[pp.PARAMETERS][self.chemical_name]:
                    species_val = self.state.speciesAmount(species.name()).val()
                    d[pp.PARAMETERS][self.chemical_name].update(
                        {species.name(): species_val * np.ones(sd.num_cells)}
                        ) # Use a default zeros value for initialisation value
        
    
    def set_species_in_PorePy_dict_from_states(self, mdg: pp.MixedDimensionalGrid):
        """ Set values from R to PP
        
        It is assumed that set_species_from_state has been used
                
        """
        # Global indexation
        grid_index = 0
        for sd, d in mdg.subdomains(return_data=True):

            for index in range(sd.num_cells):
                mesh_point_index = index + grid_index
                for species in self.chem_system.species():
            
                    species_val = self.states[mesh_point_index].speciesAmount(
                        species.name()
                        ).val()
                    
                    d[pp.PARAMETERS][self.chemical_name][species.name()][index]=species_val
                    
                # end species-loop
            # end index-loop

            # For next subdomain
            grid_index += sd.num_cells
        # end mdg-loop

    def set_state_values(self, mdg: pp.MixedDimensionalGrid):
        """Set values from ``pp.PARAMETER`` to ``pp.STATE``

        Parameters:
            mdg: A ``PorePy`` mixed-dimensional computational grid

        """
        for _, d in mdg.subdomains(return_data=True):

            for species in self.chem_system.species():
                d[pp.STATE].update(
                    {species.name(): d[pp.PARAMETERS][self.chemical_name][species.name()]}
                )
            # end species-loop
        # end mdg-loop

    def scale_state_values(self, mdg: pp.MixedDimensionalGrid, scale_by_porosity=False):
        """The chemical values from Reaktoro are scaled by porosity.
        This scaling is also included in ``set_state_values``.
        'Remove' it out of ``pp.STATE``, i.e. this function is used after set_state_values

        Parameters:
            mdg: A ``PorePy`` mixed-dimensional computational grid
            scale_by_porosity: if values should be scaled by porosity

        """
        
        global_index = 0
        for sd, d in mdg.subdomains(return_data=True):

            # The porosity
            if scale_by_porosity:
                phi = d[pp.PARAMETERS]["mass"]["porosity"] 
            else:
                phi = 1
            # end if-else
            
            for phase in self.chem_system.phases():
                if phase.name() == "AqueousPhase":
                    for species in self.chem_system.species():
                        d[pp.STATE].update(
                            {"unscaled_" + species.name(): d[pp.STATE][species.name()] / phi}
                        )
                    # end species-loop
                    
                else: # minerals
                    self._scale_mineral_state_values(
                        sd, 
                        d, 
                        global_index)
                # end if
            # end phase-loop
            
            global_index += sd.num_cells
            
        # end mdg-loop
                
    def element_values(
        self,
        mdg: Optional[pp.MixedDimensionalGrid] = None,
        inds: Optional[Union[np.ndarray, list]] = None,
        mesh_point_index: Optional[int] = None
    ):
        """Return 'inds' of the equilibrium calulated 'elements'

        Parameters:
            mdg: A ``PorePy`` mixed-dimensional computational grid

            inds (optional): ``default=None``

                The indices of the part to be extracted.

        """

        if inds is None:
            return 0
        # end if

        # Extract the desired part of the formula matrix
        W_at_inds = self.chem_system.formulaMatrix()[:, inds]

        if mdg is None:
            
            # If inds is given, but mesh_point_index is not, return based on state
            if mesh_point_index is None and inds is not None:
                n = self.state.speciesAmounts().asarray()[inds]
                elements = W_at_inds.dot(n)    
        
            elif mesh_point_index is not None:
                n = self.states[mesh_point_index].speciesAmounts().asarray()[inds]
                elements = W_at_inds.dot(n)
        else:    
            elements = np.zeros(mdg.num_subdomain_cells() * self.num_components)
            elem_inds = np.arange(0, self.num_components)
    
            # Finally loop over the mdg to make a global vector
            for i in range(len(self.states)):
                state = self.states[i]
    
                # Concentrations
                n = state.speciesAmounts().asarray()[inds]
    
                # Elements
                b = W_at_inds.dot(n)
    
                # Fluid elements
                elements[elem_inds + i * self.num_components] = b
            # end i-loop

        return elements
