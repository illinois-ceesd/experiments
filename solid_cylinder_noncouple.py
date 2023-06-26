"""mirgecom driver for the wall-fluid coupling demonstration."""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import yaml
import logging
import sys
import gc
import numpy as np
import pyopencl as cl
from functools import partial

from dataclasses import dataclass
from arraycontext import (
    dataclass_array_container, with_container_arithmetic,
    get_container_context_recursively
)
from meshmode.dof_array import DOFArray
import grudge.op as op
from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import (
    DOFDesc, DISCR_TAG_BASE, DISCR_TAG_QUAD,
    VolumeDomainTag
)
from mirgecom.profiling import PyOpenCLProfilingArrayContext
from mirgecom.discretization import create_discretization_collection
from mirgecom.utils import force_evaluation
from mirgecom.navierstokes import ns_operator
from mirgecom.simutil import (
    check_step,
    get_sim_timestep,
    distribute_mesh,
    write_visfile,
    check_naninf_local,
    check_range_local,
    global_reduce
)
from mirgecom.restart import (
    write_restart_file
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools

from mirgecom.integrators import lsrk54_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PrescribedFluidBoundary,
    AdiabaticSlipBoundary,
    PressureOutflowBoundary,
    IsothermalWallBoundary
)
from mirgecom.fluid import make_conserved
from mirgecom.transport import SimpleTransport, MixtureAveragedTransport
from mirgecom.eos import PyrometheusMixture
from mirgecom.gas_model import GasModel, make_fluid_state

from logpyle import IntervalTimer, set_dt
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info, logmgr_set_time,
    logmgr_add_device_memory_usage,
)

from pytools.obj_array import make_obj_array

class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record):
        if self.reject:
            return (record.levelno != self.passlevel)
        else:
            return (record.levelno == self.passlevel)


#h1 = logging.StreamHandler(sys.stdout)
#f1 = SingleLevelFilter(logging.INFO, False)
#h1.addFilter(f1)
#root_logger = logging.getLogger()
#root_logger.addHandler(h1)
#h2 = logging.StreamHandler(sys.stderr)
#f2 = SingleLevelFilter(logging.INFO, True)
#h2.addFilter(f2)
#root_logger.addHandler(h2)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass


class FluidInitializer:

    def __init__(self, nspecies, pressure, temperature, mach, species_mass_fraction):

        self._nspecies = nspecies
        self._pressure = pressure
        self._temperature = temperature
        self._mach = mach
        self._yf = species_mass_fraction

    def __call__(self, x_vec, eos):

        x = x_vec[0]
        actx = x.array_context
        zeros = actx.np.zeros_like(x)

        u_x = 340.0 * self._mach + zeros
        u_y = x*0.0
        velocity = make_obj_array([u_x, u_y])

        radius = actx.np.sqrt(x_vec[0]**2 + x_vec[1]**2)

        temperature = 0.5*(1.0 - actx.np.tanh(1.0/1e-4*(radius - 2.25e-3)))*1000.0 + 300.0
        pressure = self._pressure + zeros
        y = self._yf + zeros

        mass = eos.get_density(pressure, temperature, species_mass_fractions=y)
        specmass = mass * y
        momentum = velocity*mass
        internal_energy = eos.get_internal_energy(temperature,
                                                  species_mass_fractions=y)
        kinetic_energy = 0.5 * np.dot(velocity, velocity)
        energy = mass*(internal_energy + kinetic_energy)

        return make_conserved(dim=2, mass=mass, energy=energy,
                              momentum=momentum, species_mass=specmass)


def sponge_func(cv, cv_ref, sigma):
    return sigma*(cv_ref - cv)

class GasSurfaceReactions:
    r"""Get the source term from heterogenous reaction
    Start with Park model for O2
    C(b)+O2->CO+O-1.40 eV
    """
    
    def __init__(self):
        pass
        
    def get_hetero_chem_source_terms(self, nodes, cv, dv, cantera_soln)->DOFArray:
        actx = cv.mass.array_context
        
        #dd_face = dd.trace('surface')
        #wall_temp = op.project(dcoll, dd, dd_face, dv.temperature)
        #wall_species = op.project(dcoll, dd, dd_face, cv.species_mass)
        radius = actx.np.sqrt(nodes[0]**2 + nodes[1]**2)
        wall_flag = actx.np.less(radius,0.0015+1e-7)
        nspecies = cantera_soln.n_species
        wall_temp = dv.temperature
        wall_species = cv.species_mass
        sources = wall_species*0
        #spec indexes for source terms
        o2_index = cantera_soln.species_index("O2")
        co_index = cantera_soln.species_index("CO")
        o_index = cantera_soln.species_index("O")
        #constants
        mw_c = 12.011
        mw_o = 15.999
        mw_o2 = mw_o*2
        mw_co2 = 44.010
        mw_co = mw_o+mw_c
        univ_gas_const = 8314.46261815324
        n_avo = 6.0221408e+23
        kb = 1.38064852e-23
        #reaction rate terms
        eps_o = 0.63*actx.np.exp(-1160/wall_temp)
        eps_o2 = (1.43e-3 + 0.01*actx.np.exp(-1450/wall_temp))/(1 + 2e-4* actx.np.exp(13000/wall_temp))
        f_o2 = 0.25*actx.np.sqrt(8*kb*wall_temp/(np.pi * mw_o2/n_avo))
        f_o = 0.25*actx.np.sqrt(8*kb*wall_temp/(np.pi * mw_o/n_avo))
        k_o = f_o*eps_o
        k_o2 = f_o2*eps_o2
        
        #reaction source terms, \dot{W}
        sources[o2_index] = -(wall_species[o2_index]/mw_o2)*k_o2*mw_o2
        sources[co_index] = (wall_species[o2_index]/mw_o2)*k_o2*mw_co + (wall_species[o_index]/mw_o)*k_o*mw_co
        sources[o_index] = (wall_species[o2_index]/mw_o2)*k_o2*mw_o - (wall_species[o_index]/mw_o)*k_o*mw_o
        #Fix energy balance
        
        zeros = cv.mass*0.0
        species_source = make_obj_array([
            actx.np.where(wall_flag,sources[i], zeros)
            for i in range(nspecies)
        ])
        
        #h_f = -32.3e3
        #dt_E = h_f*(cv.species_mass/mw_o2)*k_o2
        #actx.np.where(wall_flag,sources[i],0)
        return make_conserved(dim=2, mass=zeros, momentum=make_obj_array([zeros, zeros]), energy=zeros, species_mass=species_source)


class InitSponge:

    def __init__(self, *, x_min=None, x_max=None, y_min=None, y_max=None,
                 x_thickness=None, y_thickness=None, amplitude):
        """ """
        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max
        self._x_thickness = x_thickness
        self._y_thickness = y_thickness
        self._amplitude = amplitude

    def __call__(self, x_vec):
        """ """
        xpos = x_vec[0]
        ypos = x_vec[1]
        actx = xpos.array_context
        zeros = 0*xpos

        sponge_x = xpos*0.0
        sponge_y = xpos*0.0

        if (self._y_max is not None):
          y0 = (self._y_max - self._y_thickness)
          dy = +((ypos - y0)/self._y_thickness)
          sponge_y = sponge_y + self._amplitude * actx.np.where(
              actx.np.greater(ypos, y0),
                  actx.np.where(actx.np.greater(ypos, self._y_max),
                                1.0, 3.0*dy**2 - 2.0*dy**3),
                  0.0
          )

        if (self._y_min is not None):
          y0 = (self._y_min + self._y_thickness)
          dy = -((ypos - y0)/self._y_thickness)
          sponge_y = sponge_y + self._amplitude * actx.np.where(
              actx.np.less(ypos, y0),
                  actx.np.where(actx.np.less(ypos, self._y_min),
                                1.0, 3.0*dy**2 - 2.0*dy**3),
              0.0
          )

        if (self._x_max is not None):
          x0 = (self._x_max - self._x_thickness)
          dx = +((xpos - x0)/self._x_thickness)
          sponge_x = sponge_x + self._amplitude * actx.np.where(
              actx.np.greater(xpos, x0),
                  actx.np.where(actx.np.greater(xpos, self._x_max),
                                1.0, 3.0*dx**2 - 2.0*dx**3),
                  0.0
          )

        if (self._x_min is not None):
          x0 = (self._x_min + self._x_thickness)
          dx = -((xpos - x0)/self._x_thickness)
          sponge_x = sponge_x + self._amplitude * actx.np.where(
              actx.np.less(xpos, x0),
                  actx.np.where(actx.np.less(xpos, self._x_min),
                                1.0, 3.0*dx**2 - 2.0*dx**3),
              0.0
          )

        return actx.np.maximum(sponge_x,sponge_y)


@mpi_entry_point
def main(actx_class, ctx_factory=cl.create_some_context, use_logmgr=True,
         use_profiling=False, casename=None, lazy=False,
         restart_filename=None):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = 0
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    logmgr = initialize_logmgr(use_logmgr, filename=(f"{casename}.sqlite"),
                               mode="wo", mpi_comm=comm)

    cl_ctx = ctx_factory()

    if use_profiling:
        queue = cl.CommandQueue(
            cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    else:
        queue = cl.CommandQueue(cl_ctx)

    if lazy:
        actx = actx_class(comm, queue, mpi_base_tag=12000,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))
    else:
        actx = actx_class(comm, queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
                force_device_scalars=True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    rst_path = "restart_data/"
    viz_path = "viz_data/"
    vizname = viz_path+casename
    rst_pattern = rst_path+"{cname}-{step:06d}-{rank:04d}.pkl"

#    Reynolds_number = 150.0
    Mach_number = 0.0025

     # default i/o frequencies
    nviz = 250
    nrestart = 10000
    nhealth = 1
    nstatus = 100

    ngarbage = 20

    # timestepping control
    integrator = "compiled_lsrk45"
    t_final = 0.1

    local_dt = False
    constant_cfl = True
    current_cfl = 0.3
    current_dt = 0.0 #dummy if constant_cfl = True
    
    # discretization and model control
    order = 2
    use_overintegration = False

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    dim = 2
    current_t = 0
    current_step = 0

    def _compiled_stepper_wrapper(state, t, dt, rhs):
        return compiled_lsrk45_step(actx, state, t, dt, rhs)

    force_eval = True
    if integrator == "compiled_lsrk45":
        from grudge.shortcuts import compiled_lsrk45_step
        timestepper = _compiled_stepper_wrapper
        force_eval = False
        
    if rank == 0:
        print("\n#### Simulation control data: ####")
        print(f"\tnviz = {nviz}")
        print(f"\tnrestart = {nrestart}")
        print(f"\tnhealth = {nhealth}")
        print(f"\tnstatus = {nstatus}")
        if (constant_cfl == False):
          print(f"\tcurrent_dt = {current_dt}")
          print(f"\tt_final = {t_final}")
        else:
          print(f"\tconstant_cfl = {constant_cfl}")
          print(f"\tcurrent_cfl = {current_cfl}")
        print(f"\torder = {order}")
        print(f"\tTime integration = {integrator}")
        print(f"\tuse_overintegration = {use_overintegration}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mesh_filename="isothermal_cylinder_hole-v2.msh"
    restart_step = None
    if restart_file is None:
        if rank == 0:
            print(f"Reading mesh from {mesh_filename}")

        def get_mesh_data():
            from meshmode.mesh.io import read_gmsh
            mesh, tag_to_elements = read_gmsh(
                mesh_filename, force_ambient_dim=dim,
                return_tag_to_elements_map=True)
            tag_to_elements = None
            volume_to_tags = None
            return mesh, tag_to_elements, volume_to_tags

        volume_to_local_mesh_data, global_nelements = distribute_mesh(
            comm, get_mesh_data)

        local_mesh = volume_to_local_mesh_data
        local_nelements = local_mesh.nelements

    else:  # Restart
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, restart_file)
        restart_step = restart_data["step"]
        local_mesh = restart_data["local_mesh"]
        local_nelements = local_mesh.nelements
        global_nelements = restart_data["global_nelements"]
        restart_order = int(restart_data["order"])

        assert comm.Get_size() == restart_data["num_parts"]



    from grudge.dof_desc import DISCR_TAG_QUAD
    from mirgecom.discretization import create_discretization_collection
    dcoll = create_discretization_collection(actx, local_mesh, order=order)

    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = DISCR_TAG_BASE

    if rank == 0:
        logger.info("Done making discretization")

    from grudge.dof_desc import DD_VOLUME_ALL
    dd_vol_fluid = DD_VOLUME_ALL
 

    fluid_nodes = actx.thaw(dcoll.nodes(dd_vol_fluid))
 

    fluid_zeros = force_evaluation(actx, actx.np.zeros_like(fluid_nodes[0]))
    

    use_overintegration = False
    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = DISCR_TAG_BASE

    if rank == 0:
        logger.info("Done making discretization")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    import cantera

    # {{{  Set up initial state using Cantera

    # Use Cantera for initialization
    import os
    current_path = os.path.abspath(os.getcwd()) + "/"
    mechanism_file = current_path + "uiuc_with_O"

    from mirgecom.mechanisms import get_mechanism_input
    mech_input = get_mechanism_input(mechanism_file)

    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    nspecies = cantera_soln.n_species

    # Initial temperature, pressure, and mixture mole fractions are needed to
    # set up the initial state in Cantera.
    temp_cantera = 300.0

    x_fluid = np.zeros(nspecies)
    x_fluid[cantera_soln.species_index("O2")] = 0.90  # FIXME
    x_fluid[cantera_soln.species_index("O")] = 0.10	
    pres_cantera = cantera.one_atm

    cantera_soln.TPX = temp_cantera, pres_cantera, x_fluid
    y_fluid = cantera_soln.Y

    # }}}

    # {{{ Create Pyrometheus thermochemistry object & EOS

    # Import Pyrometheus EOS
    from mirgecom.thermochemistry import get_pyrometheus_wrapper_class_from_cantera
    pyrometheus_mechanism = get_pyrometheus_wrapper_class_from_cantera(
        cantera_soln, temperature_niter=3)(actx.np)

    temperature_seed = 300.0
    eos = PyrometheusMixture(pyrometheus_mechanism,
                             temperature_guess=temperature_seed)

    species_names = pyrometheus_mechanism.species_names
    print(f"Pyrometheus mechanism species names {species_names}")

    # }}}
    
    transport_model = MixtureAveragedTransport(pyrometheus_mechanism, lewis=np.ones(nspecies,))

    gas_model = GasModel(eos=eos, transport=transport_model)
    hetero_chem = GasSurfaceReactions()
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    from mirgecom.limiter import bound_preserving_limiter

    def _limit_fluid_cv(cv, pressure, temperature, dd):

        # limit species
        spec_lim = make_obj_array([
            bound_preserving_limiter(dcoll, cv.species_mass_fractions[i],
                mmin=0.0, mmax=1.0, modify_average=True, dd=dd)
            for i in range(nspecies)
        ])

        # normalize to ensure sum_Yi = 1.0
        aux = cv.mass*0.0
        for i in range(0, nspecies):
            aux = aux + spec_lim[i]
        spec_lim = spec_lim/aux

        # recompute density
        mass_lim = eos.get_density(pressure=pressure,
            temperature=temperature, species_mass_fractions=spec_lim)

        # recompute energy
        energy_lim = mass_lim*(gas_model.eos.get_internal_energy(
            temperature, species_mass_fractions=spec_lim)
            + 0.5*np.dot(cv.velocity, cv.velocity)
        )

        # make a new CV with the limited variables
        return make_conserved(dim=dim, mass=mass_lim, energy=energy_lim,
            momentum=mass_lim*cv.velocity, species_mass=mass_lim*spec_lim)

    def _get_fluid_state(cv, temp_seed):
        return make_fluid_state(cv=cv, gas_model=gas_model,
            temperature_seed=temp_seed, limiter_func=_limit_fluid_cv,
            limiter_dd=dd_vol_fluid)

    get_fluid_state = actx.compile(_get_fluid_state)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    fluid_init = FluidInitializer(nspecies=nspecies, pressure=653,
        temperature=300.0, mach=Mach_number, species_mass_fraction=y_fluid)

    

    if restart_file is None:
        current_step = 0
        current_t = 0.0
        if rank == 0:
            logging.info("Initializing soln.")

        fluid_cv = fluid_init(fluid_nodes, eos)
        fluid_tseed = force_evaluation(actx, temperature_seed + fluid_zeros)


    else:
        if rank == 0:
            logger.info("Restarting soln.")

        restart_step = restart_data["step"]
        current_step = restart_step
        current_t = restart_data["t"]
        if (np.isscalar(current_t) is False):
            current_t = np.min(actx.to_numpy(current_t[0]))

        fluid_cv = restart_data["fluid_cv"]
        fluid_tseed = restart_data["fluid_temperature_seed"]

#        fluid_cv = restart_data["cv"]
#        fluid_tseed = restart_data["temperature_seed"]
#        solid_cv = restart_data["wv"]
#        solid_tseed = restart_data["wall_temperature_seed"]

    first_step = force_evaluation(actx, current_step)

    fluid_cv = force_evaluation(actx, fluid_cv)
    fluid_tseed = force_evaluation(actx, fluid_tseed)
    fluid_state = get_fluid_state(fluid_cv, fluid_tseed)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    fluid_cv_ref = force_evaluation(actx, fluid_init(x_vec=fluid_nodes, eos=eos))
    ref_state = get_fluid_state(fluid_cv_ref, fluid_tseed)

    # initialize the sponge field
    sponge_x_thickness = 0.045
    sponge_y_thickness = 0.045

    xMaxLoc = +0.150
    xMinLoc = -0.0750
    yMaxLoc = +0.0750
    yMinLoc = -0.0750

    sponge_amp = 400.0 #may need to be modified. Let's see...

    sponge_init = InitSponge(amplitude=sponge_amp,
        x_min=xMinLoc, x_max=xMaxLoc,
        y_min=yMinLoc, y_max=yMaxLoc,
        x_thickness=sponge_x_thickness,
        y_thickness=sponge_y_thickness
    )

    sponge_sigma = force_evaluation(actx, sponge_init(x_vec=fluid_nodes))
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # FIXME this boundary conditions are temporary. It will be necessary to 
    # change them in order to match Nic's chamber.

    inflow_nodes = force_evaluation(actx, dcoll.nodes(dd_vol_fluid.trace('inflow')))
    inflow_state = make_fluid_state(cv=fluid_init(x_vec=inflow_nodes, eos=eos),
            gas_model=gas_model, temperature_seed=inflow_nodes[0]*0.0 + 300.0)
    inflow_state = force_evaluation(actx, inflow_state)

    def _inflow_boundary_state_func(**kwargs):
        return inflow_state

    outflow_nodes = force_evaluation(actx, dcoll.nodes(dd_vol_fluid.trace('outflow')))
    outflow_state = make_fluid_state(cv=fluid_init(x_vec=outflow_nodes, eos=eos),
            gas_model=gas_model, temperature_seed=outflow_nodes[0]*0.0 + 300.0)
    outflow_state = force_evaluation(actx, outflow_state)

    def _outflow_boundary_state_func(**kwargs):
        return outflow_state
   
    wall_boundary = AdiabaticSlipBoundary()
    inflow_boundary  = PrescribedFluidBoundary(boundary_state_func=_inflow_boundary_state_func)
    #outflow_boundary = PrescribedFluidBoundary(boundary_state_func=_outflow_boundary_state_func)
    surface_boundary = IsothermalWallBoundary(wall_temperature=1300)
    outflow_boundary = PressureOutflowBoundary(boundary_pressure=653)

    boundaries = {
        dd_vol_fluid.trace("inflow").domain_tag: inflow_boundary,
        dd_vol_fluid.trace("surface").domain_tag: surface_boundary,
        dd_vol_fluid.trace("outflow").domain_tag: outflow_boundary,
        dd_vol_fluid.trace("sidewall").domain_tag: wall_boundary}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    original_casename = casename
    casename = f"{casename}-d{dim}p{order}e{global_nelements}n{num_ranks}"
    logmgr = initialize_logmgr(use_logmgr, filename=(f"{casename}.sqlite"),
                               mode="wo", mpi_comm=comm)

    vis_timer = None
    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_add_device_memory_usage(logmgr, queue)
        logmgr_set_time(logmgr, current_step, current_t)

        logmgr.add_watches([
            ("step.max", "step = {value}, "),
            ("dt.max", "dt: {value:1.5e} s, "),
            ("t_sim.max", "sim time: {value:1.5e} s, "),
            ("t_step.max", "--- step walltime: {value:5g} s\n")
            ])

        try:
            logmgr.add_watches(["memory_usage_python.max",
                                "memory_usage_gpu.max"])
        except KeyError:
            pass

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

        gc_timer = IntervalTimer("t_gc", "Time spent garbage collecting")
        logmgr.add_quantity(gc_timer)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    fluid_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_fluid)

    initname = "cylinder"
    eosname = eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final,
                                     nstatus=nstatus, nviz=nviz,
                                     cfl=current_cfl,
                                     constant_cfl=constant_cfl,
                                     initname=initname,
                                     eosname=eosname, casename=casename)
    if rank == 0:
        logger.info(init_message)
        
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
    def my_write_viz(step, t, dt, fluid_state):

        fluid_viz_fields = [
            ("CV_rho", fluid_state.cv.mass),
            ("CV_rhoU", fluid_state.cv.momentum),
            ("CV_rhoE", fluid_state.cv.energy),
            ("DV_P", fluid_state.pressure),
            ("DV_T", fluid_state.temperature),
            ("dt", dt[0] if local_dt else None),
        ]

        # species mass fractions
        fluid_viz_fields.extend(
            ("Y_"+species_names[i], fluid_state.cv.species_mass_fractions[i])
            for i in range(nspecies))

        
        write_visfile(dcoll, fluid_viz_fields, fluid_visualizer,
            vizname=vizname+"-fluid", step=step, t=t,
            overwrite=True, comm=comm)

    def my_write_restart(step, t, state):
        if rank == 0:
            print("Writing restart file...")

        fluid_cv, fluid_tseed = state
        restart_fname = rst_pattern.format(cname=casename, step=step,
                                           rank=rank)
        if restart_fname != restart_filename:
            restart_data = {
                "volume_to_local_mesh_data": volume_to_local_mesh_data,
                "fluid_cv": fluid_cv,
                "fluid_temperature_seed": fluid_tseed,
                "nspecies": nspecies,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_ranks": num_ranks
            }

            write_restart_file(actx, restart_data, restart_fname, comm)

    def my_health_check(cv, dv):
        health_error = False
        pressure = actx.thaw(actx.freeze(dv.pressure))
        temperature = actx.thaw(actx.freeze(dv.temperature))
        
        if global_reduce(check_naninf_local(dcoll, "vol", pressure), op="lor"):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in pressure data.")

        if global_reduce(check_naninf_local(dcoll, "vol", temperature), op="lor"):
            health_error = True
            logger.info(f"{rank=}: NANs/Infs in temperature data.")
            
        return health_error 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    def _my_get_timestep_fluid(fluid_state, t, dt):

        if not constant_cfl:
            return dt

        return get_sim_timestep(dcoll, fluid_state, t, dt,
            current_cfl, gas_model, constant_cfl=constant_cfl,
            local_dt=local_dt, fluid_dd=dd_vol_fluid)

    my_get_timestep_fluid = actx.compile(_my_get_timestep_fluid)
    
    def my_pre_step(step, t, dt, state):

        if logmgr:
            logmgr.tick_before()

        fluid_cv, fluid_tseed = state

        fluid_cv = force_evaluation(actx, fluid_cv)
        fluid_tseed = force_evaluation(actx, fluid_tseed)
        

        # construct species-limited fluid state
        fluid_state = get_fluid_state(fluid_cv, fluid_tseed)
        fluid_cv = fluid_state.cv

        state = make_obj_array([fluid_state.cv, fluid_state.dv.temperature])

        try:

            if local_dt:
                t = force_evaluation(actx, t)
                dt_fluid = force_evaluation(actx, my_get_timestep_fluid(fluid_state, t[0], dt[0]))
                
                dt = make_obj_array([dt_fluid, dt_fluid])
            else:
                if constant_cfl:
                    dt = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl,
                                          t_final, constant_cfl, local_dt, dd_vol_fluid)

            if check_step(step=step, interval=ngarbage):
                with gc_timer.start_sub_timer():
                    from warnings import warn
                    warn("Running gc.collect() to work around memory growth issue ")
                    import gc
                    gc.collect()

            if check_step(step=step, interval=nhealth):
                health_errors = global_reduce(
                    my_health_check(fluid_state.cv, fluid_state.dv), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if check_step(step=step, interval=nviz):
                my_write_viz(step=step, t=t, dt=dt, fluid_state=fluid_state)

            if check_step(step=step, interval=nrestart):
                my_write_restart(step=step, t=t, state=state)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")

            my_write_viz(step=step, t=t, dt=dt, fluid_state=fluid_state)
            raise

        return state, dt

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def my_rhs(t, state):
        fluid_cv, fluid_tseed = state

        fluid_zeros = actx.np.zeros_like(fluid_cv.mass)

        fluid_state = make_fluid_state(cv=fluid_cv, gas_model=gas_model,
            temperature_seed=fluid_tseed, limiter_func=_limit_fluid_cv,
            limiter_dd=dd_vol_fluid)

        fluid_rhs = ns_operator(dcoll, state=fluid_state, time=t,boundaries=boundaries, gas_model=gas_model,return_gradients=False, quadrature_tag=quadrature_tag)
       

        fluid_source_terms = (
            sponge_func(cv=fluid_state.cv, cv_ref=ref_state.cv, sigma=sponge_sigma)
             + hetero_chem.get_hetero_chem_source_terms(fluid_nodes, fluid_state.cv, fluid_state.dv, cantera_soln)
            # + eos.get_species_source_terms(fluid_state.cv, fluid_state.temperature)
            # add heterogeneous chemistry in here (this should only exist on the surface)
        )
        #print(hetero_chem.get_hetero_chem_source_terms(fluid_nodes, fluid_state.cv, fluid_state.dv, cantera_soln))
     
        return make_obj_array([fluid_rhs + fluid_source_terms, fluid_zeros])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def my_post_step(step, t, dt, state):

        if step == first_step + 1:
            with gc_timer.start_sub_timer():
                gc.collect()
                # Freeze the objects that are still alive so they will not
                # be considered in future gc collections.
                logger.info("Freezing GC objects to reduce overhead of "
                            "future GC collections")
                gc.freeze()

        if logmgr:
            min_dt = np.min(actx.to_numpy(dt[0])) if local_dt else dt
            set_dt(logmgr, min_dt)
            logmgr.tick_after()

        return state, dt
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    stepper_state = make_obj_array([fluid_state.cv, fluid_state.dv.temperature])

    if local_dt == True:
        dt_fluid = force_evaluation(actx,
            my_get_timestep_fluid(fluid_state,
                                  force_evaluation(actx, current_t + fluid_zeros),
                                  force_evaluation(actx, current_dt + fluid_zeros)))


        dt = make_obj_array([dt_fluid, dt_fluid*0.0])

        t_fluid = force_evaluation(actx, current_t + fluid_zeros)

        t = make_obj_array([t_fluid, t_fluid])
    else:
        if constant_cfl:
            t = 1.0*current_t
            dt = get_sim_timestep(dcoll, fluid_state, t, current_dt,
                current_cfl, t_final, constant_cfl, local_dt, dd_vol_fluid)
        else:
            dt = 1.0*current_dt
            t = 1.0*current_t

    if rank == 0:
        logging.info("Stepping.")

    final_step, final_t, stepper_state = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      state=stepper_state,
                      dt=dt, t_final=t_final, t=t,
                      local_dt=local_dt, max_steps=1000000,
                      istep=current_step,
                      force_eval=force_eval)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

    final_cv, tseed = stepper_state

    final_fluid_state = get_fluid_state(final_cv, tseed)
    

    my_write_restart(step=final_step, t=final_t, state=stepper_state)

    my_write_viz(step=final_step, t=final_t, dt=current_dt,
                 fluid_state=final_fluid_state)

    if logmgr:
        logmgr.close()
    elif use_profiling:
        print(actx.tabulate_profiling_data())

    exit()


if __name__ == "__main__":
    import sys
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(description="MIRGE-Com 1D Flame Driver")
    parser.add_argument("-r", "--restart_file",  type=ascii,
                        dest="restart_file", nargs="?", action="store",
                        help="simulation restart file")
    parser.add_argument("-i", "--input_file",  type=ascii,
                        dest="input_file", nargs="?", action="store",
                        help="simulation config file")
    parser.add_argument("-c", "--casename",  type=ascii,
                        dest="casename", nargs="?", action="store",
                        help="simulation case name")
    parser.add_argument("--profile", action="store_true", default=False,
        help="enable kernel profiling [OFF]")
    parser.add_argument("--log", action="store_true", default=True,
        help="enable logging profiling [ON]")
    parser.add_argument("--lazy", action="store_true", default=False,
        help="enable lazy evaluation [OFF]")

    args = parser.parse_args()

    # for writing output
    casename = "cylinder_2rxns_5O"
    if(args.casename):
        print(f"Custom casename {args.casename}")
        casename = (args.casename).replace("'", "")
    else:
        print(f"Default casename {casename}")

    restart_file = None
    if args.restart_file:
        restart_file = (args.restart_file).replace("'", "")
        print(f"Restarting from file: {restart_file}")

    input_file = None
    if(args.input_file):
        input_file = (args.input_file).replace("'", "")
        print(f"Reading user input from {args.input_file}")
    else:
        print("No user input file, using default values")

    print(f"Running {sys.argv[0]}\n")
    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=args.lazy, distributed=True)

    main(actx_class, use_logmgr=args.log, 
         use_profiling=args.profile,
         lazy=args.lazy, casename=casename, restart_filename=restart_file)
