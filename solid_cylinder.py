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

from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import (
    DOFDesc, DISCR_TAG_BASE, DISCR_TAG_QUAD,
    VolumeDomainTag
)
from mirgecom.profiling import PyOpenCLProfilingArrayContext
from mirgecom.multiphysics.thermally_coupled_fluid_wall import (
    coupled_ns_heat_operator
)
from mirgecom.discretization import create_discretization_collection
from mirgecom.utils import force_evaluation
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
    AdiabaticNoslipWallBoundary,
    PressureOutflowBoundary,
    LinearizedOutflowBoundary
)
from mirgecom.fluid import make_conserved
from mirgecom.transport import SimpleTransport
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

        temperature = self._temperature + zeros
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


class SolidInitializer:

    def __init__(self, temperature):
        self._temperature = temperature

    def __call__(self, x_vec, wall_model):
        x = x_vec[0]
        actx = x.array_context
        zeros = actx.np.zeros_like(x)

        mass = wall_model.density() + zeros
        energy = mass * wall_model.enthalpy(self._temperature)
    
        return WallConservedVars(mass=mass, energy=energy)


@with_container_arithmetic(bcast_obj_array=False,
                           bcast_container_types=(DOFArray, np.ndarray),
                           matmul=True,
                           _cls_has_array_context_attr=True,
                           rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class WallConservedVars:
    mass: DOFArray
    energy: DOFArray

    @property
    def array_context(self):
        """Return an array context for the :class:`ConservedVars` object."""
        return get_container_context_recursively(self.mass)


@dataclass_array_container
@dataclass(frozen=True)
class WallDependentVars:
    thermal_conductivity: DOFArray
    temperature: DOFArray


@dataclass_array_container
@dataclass(frozen=True)
class WallState():
    cv: WallConservedVars
    dv: WallDependentVars


class WallModel:
    """Model for calculating wall quantities."""
    def __init__(self, density_func, enthalpy_func, heat_capacity_func,
                 thermal_conductivity_func):
        self._density_func = density_func
        self._enthalpy_func = enthalpy_func
        self._heat_capacity_func = heat_capacity_func
        self._thermal_conductivity_func = thermal_conductivity_func

    def density(self):
        return self._density_func()

    def heat_capacity(self, temperature=None):
        return self._heat_capacity_func()

    def enthalpy(self, temperature):
        return self._enthalpy_func(temperature)

    def thermal_diffusivity(self, mass, temperature=None,
                            thermal_conductivity=None):
        if thermal_conductivity is None:
            thermal_conductivity = self.thermal_conductivity()
        return thermal_conductivity/(mass * self.heat_capacity())

    def thermal_conductivity(self, temperature=None):
        if temperature is None:
            return self._thermal_conductivity_func()
        # FIXME add Newton iteration here when cp and enthalpy are f(T)
        return None

    def eval_temperature(self, wv, tseed=None):
        return wv.energy/(self.density()*self.heat_capacity())

    def dependent_vars(self, wv, tseed=None):
        temperature = self.eval_temperature(wv, tseed)
        kappa = self.thermal_conductivity() + wv.mass*0.0
        return WallDependentVars(thermal_conductivity=kappa,
                                 temperature=temperature)


def sponge_func(cv, cv_ref, sigma):
    return sigma*(cv_ref - cv)


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

    Reynolds_number = 150.0
    Mach_number = 0.3

     # default i/o frequencies
    nviz = 200
    nrestart = 10000
    nhealth = 1
    nstatus = 100

    ngarbage = 20

    # timestepping control
    integrator = "compiled_lsrk45"
    t_final = 100.0

    local_dt = True
    constant_cfl = True
    current_cfl = 0.1
    current_dt = 0.0 #dummy if constant_cfl = True
    
    # discretization and model control
    order = 3
    use_overintegration = False

    # TODO Use the temperature-dependent table for graphite
    wall_graphite_rho = 1625.0
    wall_graphite_cp = 770.0
    wall_graphite_kappa = 50.0
    wall_temperature = 900.0
    wall_emissivity = 1.0

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

    import os
    if rank == 0:
        os.system("rm -rf cylinder_grid.msh cylinder_grid-v2.msh")
        os.system("gmsh cylinder_grid.geo -2 cylinder_grid.msh")
        os.system("gmsh cylinder_grid.msh -save -format msh2 -o cylinder_grid-v2.msh")
    else:
        os.system("sleep 2s")

    restart_step = None
    if restart_filename:
        from mirgecom.restart import read_restart_data
        restart_data = read_restart_data(actx, restart_filename)
        volume_to_local_mesh_data = restart_data["volume_to_local_mesh_data"]
        global_nelements = restart_data["global_nelements"]
        assert restart_data["num_ranks"] == num_ranks
    else:  # import the grid
        def get_mesh_data():
            from meshmode.mesh.io import read_gmsh
            mesh, tag_to_elements = read_gmsh(
                "cylinder_grid-v2.msh", force_ambient_dim=2,
                return_tag_to_elements_map=True)
            volume_to_tags = {
                "Fluid": ["fluid"],
                "Solid": ["solid"]}
            return mesh, tag_to_elements, volume_to_tags

        def partition_generator_func(mesh, tag_to_elements, num_ranks):
            from meshmode.distributed import get_partition_by_pymetis
            return get_partition_by_pymetis(mesh, num_ranks)

        volume_to_local_mesh_data, global_nelements = distribute_mesh(
            comm, get_mesh_data, partition_generator_func)

    local_fluid_mesh = volume_to_local_mesh_data["Fluid"]
    local_solid_mesh = volume_to_local_mesh_data["Solid"]

    local_nelements = local_fluid_mesh[0].nelements \
                    + local_solid_mesh[0].nelements

    from mirgecom.discretization import create_discretization_collection
    dcoll = create_discretization_collection(
        actx,
        volume_meshes={
            vol: mesh
            for vol, (mesh, _) in volume_to_local_mesh_data.items()},
        order=order)

    dd_vol_fluid = DOFDesc(VolumeDomainTag("Fluid"), DISCR_TAG_BASE)
    dd_vol_solid = DOFDesc(VolumeDomainTag("Solid"), DISCR_TAG_BASE)

    fluid_nodes = actx.thaw(dcoll.nodes(dd_vol_fluid))
    solid_nodes = actx.thaw(dcoll.nodes(dd_vol_solid))

    fluid_zeros = force_evaluation(actx, actx.np.zeros_like(fluid_nodes[0]))
    solid_zeros = force_evaluation(actx, actx.np.zeros_like(solid_nodes[0]))

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
    mechanism_file = "inert"  # FIXME

    from mirgecom.mechanisms import get_mechanism_input
    mech_input = get_mechanism_input(mechanism_file)

    cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
    nspecies = cantera_soln.n_species

    # Initial temperature, pressure, and mixture mole fractions are needed to
    # set up the initial state in Cantera.
    temp_cantera = 300.0

    x_fluid = np.zeros(nspecies)
    x_fluid[cantera_soln.species_index("Ar")] = 1.0  # FIXME

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

    # FIXME use the MixtureTransport
    mu = (340*Mach_number)/Reynolds_number
    kappa = 1000.0*mu/0.71
    transport_model = SimpleTransport(viscosity=mu,
        thermal_conductivity=kappa, species_diffusivity=0.25*np.ones(nspecies,))

    gas_model = GasModel(eos=eos, transport=transport_model)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # {{{ Initialize wall model

    def _get_wall_density():
        return wall_graphite_rho

    def _get_wall_enthalpy(temperature):
        return wall_graphite_cp * temperature

    def _get_wall_heat_capacity():
        return wall_graphite_cp

    def _get_wall_thermal_conductivity():
        return wall_graphite_kappa

    wall_model = WallModel(
        density_func=_get_wall_density,
        enthalpy_func=_get_wall_enthalpy,
        heat_capacity_func=_get_wall_heat_capacity,
        thermal_conductivity_func=_get_wall_thermal_conductivity)

    # }}}

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

    # ~~~~~~~~~~~~~~~~~~~~

    def make_solid_state(cv, temperature_seed):
        # force temp_seed "None" for now
        dv = wall_model.dependent_vars(cv, None)
        return WallState(cv=cv, dv=dv)

    get_solid_state = actx.compile(make_solid_state)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    fluid_init = FluidInitializer(nspecies=nspecies, pressure=100000.0,
        temperature=300.0, mach=Mach_number, species_mass_fraction=y_fluid)

    solid_init = SolidInitializer(wall_temperature)

    if restart_file is None:
        current_step = 0
        current_t = 0.0
        if rank == 0:
            logging.info("Initializing soln.")

        fluid_cv = fluid_init(fluid_nodes, eos)
        fluid_tseed = force_evaluation(actx, temperature_seed + fluid_zeros)

        solid_cv = solid_init(solid_nodes, wall_model)
        solid_tseed = force_evaluation(actx, wall_temperature + solid_zeros)

    else:
        if rank == 0:
            logger.info("Restarting soln.")

        restart_step = restart_data["step"]
        current_step = restart_step
        current_t = restart_data["t"]

        fluid_cv = restart_data["fluid_cv"]
        fluid_tseed = restart_data["fluid_temperature_seed"]
        solid_cv = restart_data["solid_cv"]
        solid_tseed = restart_data["solid_temperature_seed"]

    first_step = force_evaluation(actx, current_step)

    fluid_cv = force_evaluation(actx, fluid_cv)
    fluid_tseed = force_evaluation(actx, fluid_tseed)
    fluid_state = get_fluid_state(fluid_cv, fluid_tseed)

    solid_cv = force_evaluation(actx, solid_cv)
    solid_tseed = force_evaluation(actx, solid_tseed)
    solid_state = get_solid_state(solid_cv, solid_tseed)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    fluid_cv_ref = force_evaluation(actx, fluid_init(x_vec=fluid_nodes, eos=eos))
    ref_state = get_fluid_state(fluid_cv_ref, fluid_tseed)

    # initialize the sponge field
    sponge_x_thickness = 10.0
    sponge_y_thickness = 10.0

    xMaxLoc = +50.0
    xMinLoc = -25.0
    yMaxLoc = +25.0
    yMinLoc = -25.0

    sponge_amp = 100.0 #may need to be modified. Let's see...

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
   
    wall_boundary = AdiabaticNoslipWallBoundary()
    inflow_boundary  = PrescribedFluidBoundary(boundary_state_func=_inflow_boundary_state_func)
    outflow_boundary = PrescribedFluidBoundary(boundary_state_func=_outflow_boundary_state_func) 
    #outflow_boundary = PressureOutflowBoundary(boundary_pressure=1.0)

    fluid_boundaries = {
        dd_vol_fluid.trace("inflow").domain_tag: inflow_boundary,
        dd_vol_fluid.trace("outflow").domain_tag: outflow_boundary}

    solid_boundaries = {}

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
    solid_visualizer = make_visualizer(dcoll, volume_dd=dd_vol_solid)

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
            
    def my_write_viz(step, t, dt, fluid_state, wall_state):

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

        solid_viz_fields = [
            ("CV_rho", wall_state.cv.mass),
            ("CV_rhoE", wall_state.cv.energy),
            ("DV_T", wall_state.dv.temperature),
            ("dt", dt[2] if local_dt else None),
        ]

        write_visfile(dcoll, fluid_viz_fields, fluid_visualizer,
            vizname=vizname+"-fluid", step=step, t=t,
            overwrite=True, comm=comm)
        write_visfile(dcoll, solid_viz_fields, solid_visualizer,
            vizname=vizname+"-wall", step=step, t=t, overwrite=True, comm=comm)                 

    def my_write_restart(step, t, state):
        if rank == 0:
            print("Writing restart file...")

        cv, tseed, wv, wv_tseed = state
        restart_fname = rst_pattern.format(cname=casename, step=step,
                                           rank=rank)
        if restart_fname != restart_filename:
            restart_data = {
                "volume_to_local_mesh_data": volume_to_local_mesh_data,
                "cv": cv,
                "temperature_seed": tseed,
                "nspecies": nspecies,
                "wv": wv,
                "wall_temperature_seed": wv_tseed,
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

    from grudge.dt_utils import characteristic_lengthscales
    wall_lengthscales = characteristic_lengthscales(actx, dcoll, dd=dd_vol_solid)

    def _my_get_timestep_solid(solid_state, t, dt):
        return solid_zeros + 1e-6

#        solid_cfl = 0.1

#        if not constant_cfl:
#            return dt

#        wall_alpha = \
#            solid_state.dv.thermal_conductivity/(solid_state.cv.mass
#                                                 * wall_model.heat_capacity())
#        if local_dt:
#            return wall_lengthscales**2 * solid_cfl/wall_alpha
#        else:
#            return actx.to_numpy(nodal_min(dcoll, dd_vol_wall,
#                wall_lengthscales**2 * solid_cfl/wall_alpha))[()]

    def _my_get_timestep_fluid(fluid_state, t, dt):

        if not constant_cfl:
            return dt

        return get_sim_timestep(dcoll, fluid_state, t, dt,
            current_cfl, gas_model, constant_cfl=constant_cfl,
            local_dt=local_dt, fluid_dd=dd_vol_fluid)

    my_get_timestep_fluid = actx.compile(_my_get_timestep_fluid)
    my_get_timestep_solid = actx.compile(_my_get_timestep_solid)

    def my_pre_step(step, t, dt, state):

        if logmgr:
            logmgr.tick_before()

        fluid_cv, fluid_tseed, solid_cv, solid_tseed = state

        fluid_cv = force_evaluation(actx, fluid_cv)
        fluid_tseed = force_evaluation(actx, fluid_tseed)
        solid_cv = force_evaluation(actx, solid_cv)
        solid_tseed = force_evaluation(actx, solid_tseed)

        # construct species-limited fluid state
        fluid_state = get_fluid_state(fluid_cv, fluid_tseed)
        fluid_cv = fluid_state.cv

        # construct species-limited fluid state
        solid_state = get_solid_state(solid_cv, solid_tseed)
        solid_cv = solid_state.cv

        state = make_obj_array([fluid_state.cv, fluid_state.dv.temperature,
                                solid_state.cv, solid_state.dv.temperature])

        try:

            if local_dt:
                t = force_evaluation(actx, t)
                dt_fluid = force_evaluation(actx, my_get_timestep_fluid(fluid_state, t[0], dt[0]))
                dt_solid = force_evaluation(actx, my_get_timestep_solid(solid_state, t[2], dt[2]))
                dt = make_obj_array([dt_fluid, dt_fluid, dt_solid, dt_solid])
            else:
                if constant_cfl:
                    dt = get_sim_timestep(dcoll, fluid_state, t, dt, maximum_cfl,
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
                my_write_viz(step=step, t=t, dt=dt, fluid_state=fluid_state,
                    wall_state=solid_state)

            if check_step(step=step, interval=nrestart):
                my_write_restart(step=step, t=t, state=state)

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")

            my_write_viz(step=step, t=t, dt=dt, fluid_state=fluid_state,
                         wall_state=solid_state)
            raise

        return state, dt

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def my_rhs(t, state):
        fluid_cv, fluid_tseed, solid_cv, solid_tseed = state

        fluid_zeros = actx.np.zeros_like(fluid_cv.mass)
        solid_zeros = actx.np.zeros_like(solid_cv.mass)

        fluid_state = make_fluid_state(cv=fluid_cv, gas_model=gas_model,
            temperature_seed=fluid_tseed, limiter_func=_limit_fluid_cv,
            limiter_dd=dd_vol_fluid)

        wall_state = make_solid_state(cv=solid_cv, temperature_seed=solid_tseed)

        fluid_rhs, solid_energy_rhs = coupled_ns_heat_operator(
            dcoll, gas_model, dd_vol_fluid, dd_vol_solid,
            fluid_boundaries, solid_boundaries, fluid_state,
            wall_state.dv.thermal_conductivity, wall_state.dv.temperature,
            time=t, interface_noslip=True, interface_radiation=True,
            wall_emissivity=wall_emissivity, sigma=5.67e-8,
            ambient_temperature=300.0,
            limiter_func=_limit_fluid_cv, quadrature_tag=quadrature_tag)

        fluid_source_terms = (
            sponge_func(cv=fluid_state.cv, cv_ref=ref_state.cv, sigma=sponge_sigma)
            # + eos.get_species_source_terms(fluid_state.cv, fluid_state.temperature)
            # add heterogeneous chemistry in here (this should only exist on the surface)
        )

        solid_rhs = WallConservedVars(mass=solid_zeros,
                                      energy=solid_energy_rhs)

        solid_source_terms = WallConservedVars(
            mass=solid_zeros,  # no wall degradation
            energy=10000.0 + solid_zeros
        )

        return make_obj_array([fluid_rhs + fluid_source_terms, fluid_zeros,
                               solid_rhs + solid_source_terms, solid_zeros])

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

    stepper_state = make_obj_array([fluid_state.cv, fluid_state.dv.temperature,
                                    solid_state.cv, solid_state.dv.temperature])

    if local_dt == True:
        dt_fluid = force_evaluation(actx,
            my_get_timestep_fluid(fluid_state,
                                  force_evaluation(actx, current_t + fluid_zeros),
                                  force_evaluation(actx, current_dt + fluid_zeros)))

        dt_solid = force_evaluation(actx,
            my_get_timestep_solid(solid_state,
                                  force_evaluation(actx, current_t + solid_zeros),
                                  force_evaluation(actx, current_dt + solid_zeros)))

        dt = make_obj_array([dt_fluid, dt_fluid*0.0, dt_solid, dt_solid*0.0])

        t_fluid = force_evaluation(actx, current_t + fluid_zeros)
        t_solid = force_evaluation(actx, current_t + solid_zeros)

        t = make_obj_array([t_fluid, t_fluid, t_solid, t_solid])
    else:
        if constant_cfl:
            dt = get_sim_timestep(dcoll, fluid_state, t, maximum_fluid_dt,
                maximum_cfl, t_final, constant_cfl, local_dt, dd_vol_fluid)
        else:
            dt = 1.0*maximum_fluid_dt
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

    final_cv, tseed, final_wv, wv_tseed = stepper_state

    final_fluid_state = get_fluid_state(final_cv, tseed)
    final_solid_state = get_solid_state(final_wv, wv_tseed)

    my_write_restart(step=final_step, t=final_t, state=stepper_state)

    my_write_viz(step=final_step, t=final_t, dt=current_dt,
                 fluid_state=final_fluid_state, wall_state=final_solid_state)

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
    casename = "cylinder"
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
