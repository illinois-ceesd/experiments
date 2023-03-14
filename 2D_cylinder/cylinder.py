"""mirgecom driver for the Y0 demonstration.

Note: this example requires a *scaled* version of the Y0
grid. A working grid example is located here:
github.com:/illinois-ceesd/data@y0scaled
"""

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
import numpy as np
import pyopencl as cl
from functools import partial

from arraycontext import thaw, freeze

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from grudge.eager import EagerDGDiscretization
from grudge.shortcuts import make_visualizer

from mirgecom.profiling import PyOpenCLProfilingArrayContext
from mirgecom.navierstokes import ns_operator
from mirgecom.simutil import (
    check_step,
    get_sim_timestep,
    generate_and_distribute_mesh,
    write_visfile,
    check_naninf_local,
    check_range_local,
    global_reduce,
    force_evaluation
)
from mirgecom.restart import (
    write_restart_file
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
import pyopencl.tools as cl_tools

from mirgecom.integrators import (
    rk4_step,
    lsrk54_step,
    euler_step
)
from grudge.shortcuts import compiled_lsrk45_step
from mirgecom.steppers import advance_state
from mirgecom.boundary import (
    PrescribedFluidBoundary,
    AdiabaticNoslipWallBoundary,
    PressureOutflowBoundary,
    LinearizedOutflowBoundary
)
from mirgecom.fluid import make_conserved
from mirgecom.transport import SimpleTransport
from mirgecom.viscous import (
    get_viscous_timestep,
    get_viscous_cfl
)
from mirgecom.eos import IdealSingleGas
from mirgecom.gas_model import GasModel, make_fluid_state

from logpyle import IntervalTimer, set_dt
from mirgecom.logging_quantities import (
    initialize_logmgr,
    logmgr_add_cl_device_info, logmgr_set_time, LogUserQuantity,
    logmgr_add_many_discretization_quantities, logmgr_add_device_memory_usage,
    set_sim_state
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


h1 = logging.StreamHandler(sys.stdout)
f1 = SingleLevelFilter(logging.INFO, False)
h1.addFilter(f1)
root_logger = logging.getLogger()
root_logger.addHandler(h1)
h2 = logging.StreamHandler(sys.stderr)
f2 = SingleLevelFilter(logging.INFO, True)
h2.addFilter(f2)
root_logger.addHandler(h2)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MyRuntimeError(RuntimeError):
    """Simple exception to kill the simulation."""

    pass

def get_mesh(dim, read_mesh=True):
    """Get the mesh."""
    from meshmode.mesh.io import read_gmsh
    mesh_filename = "grid-v2.msh"
    mesh = partial(read_gmsh, filename=mesh_filename, force_ambient_dim=dim)

    return mesh


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
         use_leap=False, use_profiling=False, casename=None, lazy=False,
         rst_filename=None):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = 0
    rank = comm.Get_rank()
    nparts = comm.Get_size()

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

    # ~~~~~~~~~~~~~~~~~~

    restart_path = "restart_data/"
    viz_path = "viz_data/"
    vizname = viz_path+casename
    snapshot_pattern = restart_path+"{cname}-{step:06d}-{rank:04d}.pkl"

    Reynolds_number = 150.0
    Mach_number = 0.3

     # default i/o frequencies
    nviz = 1000
    nrestart = 10000
    nhealth = 1
    nstatus = 100

    # default timestepping control
    integrator = "compiled_lsrk45"
    t_final = 100.0

    constant_cfl = True
    current_cfl = 0.40
    current_dt = 0.0 #dummy if constant_cfl = True
    
    # discretization and model control
    order = 3
    use_overintegration = False

######################################################

    dim = 2
    current_t = 0
    current_step = 0

    # param sanity check
    allowed_integrators = ["euler", "rk2", "rk4", \
                         "ssprk32", "ssprk43", "lsrk54", "compiled_lsrk45"]
    if(integrator not in allowed_integrators):
      error_message = "Invalid time integrator: {}".format(integrator)
      raise RuntimeError(error_message)

    def _compiled_stepper_wrapper(state, t, dt, rhs):
        return compiled_lsrk45_step(actx, state, t, dt, rhs)

    force_eval = True
    if integrator == "euler":
        timestepper = euler_step
    if integrator == "rk2":
        timestepper = rk2_step
    if integrator == "rk4":
        timestepper = rk4_step
    if integrator == "lsrk54":
        timestepper = lsrk54_step
    if integrator == "compiled_lsrk45":
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

    eos = IdealSingleGas()

    mu = (340*Mach_number)/Reynolds_number
    kappa = 1000.0*mu/0.71
    transport_model = SimpleTransport(viscosity=mu, thermal_conductivity=kappa)
    
    gas_model = GasModel(eos=eos, transport=transport_model)

    def get_fluid_state(cv):
        return make_fluid_state(cv=cv, gas_model=gas_model)

    construct_fluid_state = actx.compile(get_fluid_state)

#####################################################################

    def uniform_flow(x_vec, eos, cv=None, **kwargs):
        gamma = eos.gamma()
        R = eos.gas_const()
        
        x = x_vec[0]
        
        u_x = 340*0.3 + 0.0*x
        u_y = 0.0 + 0.0*x

        pressure = 100000.0 + 0.0*x
        mass = 1.0 + 0.0*x

        velocity = make_obj_array([u_x,u_y])   
        ke = .5*np.dot(velocity, velocity)*mass

        rho_e = pressure/(eos.gamma()-1) + ke
        return make_conserved(dim, mass=mass, energy=rho_e,
                              momentum=mass*velocity)

    flow_init = uniform_flow

    ##################################################

    restart_step = None
    if restart_file is None:
        local_mesh, global_nelements = generate_and_distribute_mesh(
            comm, get_mesh(dim=dim))
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

############################################################

    if rank == 0:
        print("Making discretization")
        logging.info("Making discretization")

    restart_step = None
    if restart_file is None:        
        local_mesh, global_nelements = generate_and_distribute_mesh(
            comm, get_mesh(dim=dim))
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
    dcoll = create_discretization_collection(actx, local_mesh, order)
    nodes = thaw(dcoll.nodes(), actx)

    if use_overintegration:
        quadrature_tag = DISCR_TAG_QUAD
    else:
        quadrature_tag = None

    from grudge.dof_desc import DD_VOLUME_ALL
    dd = DD_VOLUME_ALL

    ##################################################  

    if restart_file is None:
        if rank == 0:
            logging.info("Initializing soln.")
        current_cv = flow_init(x_vec=nodes, eos=eos, time=0.)
    else:
        current_t = restart_data["t"]
        current_step = restart_step

        if restart_order != order:
            restart_discr = EagerDGDiscretization(
                actx,
                local_mesh,
                order=restart_order,
                mpi_communicator=comm)
            from meshmode.discretization.connection import make_same_mesh_connection
            connection = make_same_mesh_connection(
                actx,
                dcoll.discr_from_dd("vol"),
                restart_discr.discr_from_dd("vol"))

            current_cv = connection(restart_data["state"])
        else:
            current_cv = restart_data["state"]

        if logmgr:
            logmgr_set_time(logmgr, current_step, current_t)

    current_state = construct_fluid_state(cv=current_cv)
    current_state = force_evaluation(actx, current_state)

    ##################################################

    ref_state = construct_fluid_state(cv=flow_init(x_vec=nodes, eos=eos))

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

    sponge_sigma = force_evaluation(actx, sponge_init(x_vec=nodes))
    
    ############################################################################

    inflow_nodes = force_evaluation(actx, dcoll.nodes(dd.trace('inflow')))
    inflow_state = construct_fluid_state(cv=flow_init(x_vec=inflow_nodes, eos=eos))
    inflow_state = force_evaluation(actx, inflow_state)

    def _inflow_boundary_state_func(**kwargs):
        return inflow_state

    outflow_nodes = force_evaluation(actx, dcoll.nodes(dd.trace('outflow')))
    outflow_state = construct_fluid_state(cv=flow_init(x_vec=outflow_nodes, eos=eos))
    outflow_state = force_evaluation(actx, outflow_state)

    def _outflow_boundary_state_func(**kwargs):
        return outflow_state
   
    wall_boundary = AdiabaticNoslipWallBoundary()
    inflow_boundary  = PrescribedFluidBoundary(boundary_state_func=_inflow_boundary_state_func)
    outflow_boundary = PrescribedFluidBoundary(boundary_state_func=_outflow_boundary_state_func) 
    #outflow_boundary = PressureOutflowBoundary(boundary_pressure=1.0)

    boundaries = {dd.trace("inflow").domain_tag: inflow_boundary,
                  dd.trace("outflow").domain_tag: outflow_boundary,
                  dd.trace("wall").domain_tag: wall_boundary}

    ##################################################

    vis_timer = None
    log_cfl = LogUserQuantity(name="cfl", value=current_cfl)

    logmgr_add_device_memory_usage(logmgr, queue)
    try:
        logmgr.add_watches(["memory_usage_python.max"])
    except KeyError:
        pass

    if logmgr:
        logmgr_add_cl_device_info(logmgr, queue)
        logmgr_set_time(logmgr, current_step, current_t)

        logmgr.add_watches([
            ("step.max", "step: {value}, "),
            ("dt.max", "dt: {value:1.6e} s, "),
            ("t_sim.max", "sim time: {value:1.6e} s, "),
            ("t_step.max", "------- step walltime: {value:6g} s\n")])

        if use_profiling:
            logmgr.add_watches(["pyopencl_array_time.max"])

        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

        gc_timer = IntervalTimer("t_gc", "Time spent garbage collecting")
        logmgr.add_quantity(gc_timer)

    visualizer = make_visualizer(dcoll)

    initname = "cylinder"
    eosname = gas_model.eos.__class__.__name__
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
        
#########################################################################

    def vol_min_loc(x):
        from grudge.op import nodal_min_loc
        return actx.to_numpy(nodal_min_loc(dcoll, "vol", x))[()]

    def vol_max_loc(x):
        from grudge.op import nodal_max_loc
        return actx.to_numpy(nodal_max_loc(dcoll, "vol", x))[()]

    def vol_min(x):
        from grudge.op import nodal_min
        return actx.to_numpy(nodal_min(dcoll, "vol", x))[()]

    def vol_max(x):
        from grudge.op import nodal_max
        return actx.to_numpy(nodal_max(dcoll, "vol", x))[()]
        
#########################################################################

    def my_write_status(step, t, dt, dv, state):
        if constant_cfl:
            cfl = current_cfl
        else:
            from mirgecom.viscous import get_viscous_cfl
            cfl_field = get_viscous_cfl(dcoll, dt, state=state)
            from grudge.op import nodal_max
            cfl = actx.to_numpy(nodal_max(dcoll, "vol", cfl_field))
        status_msg = f"Step: {step}, T: {t}, DT: {dt}, CFL: {cfl}"

        if rank == 0:
            logger.info(status_msg)
            
        if constant_cfl:
            return cfl
        else:
            return cfl_field
            
    from mirgecom.fluid import velocity_gradient
    def my_write_viz(step, t, state, ts_field=None,
                     grad_cv=None, ref_cv=None, sponge_sigma=None):
        
        zVort = None

        viz_fields = [("CV", state.cv),
                      ("DV_U", state.cv.velocity[0]),
                      ("DV_V", state.cv.velocity[1]),
                      ("DV_P", state.dv.pressure),
                      ("DV_T", state.dv.temperature),
                      ("dt" if constant_cfl else "cfl", ts_field)
                      ]

        if (grad_cv is not None):
            grad_v = velocity_gradient(state.cv,grad_cv)
            dudx = grad_v[0][0]
            dudy = grad_v[0][1]
            dvdx = grad_v[1][0]
            dvdy = grad_v[1][1]
            
            zVort = dvdx - dudy

            viz_fields.extend((
                ("Z_vort", zVort),
                ("ref_cv", ref_cv),
                ("sponge_sigma", sponge_sigma),
            ))
                              
        from mirgecom.simutil import write_visfile
        write_visfile(dcoll, viz_fields, visualizer, vizname=vizname,
                      step=step, t=t, overwrite=True)                      

    def my_write_restart(step, t, cv):
        rst_fname = snapshot_pattern.format(cname=casename, step=step, rank=rank)
        if rst_fname != restart_file:
            rst_data = {
                "local_mesh": local_mesh,
                "state": cv,
                "t": t,
                "step": step,
                "order": order,
                "global_nelements": global_nelements,
                "num_parts": nparts
            }
            write_restart_file(actx, rst_data, rst_fname, comm)

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

#########################################################################

    def _my_get_timestep_fluid(fluid_state, t, dt):

        if not constant_cfl:
            return dt

        return get_sim_timestep(dcoll, fluid_state, t, dt,
            current_cfl, t_final=t_final, constant_cfl=constant_cfl,
            local_dt=False, fluid_dd=dd)

    my_get_timestep_fluid = actx.compile(_my_get_timestep_fluid)

#########################################################################

    def my_pre_step(step, t, dt, state):

        if logmgr:
            logmgr.tick_before()

        try:

            fluid_state = construct_fluid_state(state)
            fluid_state = force_evaluation(actx, fluid_state)
         
            dt = get_sim_timestep(dcoll, fluid_state, t, dt, current_cfl,
                                  t_final, constant_cfl)
#            dt = my_get_timestep_fluid(fluid_state, t, dt)   

            do_viz = check_step(step=step, interval=nviz)
            do_restart = check_step(step=step, interval=nrestart)
            do_health = check_step(step=step, interval=nhealth)

            ngarbage = 200
            collect_garbage = check_step(step=step, interval=ngarbage)

            if collect_garbage:
                with gc_timer.start_sub_timer():
                    from warnings import warn
                    warn("Running gc.collect() to work around memory growth issue ")
                    import gc
                    gc.collect()

            if do_health:
                dv = fluid_state.dv
                cv = fluid_state.cv
                health_errors = global_reduce(my_health_check(cv, dv), op="lor")
                if health_errors:
                    if rank == 0:
                        logger.info("Fluid solution failed health check.")
                    raise MyRuntimeError("Failed simulation health check.")

            if do_restart:
                my_write_restart(step=step, t=t, cv=cv)

            if do_viz:
                cv = fluid_state.cv
                dv = fluid_state.dv

                ns_rhs, grad_cv, grad_t = \
                    ns_operator(dcoll, state=fluid_state, time=t,
                                boundaries=boundaries, gas_model=gas_model,
                                return_gradients=True,
                                quadrature_tag=quadrature_tag)
                
                my_write_viz(step=step, t=t, state=fluid_state, ts_field=None,
                             #ref_cv=ref_state.cv, sponge_sigma=sponge_sigma,
                             grad_cv=grad_cv
                )

        except MyRuntimeError:
            if rank == 0:
                logger.info("Errors detected; attempting graceful exit.")

            fluid_state = construct_fluid_state(state)
            fluid_state = force_evaluation(actx, fluid_state)
            my_write_viz(step=step, t=t, state=fluid_state)
            raise

        return state, dt


    def my_rhs(t, state):
        fluid_state = make_fluid_state(cv=state, gas_model=gas_model)
        
        cv_rhs = ns_operator(dcoll, state=fluid_state, time=t,
                        boundaries=boundaries, gas_model=gas_model,
                        return_gradients=False, quadrature_tag=quadrature_tag)

        sponge = sponge_func(cv=fluid_state.cv, cv_ref=ref_state.cv,
                             sigma=sponge_sigma)

        return cv_rhs + sponge


    def my_post_step(step, t, dt, state):
        if logmgr:        
            set_dt(logmgr, dt)
            logmgr.tick_after()
        return state, dt
    
    ##########################################################################

    current_dt = get_sim_timestep(dcoll, current_state, current_t, current_dt,
                                  current_cfl, t_final, constant_cfl)

    if rank == 0:
        logging.info("Stepping.")

    (current_step, current_t, current_cv) = \
        advance_state(rhs=my_rhs, timestepper=timestepper,
                      pre_step_callback=my_pre_step,
                      post_step_callback=my_post_step,
                      state=current_state.cv,
                      dt=current_dt, t_final=t_final, t=current_t,
                      istep=current_step,
                      force_eval=force_eval)
    current_state = make_fluid_state(cv=current_cv, gas_model=gas_model)

    # Dump the final data
    if rank == 0:
        logger.info("Checkpointing final state ...")

#    ts_field, cfl, dt = my_get_timestep(current_t, current_dt, current_state)
#    my_write_viz(step=current_step, t=current_t, cv=current_state.cv,
#                   dv=current_state.dv, ref_cv=ref_state.cv)
#    my_write_restart(step=current_step, t=current_t, cv=current_state.cv)

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
         lazy=args.lazy, casename=casename, rst_filename=restart_file)
