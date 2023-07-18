from functools import partial

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

import components as c

# Algorithm to solve the ODE system using diffrax and the Equinox Class
@partial(jax.jit, static_argnames=['cpModel','runTime'])
def solveModel(
    cpModel,
    runTime: float,
    states: dict[str,float],
):
    
    # Set the parameters for the ODE solver
    #runTime = 1.0
    rtol=1e-3
    atol=1e-6
    dtmax=1e-2
    max_steps=16**4
    dt_fixed=1e-3
    dt_dense=1e-2
    t_stabilise=0.0
    fixed_step=False
    
    # Create the ODE solver
    ode_solver=diffrax.Tsit5()
    #ode_solver=diffrax.RK4()
    #ode_solver=diffrax.Euler(scan_stages=True)
    #ode_solver=diffrax.ImplicitEuler(scan_stages=True)
    #ode_solver=diffrax.ImplicitMidpoint(scan_stages=True)
    #ode_solver=diffrax.ImplicitRK4(scan_stages=True)
    #ode_solver=diffrax.ImplicitTrapezoid(scan_stages=True)
    #ode_solver=diffrax.Midpoint(scan_stages=True)

    # Create the ODE term
    term = diffrax.ODETerm(cpModel)

    # Create the nonlinear solver
    nl_solver = diffrax.NewtonNonlinearSolver(
        rtol=rtol,
        atol=atol,
    )

    # Create the stepsize controller
    if fixed_step:
        stepsize_controller = diffrax.ConstantStepSize()
        dt0 = dt_fixed
        max_steps = int(runTime / dt0)
    else:
        stepsize_controller = diffrax.PIDController(
            rtol=rtol,
            atol=atol,
            dtmax=dtmax,
            pcoeff=0.4,
            icoeff=0.3,
            dcoeff=0.0,
        )
        dt0 = 0.01

    # Solve the ODE system
    res = diffrax.diffeqsolve(
        term,
        ode_solver,
        0.0,
        runTime,
        dt0,
        states,
        args=(nl_solver,),
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
        saveat=diffrax.SaveAt(steps=True, dense=True),
        #adjoint=diffrax.NoAdjoint(),
    )

    # create the time vector and the dense solution
    t_dense = jnp.linspace(t_stabilise, runTime, int((runTime - t_stabilise) / dt_dense) + 1)
    y_dense = jax.vmap(res.evaluate)(t_dense)

    # Calculate the derivatives and the outputs
    deriv, out = cpModel(t_dense, y_dense, (nl_solver,), return_outputs=True)

    return (res, t_dense, y_dense, deriv, out)



# Equinox class holding the system of equations for the CardioPulmonary Model to be used by JAX
class CardioPulmonaryModel(eqx.Module):
    capacitorsObjects: dict[int, c.Capacitor]
    resistorsObjects: dict[int, c.Resistor]
    connectionsObjects: dict[int,c.Connections]
    states: dict[int,float]

    def __call__(
        self,
        t: jnp.ndarray,
        y: dict,
        args: (
            tuple[diffrax.AbstractNonlinearSolver]
            | tuple[diffrax.AbstractNonlinearSolver, jnp.ndarray]
        ),
        return_outputs: bool = False,
    ) -> dict:
        
        # Calculates all the pressures and adds them to a dictionary
        pressures = {key: capacitor.pressure(t,y) for key, capacitor in self.capacitorsObjects.items()}
        #print('pressures--------------')
        #print(pressures.keys())

        # Calculates the derivatives of the capacitors
        dC = {capacitor.capIdx: 0.0 for capacitor in self.capacitorsObjects.values() if capacitor.capIdx != 'u'}
        #print('dC--------------')
        #print(dC.keys())

        
        # Calculates the flows and adds them to a dictionary
        flow = {value.flowIdx: value.flow_rate(t,y,pressures) for value in self.resistorsObjects.values()}
        #print('flow--------------')
        #print(flow.keys())

        # Calculates the derivatives of the flows 
        dI = {value.flowIdx: value.flow_rate_deriv(t,y,pressures) for value in self.resistorsObjects.values()}
        #print('dI--------------')
        #print(dI.keys())


        # Calculates the derivatives of the resistors
        dR = {value.rIdx: 0.0 for value in self.resistorsObjects.values()}
        #print('dR--------------')
        #print(dR.keys())

        # Calculates the derivatives of the volumes
        dV = {key: value.dV(flow) for key, value in self.connectionsObjects.items()}  
        #print('dV--------------')
        #print(dV.keys())

        dBiasP = {'Pleural':0.0}
        
        derivatives = dBiasP | dV | dC | dI | dR
        #print('derivatives--------------')
        #print(derivatives.keys())

        if not return_outputs:
            
            return derivatives

        outputs = pressures
        
        return derivatives, outputs  
    