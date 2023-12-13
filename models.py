from functools import partial

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

import equations as eq

import numpy as np
import time

import matplotlib.pyplot as mpl

# Algorith to solve the ODE system using diffrax and the Equinox Class
@partial(jax.jit, static_argnames=['cpModel','runTime'])
def solveModel(
    cpModel,
    runTime: float,
    states: dict[str,float],
    simParameters,
    startTime: float = 0.0,
):
    
    # Set the parameters for the ODE solver
    #runTime = 1.0
    rtol=1e-3
    atol=1e-6
    dtmin=1e-6
    dtmax=1e-2
    max_steps=16**5
    dt_fixed=1e-3
    dt_dense=1e-2
    t_stabilise=0.0
    fixed_step=True
    dt0 = simParameters['dt']
    
    # Create the ODE solver
    #ode_solver=diffrax.Tsit5()
    ode_solver=diffrax.Euler()
    #ode_solver=diffrax.ImplicitEuler()
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
            dtmin=dtmin,
            dtmax=dtmax,
            pcoeff=0.4,
            icoeff=0.3,
            dcoeff=0.0,
        )
        dt0 = simParameters['dt']

    # Solve the ODE system
    res = diffrax.diffeqsolve(
        term,
        ode_solver,
        startTime,
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

def runSolver(cpModel, states, simParameters, pressureRes, runsRes, trun, totalRuns = 1, runTime = 1, save=True, printTime = True): 
    for nruns in range(totalRuns):
        
        
        # Run the model with the updated states
        res, t_dense, y_dense, deriv, out = solveModel(cpModel,runTime,states, simParameters)

        if pressureRes == {}:
            pressureRes = {key: np.array([]) for key in out.keys()} 
            runsRes = {key: np.array([]) for key in y_dense.keys()} 
        # Concatenate the results of the runs to the previous results
        if save:
            pressureRes = {key: np.concatenate((pressureRes[key] , np.array(value[1:])),axis = 0) for key, value in out.items()}  
            runsRes = {key: np.concatenate((runsRes[key] , np.array(value[1:])),axis = 0) for key, value in y_dense.items()} 

        if (printTime and save):
            now = time.time()
            print(str(np.round(now - trun, 4)) + ' -> Run:' + str(nruns) + ' of ' + str(totalRuns-1) + ' completed!')
        if (printTime and (not save)): 
            now = time.time()
            print(str(np.round(now - trun, 4)) + ' -> Ignored Run:' + str(nruns) + ' of ' + str(totalRuns-1) + ' completed!')
        trun = time.time()

        # Update the states with the last values of the previous run
        states = {key: float(y_dense[key][-1]) for key in states.keys()}
        states['T0'] = states['T']

    return pressureRes, runsRes, trun, states

# Equinox class holding the system of equations for the CardioPulmonary Model to be used by JAX
class CardioPulmonaryModel(eqx.Module):

    states: dict[str,float]

    capacitors: dict[str, eq.Capacitor]
    regions : dict[str, eq.Capacitor]
    resistors : dict[str, eq.Resistor]
    inductors : dict[str, eq.Resistor]
    membraneResistors : dict[str, eq.Resistor]
    partialPressures : dict[str, eq.GasTransport]
    connections : dict[str, eq.Connections]
    cycles : dict[str, eq.Cycle]
    timekeeping : dict[str, eq.PeriodicTrigger]
    resistanceVariations : dict[str, eq.NoController]
    capacityVariations : dict[str, eq.NoController]
    elastanceVariations : dict[str, eq.NoController]
    integrators : dict[str, eq.CycleIntegrator]
    averages : dict[str, eq.CycleAverage]

    def __init__(self, states, modelObjects):
        self.states = states

        self.capacitors = modelObjects['capacitors']
        self.regions = modelObjects['regions']
        self.resistors = modelObjects['resistors']
        self.inductors = modelObjects['inductors']###
        self.membraneResistors = modelObjects['membraneResistors']
        self.partialPressures = modelObjects['partialPressures']
        self.connections = modelObjects['connections']
        self.cycles = modelObjects['cycles']
        self.timekeeping = modelObjects['timekeeping']
        self.resistanceVariations = modelObjects['resistanceVariations']
        self.capacityVariations = modelObjects['capacityVariations']
        self.elastanceVariations = modelObjects['elastanceVariations']###
        self.integrators = modelObjects['integrators']
        self.averages = modelObjects['averages']


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
        
        # Calculates all the pressures and adds them to a dictionary ##############################################
        pressuresRegions = {key: capacitor.pressure(t,y) for key, capacitor in self.regions.items()}
        pressuresOriginal = {key: capacitor.pressure(t,y) for key, capacitor in self.capacitors.items()}

        # Adds the partial pressures to the pressures dictionary
        partialPressures = {key: y[key] for key in self.partialPressures.keys()}
        #pressures = {**pressuresOriginal, **partialPressures, **pressuresRegions}
        pressures = pressuresOriginal | partialPressures | {key: y[key] for key, capacitor in self.regions.items()}

        # Calculates the elastance derivatives and adds them to a dictionary ######################################
        dE = {value.varToControlIdx: value.dControl(y) for value in self.elastanceVariations.values()}

        # Calculates the flows and adds them to a dictionary ######################################################
        flow = {value.flowIdx: value.flow_rate(t,y,pressures) for value in self.resistors.values()}
        membraneFlow = {value.flowIdx: value.flow_rate(t,y,pressures) for value in self.membraneResistors.values()}
        indFlow = {value.flowIdx: y[value.flowIdx] for value in self.inductors.values()}
        allFlows = flow | membraneFlow | indFlow
        yTmp = y | pressuresOriginal | allFlows

        dQ = {value.flowIdx: value.flow_rate_deriv(t,y,pressures) for value in self.inductors.values()}

        # TODO Calculates the derivatives of the resistors ########################################################
        dR = {value.varToControlIdx: value.dControl(yTmp) for value in self.resistanceVariations.values()}

                # TODO Calculates the derivatives of the capacitors #######################################################
        dC = {value.varToControlIdx: value.dControl(yTmp) for value in self.capacityVariations.values()}

        # Calculates the derivatives of the volumes ################################################################
        dV = {key: value.dV(yTmp) for key, value in self.connections.items()}  

        # Calculates the derivatives of the partial pressures ######################################################
        dP = {key: value.dP(yTmp,allFlows,dV) for key, value in self.partialPressures.items()}

        # TODO Calculates the derivatives of the bias pressures (Pleural for now) ##################################

        # Calculates the derivatives for the times #################################################################
        tt = {
            'T': 1.0,
            'T0': 0.0,
        }

        # Calculates the derivatives for the triggers and timers and cycles #######################################
        dTrigger = {('trigger' + key): value.trigger(y,t) for key, value in self.timekeeping.items()}
        dTimer = {('timer' + key): value.timer(y,t) for key, value in self.timekeeping.items()}
        dCycles = {key: value.dCycle(y,t) for key, value in self.cycles.items()}

        # Calculates the derivatives for the integrators ##########################################################
        dI = {key: value.process(yTmp,t) for key, value in self.integrators.items()}
        dA = {key: value.process(yTmp,t) for key, value in self.averages.items()}
        
        

        ##########################################################################################################
        # Concatenate all the derivatives to a single dictionary #################################################
        derivatives = dV | dC | dE | dR | dQ | dP | tt | dTrigger | dTimer | dCycles | pressuresRegions | dA | dI
        #derivatives = dV | dC | dE | dR | dQ | tt | dTrigger | dTimer | dCycles | pressuresRegions # | dAverages | dIntegrators


        if not return_outputs:
            
            return derivatives

        outputs = pressures | flow | membraneFlow
        
        return derivatives, outputs  
    