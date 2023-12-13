import numpy as np
import modelGenerator
import models
import time
import jax
import jax.numpy as jnp
import HDF5API
import matplotlib.pyplot as mpl
import matplotlib.gridspec as gridspec
import plots

#███    ███  ██████  ██████  ███████ ██           ██████  ██████  ███    ██ ███████     ███████ ████████ ██    ██  ██████ ████████ 
#████  ████ ██    ██ ██   ██ ██      ██          ██      ██    ██ ████   ██ ██          ██         ██    ██    ██ ██         ██    
#██ ████ ██ ██    ██ ██   ██ █████   ██          ██      ██    ██ ██ ██  ██ █████       ███████    ██    ██    ██ ██         ██    
#██  ██  ██ ██    ██ ██   ██ ██      ██          ██      ██    ██ ██  ██ ██ ██               ██    ██    ██    ██ ██         ██    
#██      ██  ██████  ██████  ███████ ███████      ██████  ██████  ██   ████ ██          ███████    ██     ██████   ██████    ██    


def modelStructureAndParameters(gasExchange=True,control=True,dcComponent=0.0,amplitude=0.0):

    simulationParameters = {
        'dt': 0.001, # sets the integration step duration, the output is always 0.01 step
    }
    
    # Used to name the prefixes for naming of variables in the model
    prefixes = {
        'flow': 'Q_',
        'pressure': 'P_',
        'resistor': 'R_',
        'capacitor': 'C_',
        'volume': 'V_',
    }

    modelSwitches = {
        'gasExchange': gasExchange,
        'control': control,
    }

    # Not in use yet. if you need to change the total volume go to the capacitors y0 instead
    params = {
        'totalBloodVolume' : 4000.0, # Total Blood Volume not in use yet
        'totalLungVolume' : 2000.0,  # mL
        'airViscosity': modelGenerator.pa2mmHg(1.81e-5),  # kg/(m*s)  Pa*s (converted to mmHg)
        'bloodViscosity': modelGenerator.pa2mmHg(3.5e-3), # kg/(m*s)  Pa*s (converted to mmHg)
    }

    # To go to ventilator mode make the dcComponent and amplitude different from 0 and the Thorax elastances constant
    ventilatorParams = {
        'inputName': 'Ventilator',
        #'dcComponent': 12.0,  # baseline prassure mmHg
        #'amplitude': 7.0,  # breath amplitude mmHg
        'dcComponent': dcComponent,  # baseline prassure mmHg
        'amplitude': amplitude,  # breath amplitude mmHg
        'I': 1,
        'E': 3,
        'slopeFraction': 0.2,
    }

    # Names for the nodes to use in the model
    compartmentNames = ['C1', 'C2', 'Thx','Atm','Ven']

    ############################################### Connectivity Matrix  ####################################################
    #                               C1 C2 Thx Atm Ven
    connectivityMatrix = np.array([ 0, 0, 0, 0, 0, #C1
                                    0, 0, 0, 0, 0, #C2
                                    0, 0, 0, 0, 0, # Thr
                                    0, 0, 0, 0, 0, # Atm
                                    0, 0, 0, 0, 0, # Ven
                                ]).reshape(5, 5)


    ############################################### Resistor Parameters  ####################################################
    # Reisitors parameters (Names and number of resistors should be calculated by the connectivityMatrix) 
    resistorNames = ['C1C2']

    #                                     R       L       y0  type
    resistorParamsMatrix = np.array([   0.005, 0.00000, 0.000, 1,  # AsCs
                                        ]).reshape(1, 4)

    ############################################### Capacitor Parameters  ####################################################
    #                                      C       V0     Emax    Emin   Trise    y0   type
    capacitorsParamsMatrix = np.array([ 100.0000, 0000.0, 0.0000, 0.0000, 0.0000, 1000.0, 2,  # C1
                                        100.0000, 0000.0, 0.0000, 0.0000, 0.0000, 1000.0, 1,  # C2
                                        20.00, 1000.0, 1000.0, 1000.0, 0.0000, 0.0000, 4,  # Thx 
                                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 5,  # Atm
                                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 6,  # Ven
                                        ]).reshape(5, 7)

    ############################################### Pressure Bias Map  ####################################################    
    #                                       C1 C2 Thx 
    connectivityPresBiasMatrix = np.array([ 0, 0, 0, 0, 0, #C1
                                            0, 0, 0, 0, 0, #C2
                                            0, 1, 0, 0, 0, # Thr
                                            1, 0, 1, 1, 1, # Atm
                                            0, 0, 0, 0, 0, # Ven
                                ]).reshape(5, 5)


    ############################################### Region Volume Map  ####################################################
    #                                       C1 C2 Thx 
    connectivityRegionVolMatrix = np.array([ 0, 0, 0, 0, 0, #C1
                                             0, 0, 0, 0, 0, #C2
                                             0, 1, 0, 0, 0, # Thr
                                             0, 0, 0, 0, 0, # Atm
                                             0, 0, 0, 0, 0, # Ven
                                ]).reshape(5, 5)



    ############################################### Membrane Resisros Map Map  ############################################
    #                                           C1 C2 Thx 
    connectivityMemResistorsMatrix = np.array([ 0, 0, 0, 0, 0, #C1
                                                1, 0, 0, 0, 0, #C2
                                                0, 0, 0, 0, 0, # Thr
                                                0, 0, 0, 0, 0, # Atm
                                                0, 0, 0, 0, 0, # Ven
                                ]).reshape(5, 5)


    ############################################### Cycles and Durations  #################################################

    cycles = ['RC']
    cyclesDuration = [5.0]
        #                                C1 C2 Thx Atm 
    cyclesDistributionMatrix = np.array([0, 0,  1,  0, 0  # RC
                                    ]).reshape(1, 5)
    
    ############################################### Pressure Bias Map  ####################################################

    # Regions of different gas concentrations
    gasRegions = ['Atmosphere','Alveoli','Arterial', 'Venous', 'Tissues']
    # Gas species to consider
    gases = ['O2','C2','N2']

    # Gases partial pressures by region
    #                                O2      C2    N2    Total state
    gasPartialPressures = np.array([145.0, 7.000, 608.0, 760.0,  1,  # Atmosphere
                                    87.00, 40.00, 608.0, 760.0,  1,  # Alveoli
                                    81.60, 40.00, -1.00, 760.0,  2,  # Arterial
                                    71.50, 46.00, -1.00, 760.0,  2,  # Venous
                                    30.00, 50.00, -1.00, 760.0,  2,  # Tissues
                                ]).reshape(5, 5)

    # Membrane Reisitors parameters (Names and number of resistors should be calculated by the connectivityMemResistorsMatrix) 
    memResistorNames = ['O2_C1C2']
    #                                     R       L       y0  type
    resistorMemParamsMatrix = np.array([0.01, 0.00000, 0.000, 1,  # CpLa_O2
                                        ]).reshape(1, 4)

    # Region connbectivity with the nodes for partial pressure distribution
    #                                   C1 C2 Thx Atm
    gasDistributionMatrix = np.array([  1, 0, 0, 1, 1,   # Atmosphere
                                        0, 1, 0, 0, 0,   # Alveoli
                                        0, 0, 0, 0, 0,   # Arterial
                                        0, 0, 0, 0, 0,   # Venous
                                        0, 0, 0, 0, 0,   # Tissues
                                    ]).reshape(5, 5)
    
    ############################################### Controllers  ##########################################################
    #                      period  varName varInt Type 
    integrators = np.array([
                    ]).reshape(0, 0)
    
    #                 AverageVar  ofVar    AtmPress Period type
    averages = np.array([ ]).reshape(0, 0)
    
    #                       TargetVar ToCtrllVar TargetVal MinVal MaxVal  Kp    Kd  type
    controllers = np.array([]).reshape(0, 0)

    modelStructure = {
            'prefixes': prefixes,
            'modelSwitches': modelSwitches,
            'params': params,
            'ventilatorParams': ventilatorParams,
            'compartmentNames': compartmentNames,
            'connectivityMatrix': connectivityMatrix,
            'resistorNames': resistorNames,
            'resistorParamsMatrix': resistorParamsMatrix,
            'capacitorsParamsMatrix': capacitorsParamsMatrix,
            'connectivityPresBiasMatrix': connectivityPresBiasMatrix,
            'connectivityRegionVolMatrix': connectivityRegionVolMatrix,
            'connectivityMemResistorsMatrix': connectivityMemResistorsMatrix,
            'gasRegions': gasRegions,
            'gases': gases,
            'gasPartialPressures': gasPartialPressures,
            'memResistorNames': memResistorNames,
            'resistorMemParamsMatrix': resistorMemParamsMatrix,
            'gasDistributionMatrix': gasDistributionMatrix,
            'cycles': cycles,
            'cyclesDuration': cyclesDuration,
            'cyclesDistributionMatrix': cyclesDistributionMatrix,
            'simulationParameters': simulationParameters,
            'averages': averages,
            'integrator': integrators,
            'controllers': controllers,
        }
    
    return modelStructure

def runSimulation(states, modelObjects, runTime, totalRunsToIgnore, totalRuns, pressureRes, runsRes, structures):
    trun = time.time()
    # Create the model object with the parameters and the equations to be solved by JAX
    cpModel = models.CardioPulmonaryModel(states, modelObjects)

    ###############################################################
    # Second run to allow the system to stalibilse
    if totalRunsToIgnore > 0:
        pressureRes, runsRes, trun, states = models.runSolver(
                cpModel, 
                states, 
                structures['modelStructure']['simulationParameters'],
                pressureRes, 
                runsRes, 
                trun, 
                totalRunsToIgnore, 
                runTime, 
                save = False, 
                printTime = True
            )
    ###############################################################
    # TODO add results to HDF5 file
    # Third run to get the results
    pressureRes, runsRes, trun, states = models.runSolver(
            cpModel, 
            states, 
            structures['modelStructure']['simulationParameters'],
            pressureRes, 
            runsRes, 
            trun, 
            totalRuns, 
            runTime, 
            save = True, 
            printTime = True
        )

    return pressureRes, runsRes, trun, states


with jax.default_device(jax.devices("cpu")[0]):
        jax.config.update("jax_enable_x64", True)

        runTime = 15
        totalRunsToIgnore = 0
        totalRuns = 2

        modelStructure = modelStructureAndParameters(gasExchange=True,control=True,dcComponent=0.0,amplitude=0.0)

        states, modelObjects, structures = modelGenerator.modelInit(modelStructure)
        pressureRes, runsRes, trun, states = runSimulation(states, modelObjects, runTime=runTime, totalRunsToIgnore=totalRunsToIgnore, totalRuns=totalRuns, pressureRes={}, runsRes={}, structures=structures)

totalTime = runTime*(totalRuns)

sampPeriod = 0.01

atmPressure = states['P_Atm']
results = pressureRes | runsRes

t = np.arange(0, totalTime, sampPeriod)
fig = mpl.figure(figsize=(15, 10))
grid = gridspec.GridSpec(3, 2, figure=fig)

################################################################################################################################
ax_1 = fig.add_subplot(grid[0, 0])
plots.buildLeftRightAxisDependentScales(ax_1,results,'V_C1', 'V_C2', 'C1 and C2 Volumes','Volume C1 [mL]','Volume C2 [mL]','tab:blue','tab:red',0.0,t)

ax_2 = fig.add_subplot(grid[1, 0])
plots.buildLeftRightAxisIndependentPressureScales(ax_2,results,'P_C1', 'P_C2', 'C1 and C2 Pressures','Pressure C1 [mmHg]','Pressure C2 [mmHG]','tab:blue','tab:red',760,t)
ax_2.set_xlabel('Time [s]')

ax_3 = fig.add_subplot(grid[2, 0])
plots.buildLeftRightAxisIndependentScales(ax_3,results,'E_C_Thx', 'e_C_Thx', 'Thorax','Max Vol','Min Vol','tab:blue','tab:red',0.0,t)
ax_3.set_xlabel('Time [s]')

ax_4 = fig.add_subplot(grid[0, 1])
plots.buildLeftRightAxisDependentScales(ax_4,results,'P_C2_C1', 'P_C2_C2', 'Carbon dioxide','C1','C2','tab:blue','tab:red',0.0,t)
ax_4.set_xlabel('Time [s]')

ax_5 = fig.add_subplot(grid[1, 1])
plots.buildLeftRightAxisDependentScales(ax_5,results,'P_O2_C1', 'P_O2_C2', 'Oxygen','C1','C2','tab:blue','tab:red',0.0,t)
ax_5.set_xlabel('Time [s]')

ax_6 = fig.add_subplot(grid[2, 1])
plots.buildLeftRightAxisDependentScales(ax_6,results,'P_N2_C1', 'P_N2_C2', 'Nitrogen','C1','C2','tab:blue','tab:red',0.0,t)
ax_6.set_xlabel('Time [s]')

'''
ax_3 = fig.add_subplot(grid[4, 0])
results['uVol'] = calculateThoraxUnstressedVolume(modelObjects,results)
buildLeftRightAxisIndependentScales(ax_3,results,'P_Ven','uVol','Ventilator Pressure [mmHg] and Unstressed Volume of Thorax [mL]','Ventilator','Thorax','tab:blue','tab:cyan',atmPressure,t)
ax_3.set_xlabel('Time [s]')

ax_4 = fig.add_subplot(grid[5, 0])
buildLeftRightAxisIndependentPressureScales(ax_4,results,'P_La','P_Thx','Lung Pressures [mmHg]','Alveoli','Thorax','tab:green','tab:blue',atmPressure,t)
ax_4.set_xlabel('Time [s]')

ax_9 = fig.add_subplot(grid[6, 0])
results['V_Thx'] = np.array(results['V_Vt'])+np.array(results['V_Hr'])+np.array(results['V_Ap'])+np.array(results['V_Cp'])+np.array(results['V_Vp'])+np.array(results['V_Hl'])+np.array(results['V_La'])+np.array(results['V_Lb'])
buildLeftRightAxisIndependentScales(ax_9,results,'V_La','V_Thx','Lung Volumes [mL]','Alveoli','Thorax','tab:green','tab:blue',0.0,t)
ax_9.set_xlabel('Time [s]')

################################################################################################################################
ax_5 = fig.add_subplot(grid[:-4, 1])
thoraxMouthPressureVolumeAxes(ax_5,results1,atmPressure,sampPeriod,t)

ax_6 = fig.add_subplot(grid[3, 1])
buildLeftRightAxisIndependentScales(ax_6,results1,'C_La', 'V_La', 'Alveoli Complaince and Volume','Compliance [mL/mmHg]','Volume [mL]','tab:olive','tab:green',0.0,t)
ax_6.set_xlabel('Time [s]')

ax_7 = fig.add_subplot(grid[4, 1])
results1['uVol'] = calculateThoraxUnstressedVolume(modelObjects,results1)
buildLeftRightAxisIndependentScales(ax_7,results1,'P_Ven','uVol','Ventilator Pressure [mmHg] and Unstressed Volume of Thorax [mL]','Ventilator','Thorax','tab:blue','tab:cyan',atmPressure,t)
ax_7.set_xlabel('Time [s]')

ax_8 = fig.add_subplot(grid[5, 1])
buildLeftRightAxisIndependentPressureScales(ax_8,results1,'P_La','P_Thx','Lung Pressures [mmHg]','Alveoli','Thorax','tab:green','tab:blue',atmPressure,t)
ax_8.set_xlabel('Time [s]')

ax_10 = fig.add_subplot(grid[6, 1])
results1['V_Thx'] = np.array(results1['V_Vt'])+np.array(results1['V_Hr'])+np.array(results1['V_Ap'])+np.array(results1['V_Cp'])+np.array(results1['V_Vp'])+np.array(results1['V_Hl'])+np.array(results1['V_La'])+np.array(results1['V_Lb'])
buildLeftRightAxisIndependentScales(ax_10,results1,'V_La','V_Thx','Lung Volumes [mL]','Alveoli','Thorax','tab:green','tab:blue',0.0,t)
ax_10.set_xlabel('Time [s]')
'''
################################################################################################################################
mpl.tight_layout()
mpl.show()