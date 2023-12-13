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
    compartmentNames = ['As', 'Cs', 'Vs', 'Vt', 'Hr', 'Ap', 'Cp', 'Vp', 'Hl', 'Lt', 'Lb', 'La', 'Atm', 'Thx', 'Plr', 'Tis', 'Ven', 'Per']

    ############################################### Connectivity Matrix  ####################################################
    #                               As Cs Vs Vt Hr Ap Cp Vp Hl Lt Lb La Atm Thx Plr Tis Ven col/row
    connectivityMatrix = np.array([ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # As
                                    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Cs
                                    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Vs
                                    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Vt
                                    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Hr
                                    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Ap
                                    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Cp
                                    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Vp
                                    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Hl
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,  0,  0,  0,  0,  0,  0,  # Lt
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,  0,  0,  0,  0,  0,  0,  # Lb
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # La
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Atm
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Thx
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Plr
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Tis
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,  0,  0,  0,  0,  0,  0,  # Ven
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Per
                                        ]).reshape(18, 18)


    ############################################### Resistor Parameters  ####################################################
    # Reisitors parameters (Names and number of resistors should be calculated by the connectivityMatrix) 
    resistorNames = ['AsCs', 'CsVs', 'VsVt', 'VtHr', 'HrAp', 'ApCp', 'CpVp', 'VpHl', 'HlAs', 'VenLt', 'LtLb', 'LbLa']

    #                                     R       L       y0  type
    resistorParamsMatrix = np.array([   0.8133, 0.00000, 0.000, 1,  # AsCs #!! controlled
                                        0.080, 0.00000, 0.000, 1,  # CsVs
                                        0.060, 0.00000, 0.000, 1,  # VsVt
                                        0.010, 0.00001, 0.000, 0,  # VtHr
                                        0.003, 0.00010, 0.000, 0,  # HrAp
                                        0.0637, 0.00000, 0.000, 1,  # ApCp #!! controlled
                                        0.005, 0.00000, 0.000, 1,  # CpVp #!! -0.005
                                        0.005, 0.00001, 0.000, 0,  # VpHl #!! -0.005
                                        0.006, 0.00010, 0.000, 0,  # HlAs
                                        0.003, 0.00100, 0.000, 1,  # VenLt
                                        0.005, 0.00000, 0.000, 1,  # LtLb
                                        0.002, 0.00000, 0.000, 1,  # LbLa
                                        ]).reshape(12, 4)
    '''
    # Heldt Resistors
    resistorParamsMatrix = np.array([   0.820, 0.00000, 0.000, 1,  # AsCs
                                        0.080, 0.00000, 0.000, 1,  # CsVs
                                        0.060, 0.00000, 0.000, 1,  # VsVt
                                        0.017, 0.00001, 0.000, 0,  # VtHr
                                        0.003, 0.00010, 0.000, 0,  # HrAp
                                        0.080, 0.00000, 0.000, 1,  # ApCp
                                        0.010, 0.00000, 0.000, 1,  # CpVp
                                        0.010, 0.00001, 0.000, 0,  # VpHl
                                        0.006, 0.00010, 0.000, 0,  # HlAs
                                        0.003, 0.00000, 0.000, 1,  # VenLt
                                        0.005, 0.00000, 0.000, 1,  # LtLb
                                        0.002, 0.00000, 0.000, 1,  # LbLa
                                        ]).reshape(12, 4)
    '''


    ############################################### Capacitor Parameters  ####################################################
    #                                      C       V0     Emax    Emin   Trise    y0   type
    capacitorsParamsMatrix = np.array([ 2.0000, 400.00, 0.0000, 0.0000, 0.0000, 593.43, 2,  # As
                                        15.000, 0.0000, 0.0000, 0.0000, 0.0000, 312.07, 1,  # Cs
                                        300.00, 0.0000, 0.0000, 0.0000, 0.0000, 2601.9, 1,  # Vs #!!
                                        17.000, 000.00, 0.0000, 0.0000, 0.0000, 108.48, 1,  # Vt
                                        0.0000, 0.0000, 0.4169, 0.0500, 0.0000, 74.982, 0,  # Hr 
                                        4.0000, 090.00, 0.0000, 0.0000, 0.0000, 191.95, 2,  # Ap
                                        4.0000, 0.0000, 0.0000, 0.0000, 0.0000, 73.774, 1,  # Cp
                                        8.000, 00.000, 0.0000, 0.0000, 0.0000, 142.29, 1,  # Vp
                                        0.0000, 0.0000, 1.2076, 0.1000, 0.0000, 99.717, 0,  # Hl 
                                        0.5000, 75.000, 0.0000, 0.0000, 0.0000, 75.007, 2,  # Lt ####
                                        1.0000, 75.000, 0.0000, 0.0000, 0.0000, 77.735, 2,  # Lb ####
                                        910.36, 0000.0, 0.0000, 0.0000, 0.0000, 2500.2, 2,  # La ####
                                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 760.00, 5,  # Atm
                                        100.00, 4000.0, 4472.1, 3538.5, 0.0000, 757.30, 4,  # Thx 
                                        4000.0, 3000.0, 0.0000, 0.0000, 0.0000, 757.30, 3,  # Plr
                                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 760.00, 5,  # Tis
                                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 760.00, 6,  # Ven
                                        70.0, 000.0, 0.0000, 0.0000, 0.0000, 0.0000, 3,  # Per

                                        ]).reshape(18, 7)
    '''
    # Heldt Capacitors
    capacitorsParamsMatrix = np.array([ 2.0000, 150.00, 0.0000, 0.0000, 0.0000, 272.09, 2,  # As
                                        10.000, 0.0000, 0.0000, 0.0000, 0.0000, 155.65, 1,  # Cs
                                        112.00, 0.0000, 0.0000, 0.0000, 0.0000, 2604.3, 1,  # Vs
                                        17.000, 00.000, 0.0000, 0.0000, 0.0000, 528.22, 1,  # Vt
                                        0.0000, 0.0000, 0.8333, 0.0500, 0.0000, 000.98, 0,  # Hr ***
                                        4.3000, 50.000, 0.0000, 0.0000, 0.0000, 126.23, 2,  # Ap
                                        4.0000, 0.0000, 0.0000, 0.0000, 0.0000, 104.74, 1,  # Cp
                                        8.400, 00.000, 0.0000, 0.0000, 0.0000, 181.36, 1,  # Vp
                                        0.0000, 0.0000, 2.5000, 0.1000, 0.0000, 026.32, 0,  # Hl ***
                                        2.4000, 75.000, 0.0000, 0.0000, 0.0000, 75.210, 2,  # Lt ####
                                        3.0000, 75.000, 0.0000, 0.0000, 0.0000, 75.780, 2,  # Lb ####
                                        0125.0, 1000.0, 0.0000, 0.0000, 0.0000, 1054.5, 2,  # La ####
                                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 5,  # Atm
                                        000.00, 4500.0, 0.0060, E_minThx, 0.0000, 0.0000, 4,  # Thx 
                                        1000.0, 0100.0, 0.0000, 0.0000, 0.0000, 0.0000, 3,  # Plr
                                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 5,  # Tis
                                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 6,  # Ven

                                        ]).reshape(17, 7)
    '''


    ############################################### Pressure Bias Map  ####################################################    
    #                                       As Cs Vs Vt Hr Ap Cp Vp Hl Lt Lb La Atm Thx Plr Tis Ven
    connectivityPresBiasMatrix = np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # As
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Cs
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Vs
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Vt
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Hr
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Ap
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Cp
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Vp
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Hl
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Lt
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Lb
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # La
                                            1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,  1,  1,  0,  1,  1,  0,  # Atm
                                            0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1,  0,  0,  1,  0,  0,  1,  # Thx
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Plr
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Tis
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Ven
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Per
                                        ]).reshape(18, 18)


    ############################################### Region Volume Map  ####################################################
    #                                       As Cs Vs Vt Hr Ap Cp Vp Hl Lt Lb La Atm Thx Plr Tis Ven
    connectivityRegionVolMatrix = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # As
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Cs
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Vs
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Vt
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Hr
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Ap
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Cp
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Vp
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Hl
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Lt
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Lb
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # La
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Atm
                                            0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1,  0,  0,  0,  0,  0,  0,  # Thx
                                            0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1,  0,  0,  0,  0,  0,  0,  # Plr
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Tis
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Ven
                                            0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Per
                                        ]).reshape(18, 18)


    ############################################### Membrane Resisros Map Map  ############################################
    #                                           As Cs Vs Vt Hr Ap Cp Vp Hl Lt Lb La Atm Thx Plr Tis Ven
    connectivityMemResistorsMatrix = np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,   0, # As
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Cs
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Vs
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Vt
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Hr
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Ap
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Cp
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Vp
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Hl
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Lt
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Lb
                                                0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # La
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Atm
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Thx
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Plr
                                                0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Tis
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Ven
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Per
                                        ]).reshape(18, 18)

    ############################################### Cycles and Durations  #################################################

    cycles = ['HC', 'RC']
    #cyclesDuration = [0.9561, 2.8685]
    cyclesDuration = [0.8, 5.0]
    #                                    As Cs Vs Vt Hr Ap Cp Vp Hl Lt Lb La Atm Thx Plr Tis Ven
    cyclesDistributionMatrix = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,  0,  0,  0,  0,  0,  # HC
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  1,  0,  0,  1,  # RC
                                    ]).reshape(2, 17)
    
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
    memResistorNames = ['O2_CpLa', 'C2_CpLa', 'O2_CsTiss', 'C2_CsTiss']
    #                                     R       L       y0  type
    resistorMemParamsMatrix = np.array([3.000/6, 0.00000, 0.000, 1,  # CpLa_O2
                                        1.200/6, 0.00000, 0.000, 1,  # CpLa_C2
                                        3.000/6, 0.00000, 0.000, 1,  # CsTiss_O2
                                        1.200/6, 0.00000, 0.000, 1,  # CsTiss_C2
                                    ]).reshape(4, 4)

    # Region connbectivity with the nodes for partial pressure distribution
    #                                   As Cs Vs Vt Hr Ap Cp Vp Hl Lt Lb La Atm Thx Plr Tis Ven
    gasDistributionMatrix = np.array([  0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,  1,  0,  0,  0,  1,  1,  # Atmosphere
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,  0,  0,  0,  0,  0,  0,  # Alveoli
                                        1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Arterial
                                        0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  # Venous
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  1,  0,  0,  # Tissues
                                    ]).reshape(5, 18)
    
    ############################################### Controllers  ##########################################################
    #                      period  varName varInt Type 
    integrators = np.array(['HC', 'i_Hl',  'Q_HlAs',  0,  # i_Hl
                            'HC', 'i_COl', 'i_Hl',  1,  # i_COl
                            'HC', 'i_Hr',  'Q_HrAp',  0,  # i_Hr
                            'HC', 'i_COr', 'i_Hr',  1,  # i_COr
                            'RC', 'i_MaxLa',  'V_La',  2,  # i_RC
                            'RC', 'i_Max',  'i_MaxLa',  1,  # i_TV
                            'RC', 'i_MinLa',  'V_La',  3,  # i_RC
                            'RC', 'i_Min',  'i_MinLa',  1,  # i_TV
                            'HC', 'i_MaxAp',  'P_Ap',  2,  # i_RC
                            'HC', 'i_Max_Ap',  'i_MaxAp',  1,  # i_TV
                            'HC', 'i_MinAp',  'P_Ap',  3,  # i_RC
                            'HC', 'i_Min_Ap',  'i_MinAp',  1,  # i_TV
                            'HC', 'i_MaxAs',  'P_As',  2,  # i_RC
                            'HC', 'i_Max_As',  'i_MaxAs',  1,  # i_TV
                            'HC', 'i_MinAs',  'P_As',  3,  # i_RC
                            'HC', 'i_Min_As',  'i_MinAs',  1,  # i_TV
                    ]).reshape(16, 4)
    
    #                 AverageVar  ofVar    AtmPress Period type
    averages = np.array(['a_As', 'P_As',   'P_Atm',  15.0,   0,  # As
                         'a_Ap', 'P_Ap',   'P_Atm',  15.0,   0,  # Ap
                         'a_COl', 'i_COl', 70.8,  1.0,   1,  # COl
                         'a_COr', 'i_COr', 68.5,  1.0,   1,  # COr
                         'a_Min', 'i_Min', 2500.7,  1.0,   1,  # TV
                         'a_Max', 'i_Max', 2992.3,  1.0,   1,  # TV
                    ]).reshape(6, 5)
    
    #                       TargetVar ToCtrllVar TargetVal MinVal MaxVal  Kp    Kd  type
    controllers = np.array([]).reshape(0, 0)
    '''
    controllers = np.array([ 'a_As',   'R_AsCs',   85.0,    0.20,  2.00,  0.01, 3.0,  0,  # As
                             'a_Ap',   'R_ApCp',   20.0,    0.01,  0.60,  0.01, 3.0,  0,  # Ap
                             'a_COl',  'E_C_Hl',   85.0,    0.02,  40.0,  0.01, 3.0,  0,  # E_Hl
                             'a_COr',  'E_C_Hr',   85.0,    0.02,  20.0,  0.01, 3.0,  0,  # E_Hr
                             'V_La',   'C_La',    2500.0,   1000.0, 55.0,  0.002, 3.0,  2,  # C_La
                             'V_Hr',   'e_C_Hr',   100.0,   0.03,   0.3,  0.002, 3.0,  2,  # C_La
                             #'a_Min',  'e_C_Thx', 2500.0,   0.0001, 0.009, 0.01,  3.0,  0,  # E_Thx
                             #'a_Max',  'E_C_Thx', 3000.0,   0.005, 0.05, 0.01,  3.0,  0,  # E_Thx
                             #'a_VLa',   'C_La',    1500.0,   75.0,  250.0, 0.004,  3.0,  2,  # C_La
                        ]).reshape(6, 8)
    '''

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

def calculateDoubleSigmoid(idx, structures):
    structure = structures[idx]
    
    targetValue = float(structure[2])
    minValueToControl = float(structure[3])
    slope = float(structure[5])
    slope1 = float(structure[6])

    step = 1
    separation = 0.4
    
    duration = float(structure[2])*2
    V0 = np.arange(0,duration,step)
    sigmoid = []
    sigmoid1 = []
    
    offset = minValueToControl
    inflectionPoint = targetValue
    dist = inflectionPoint * separation
    amplitude = 0.0 - minValueToControl
    
    
    for i in V0:
        #sigmoid.append(amplitude / (1 + np.exp(-slope * (i - inflectionPoint-dist))) + offset)
        #sigmoid1.append(amplitude / (1 + np.exp(slope * (i - inflectionPoint+dist))) + offset)

        sigmoid.append(amplitude / (1 + np.exp(-slope * (i - inflectionPoint-dist))) + offset)
        sigmoid1.append(amplitude / (1 + np.exp(slope1 * (i - inflectionPoint+dist))) + offset)

    difference = np.abs(np.array(sigmoid) - np.array(sigmoid1))
    intersection = np.where(difference == np.min(difference))[0][0]
    mergedSigmoids = sigmoid1[0:int(intersection)] + sigmoid[int(intersection)-1:-1]

    structures[idx][4] = intersection
    structures[idx][7] = 3
    '''
    mpl.plot(V0,sigmoid, color='tab:red', label = 'Sigmoid')
    mpl.plot(V0,sigmoid1, color='tab:blue', label = 'Sigmoid1')
    mpl.plot(V0, mergedSigmoids, color='tab:green', label = 'Merged')
    #mpl.plot(V0,doubleCosine, color='tab:green', label = 'DoubleCosine')
    #mpl.plot(np.array(results[structure[0]]), np.array(results[structure[1]]), color='tab:blue', label = structure[1])
    mpl.title(structure[1] + ' Max: ' + str(minValueToControl) + ' Target: ' + str(targetValue) )
    mpl.ylabel('Compliance [mL/mmHg]')
    mpl.xlabel('Volume [mL]')
    mpl.legend()
    mpl.tight_layout()
    mpl.show()
    '''

    return structures

#'dcComponent': 12.0,  # baseline prassure mmHg
#'amplitude': 7.0,  # breath amplitude mmHg
#'E_minThx': 0.002,  # minimum elastance of the thorax if self-breathing
#'E_minThx': 0.006,  # minimum elastance of the thorax if NOT self-breathing
# TODO-BUG: Gas exchange: is not working for the mode with ventilator
# TODO-BUG: Gas exchange: Partial Pressure of Nitrogen is not making sense in self-breathing.
# TODO-BUG: Gas exchange: Partial Pressures on the alveoli do not add up to Atmospheric pressure

#███████ ███████ ██      ███████       ██████  ██████      ██ ██    ██ ███████ ███    ██ ████████ ██ ██       █████  ████████  ██████  ██████  
#██      ██      ██      ██            ██   ██ ██   ██    ██  ██    ██ ██      ████   ██    ██    ██ ██      ██   ██    ██    ██    ██ ██   ██ 
#███████ █████   ██      █████   █████ ██████  ██████    ██   ██    ██ █████   ██ ██  ██    ██    ██ ██      ███████    ██    ██    ██ ██████  
#     ██ ██      ██      ██            ██   ██ ██   ██  ██     ██  ██  ██      ██  ██ ██    ██    ██ ██      ██   ██    ██    ██    ██ ██   ██ 
#███████ ███████ ███████ ██            ██████  ██   ██ ██       ████   ███████ ██   ████    ██    ██ ███████ ██   ██    ██     ██████  ██   ██

def runSelfBreathingVentilatorTest():
    with jax.default_device(jax.devices("cpu")[0]):
        jax.config.update("jax_enable_x64", True)

        runTime = 5
        totalRunsToIgnore = 50
        totalRuns = 1
        
        totalRunsToIgnore1 = 0
        totalRuns1 = 0

        nrLevels = 6

        totalTime = runTime*(totalRuns)
        totalTime1 = runTime*(totalRuns1 * nrLevels)

        totalTime = totalTime + totalTime1

        sampPeriod = 0.01
        atmPressure = 760.0
        
        # Run 1
        modelStructure = modelStructureAndParameters(gasExchange=False,control=True)
        #modelStructure['ventilatorParams']['dcComponent'] = 8.0 #3.67
        #modelStructure['ventilatorParams']['amplitude'] = 7.0 #8.8
        #modelStructure['capacitorsParamsMatrix'][13][2] = modelStructure['capacitorsParamsMatrix'][13][3] # E_C_Thx
        #modelStructure['capacitorsParamsMatrix'][13][0] = 100.0 + 50 # Compliance Thorax
        modelStructure['capacitorsParamsMatrix'][2][0] =  150  # Compliance systemic veins


        '''
        #                                         TargetVar ToCtrllVar TargetVal MinVal MaxVal     Kp    Kd  type
        modelStructure['controllers'] = np.array(['V_La',   'C_La',    3200.0,   1000.0, 000.0,  0.006, 0.006,  2,  # C_La
                                                  'V_Hr',   'e_C_Hr',  180.00,   0.0100, 0.400,  0.050, 3.000,  2,  # e_C_Hr
                                                  'V_Hl',   'e_C_Hl',  120.00,   0.0100, 0.100,  0.050, 3.000,  2,  # e_C_Hl
                            ]).reshape(3, 8)
        modelStructure['controllers'] = calculateDoubleSigmoid(0, modelStructure['controllers'])
        print(modelStructure['controllers'])
        '''
        modelStructure['controllers'] = np.array(['a_As',   'R_AsCs',   86.84,    0.20,  2.00,  0.01, 3.0,  0,  # As
                                                  'a_Ap',   'R_ApCp',   17.0,    0.010,  0.60,  0.01, 3.0,  0,  # Ap
                                                  'a_COl',  'E_C_Hl',   70.0,    0.010,  40.0,  0.02, 3.0,  0,  # E_Hl
                                                  'a_COr',  'E_C_Hr',   70.0,    0.010,  100.0,  0.02, 3.0,  0,  # E_Hr
                                                  'V_La',   'C_La',    3500.0,   600.0, 000.0,  0.004, 0.004,  2,  # C_La
                                                  'V_Hr',   'e_C_Hr',  240.00,   0.0300, 0.250,  0.050, 3.000,  2,  # e_C_Hr
                                                  'V_Hl',   'e_C_Hl',  240.00,   0.0400, 0.100,  0.050, 3.000,  2,  # e_C_Hl
                                                  'a_Min',  'e_C_Thx', 2500.0,   2000.0, 6000, 0.01,  3.0,  0,  # E_Thx
                                                  'a_Max',  'E_C_Thx', 3000.0,   2000.0, 6000, 0.01,  3.0, 0,  # E_Thx
                            ]).reshape(9, 8)
        modelStructure['controllers'] = calculateDoubleSigmoid(4, modelStructure['controllers'])

        
        states, modelObjects, structures = modelGenerator.modelInit(modelStructure)
        pressureRes, runsRes, trun, states = runSimulation(states, modelObjects, runTime=runTime, totalRunsToIgnore=totalRunsToIgnore, totalRuns=totalRuns, pressureRes={}, runsRes={}, structures=structures)
        
        modelStructure['ventilatorParams']['dcComponent'] = -2.0 #3.67
        modelStructure['ventilatorParams']['amplitude'] = 7.0 #8.8
        states['E_C_Thx'] = states['e_C_Thx']
        # modelStructure['capacitorsParamsMatrix'][13][2] = modelStructure['capacitorsParamsMatrix'][13][3]
        #modelStructure['capacitorsParamsMatrix'][13][0] = 100.0 + 50 # Compliance Thorax
        #modelStructure['capacitorsParamsMatrix'][2][0] =  300.0 - 50  # Compliance systemic veins


        #                                         TargetVar ToCtrllVar TargetVal MinVal MaxVal     Kp    Kd  type
        modelStructure['controllers'] = np.array(['V_La',   'C_La',    3500.0,   600.0, 000.0,  0.004, 0.004,  2,  # C_La
                                                  'V_Hr',   'e_C_Hr',  240.00,   0.0100, 0.250,  0.050, 3.000,  2,  # e_C_Hr
                                                  'V_Hl',   'e_C_Hl',  240.00,   0.0200, 0.100,  0.050, 3.000,  2,  # e_C_Hl
                            ]).reshape(3, 8)
        modelStructure['controllers'] = calculateDoubleSigmoid(0, modelStructure['controllers'])
        print(modelStructure['controllers'])
        for i in range(0,nrLevels):
            modelStructure['ventilatorParams']['dcComponent'] = modelStructure['ventilatorParams']['dcComponent'] + 3
            #states['C_Vs'] = 225.0
            ignoredStates, modelObjects, structures = modelGenerator.modelInit(modelStructure)
            pressureRes, runsRes, trun, states = runSimulation(states, modelObjects, runTime=runTime, totalRunsToIgnore=totalRunsToIgnore1, totalRuns=totalRuns1, pressureRes=pressureRes, runsRes=runsRes, structures=structures)
            #pressureRes, runsRes, trun, states = runSimulation(states, modelObjects, runTime=runTime, totalRunsToIgnore=totalRunsToIgnore1, totalRuns=totalRuns1, pressureRes={}, runsRes={}, structures=structures)
            results = pressureRes | runsRes
        plots.plotDoubleSigmoid(0, structures,results)
        plots.plotDoubleSigmoidColorsRC(0, structures,results)


    


    #plotTriggers('HC',results, totalTime, sampPeriod, atmPressure)
    #plotTriggers('RC',results, totalTime, sampPeriod, atmPressure)

    #plots.plotSigmoid(4,results, totalTime, sampPeriod, atmPressure, structures)
    plots.plotSigmoid(1, structures,results)
    plots.plotSigmoid(2, structures,results)
    plots.plotCardioVolumes(results, totalTime, sampPeriod, atmPressure)
    plots.plotPressureVolumeHeart(results, totalTime, sampPeriod, atmPressure,modelObjects)
    plots.plotPressureVolumeAlveoli(results, totalTime, sampPeriod, atmPressure,modelObjects)
    plots.plotcardioFlows(results, totalTime, sampPeriod, atmPressure,modelObjects)

    #plots.plotCardioVolumes(results1, totalTime1, sampPeriod, atmPressure)
    #plots.plotPressureVolumeHeart(results1, totalTime1, sampPeriod, atmPressure,modelObjects)
    #plots.plotPressureVolumeAlveoli(results1, totalTime1, sampPeriod, atmPressure,modelObjects)
    plots.plotControllers(results, totalTime, sampPeriod, atmPressure)

    if structures['modelStructure']['modelSwitches']['gasExchange']:
        plots.plotPartialPressures(results, totalTime, sampPeriod, atmPressure,modelObjects)

    #plots.compareRunsCardioPV(results, results1, modelObjects, totalTime, sampPeriod, atmPressure, 'Self-Breathing Pressure-Volume Curves', 'Ventilator Pressure-Volume Curves')
    #plots.compareRunsLungPV(results, results1, modelObjects, totalTime, sampPeriod, atmPressure, 'Self-Breathing Pressure-Volume Curves', 'Ventilator Pressure-Volume Curves')
    return states,modelObjects,modelStructure

def runPEEPVentilatorTest():
    with jax.default_device(jax.devices("cpu")[0]):
        jax.config.update("jax_enable_x64", True)

        runTime = 5
        totalRunsToIgnore = 10
        totalRuns = 1
        
        totalRunsToIgnore1 = 0
        totalRuns1 = 1

        nrLevels = 11

        totalTime = runTime*(totalRuns)
        totalTime1 = runTime*(totalRuns1 * nrLevels)

        totalTime = totalTime + totalTime1

        sampPeriod = 0.01
        atmPressure = 760.0
        
        # Run 1
        modelStructure = modelStructureAndParameters(gasExchange=False,control=True)
        modelStructure['ventilatorParams']['dcComponent'] = 8.0 #3.67
        modelStructure['ventilatorParams']['amplitude'] = 7.0 #8.8
        modelStructure['capacitorsParamsMatrix'][13][2] = modelStructure['capacitorsParamsMatrix'][13][3]
        modelStructure['capacitorsParamsMatrix'][13][0] = 100.0 + 50 # Compliance Thorax
        modelStructure['capacitorsParamsMatrix'][2][0] =  300.0 - 50  # Compliance systemic veins


        #                                         TargetVar ToCtrllVar TargetVal MinVal MaxVal     Kp    Kd  type
        modelStructure['controllers'] = np.array(['V_La',   'C_La',    3200.0,   1000.0, 000.0,  0.005, 0.007,  2,  # C_La
                                                  'V_Hr',   'e_C_Hr',  180.00,   0.0100, 0.400,  0.050, 3.000,  2,  # e_C_Hr
                                                  'V_Hl',   'e_C_Hl',  120.00,   0.0100, 0.100,  0.050, 3.000,  2,  # e_C_Hl
                            ]).reshape(3, 8)
        modelStructure['controllers'] = calculateDoubleSigmoid(0, modelStructure['controllers'])
        print(modelStructure['controllers'])
        '''
        modelStructure['controllers'] = np.array(['a_As',   'R_AsCs',   86.84,    0.20,  2.00,  0.01, 3.0,  0,  # As
                                                  'a_Ap',   'R_ApCp',   17.0,    0.01,  0.60,  0.01, 3.0,  0,  # Ap
                                                  'a_COl',  'E_C_Hl',   70.0,    0.01,  40.0,  0.02, 3.0,  0,  # E_Hl
                                                  'a_COr',  'E_C_Hr',   70.0,    0.01,  100.0,  0.02, 3.0,  0,  # E_Hr
                                                  'V_La',   'C_La',    3200.0,   1000.0, 000.0,  0.004, 0.004,  2,  # C_La
                                                  'V_Hr',   'e_C_Hr',  180.00,   0.0100, 0.400,  0.050, 3.000,  2,  # e_C_Hr
                                                  'V_Hl',   'e_C_Hl',  120.00,   0.0100, 0.100,  0.050, 3.000,  2,  # e_C_Hl
                                                  'a_Min',  'e_C_Thx', 2500.0,   2000, 5000, 0.01,  3.0,  0,  # E_Thx
                                                  'a_Max',  'E_C_Thx', 3000.0,   2000, 5000, 0.003,  3.0,  0,  # E_Thx
                            ]).reshape(9, 8)
        '''

        
        # PEEP Runs
        states, modelObjects, structures = modelGenerator.modelInit(modelStructure)
        pressureRes, runsRes, trun, states = runSimulation(states, modelObjects, runTime=runTime, totalRunsToIgnore=totalRunsToIgnore, totalRuns=totalRuns, pressureRes={}, runsRes={}, structures=structures)
        for i in range(0,nrLevels):
            modelStructure['ventilatorParams']['dcComponent'] = modelStructure['ventilatorParams']['dcComponent'] - 1
            #states['C_Vs'] = 225.0
            ignoredStates, modelObjects, structures = modelGenerator.modelInit(modelStructure)
            pressureRes, runsRes, trun, states = runSimulation(states, modelObjects, runTime=runTime, totalRunsToIgnore=totalRunsToIgnore1, totalRuns=totalRuns1, pressureRes=pressureRes, runsRes=runsRes, structures=structures)
            #pressureRes, runsRes, trun, states = runSimulation(states, modelObjects, runTime=runTime, totalRunsToIgnore=totalRunsToIgnore1, totalRuns=totalRuns1, pressureRes={}, runsRes={}, structures=structures)
            results = pressureRes | runsRes
    
    
    
    plots.plotDoubleSigmoidColorsRC(0, structures,results)


    


    #plotTriggers('HC',results, totalTime, sampPeriod, atmPressure)
    #plotTriggers('RC',results, totalTime, sampPeriod, atmPressure)

    #plots.plotSigmoid(4,results, totalTime, sampPeriod, atmPressure, structures)
    plots.plotSigmoid(1, structures,results)
    plots.plotSigmoid(2, structures,results)
    plots.plotCardioVolumes(results, totalTime, sampPeriod, atmPressure)
    plots.plotPressureVolumeHeart(results, totalTime, sampPeriod, atmPressure,modelObjects)
    plots.plotPressureVolumeAlveoli(results, totalTime, sampPeriod, atmPressure,modelObjects)
    plots.plotcardioFlows(results, totalTime, sampPeriod, atmPressure,modelObjects)

    #plots.plotCardioVolumes(results1, totalTime1, sampPeriod, atmPressure)
    #plots.plotPressureVolumeHeart(results1, totalTime1, sampPeriod, atmPressure,modelObjects)
    #plots.plotPressureVolumeAlveoli(results1, totalTime1, sampPeriod, atmPressure,modelObjects)
    plots.plotControllers(results, totalTime, sampPeriod, atmPressure)

    if structures['modelStructure']['modelSwitches']['gasExchange']:
        plots.plotPartialPressures(results, totalTime, sampPeriod, atmPressure,modelObjects)

    #plots.compareRunsCardioPV(results, results1, modelObjects, totalTime, sampPeriod, atmPressure, 'Self-Breathing Pressure-Volume Curves', 'Ventilator Pressure-Volume Curves')
    #plots.compareRunsLungPV(results, results1, modelObjects, totalTime, sampPeriod, atmPressure, 'Self-Breathing Pressure-Volume Curves', 'Ventilator Pressure-Volume Curves')
    return states,modelObjects,modelStructure

#██    ██ ███████ ███    ██ ████████ ██████  ██  ██████ ██      ███████      ██████  ██████  ███    ███ ██████  ██       █████  ██ ███    ██  ██████ ███████ 
#██    ██ ██      ████   ██    ██    ██   ██ ██ ██      ██      ██          ██      ██    ██ ████  ████ ██   ██ ██      ██   ██ ██ ████   ██ ██      ██      
#██    ██ █████   ██ ██  ██    ██    ██████  ██ ██      ██      █████       ██      ██    ██ ██ ████ ██ ██████  ██      ███████ ██ ██ ██  ██ ██      █████   
# ██  ██  ██      ██  ██ ██    ██    ██   ██ ██ ██      ██      ██          ██      ██    ██ ██  ██  ██ ██      ██      ██   ██ ██ ██  ██ ██ ██      ██      
#  ████   ███████ ██   ████    ██    ██   ██ ██  ██████ ███████ ███████      ██████  ██████  ██      ██ ██      ███████ ██   ██ ██ ██   ████  ██████ ███████

def runVentricleComplianceTest():
    with jax.default_device(jax.devices("cpu")[0]):
        jax.config.update("jax_enable_x64", True)

        runTime = 5
        totalRunsToIgnore = 100
        totalRuns = 1
        
        totalRunsToIgnore1 = 100
        totalRuns1 = 1

        totalTime = runTime*(totalRuns)
        totalTime1 = runTime*(totalRuns1)
        sampPeriod = 0.01
        
        # Run 1
        modelStructure = modelStructureAndParameters(gasExchange=False,control=True,dcComponent=0.0,amplitude=0.0)
        #                                         TargetVar ToCtrllVar TargetVal MinVal MaxVal  Kp    Kd  type
        modelStructure['controllers'] = np.array([ 'a_As',   'R_AsCs',   85.0,    0.20,  2.00,  0.01, 3.0,  0,  # As
                                                'a_Ap',   'R_ApCp',   17.0,    0.01,  0.60,  0.01, 3.0,  0,  # Ap
                                                'a_COl',  'E_C_Hl',   70.0,    0.02,  40.0,  0.01, 3.0,  0,  # E_Hl
                                                'a_COr',  'E_C_Hr',   70.0,    0.02,  20.0,  0.01, 3.0,  0,  # E_Hr
                                                'V_La',   'C_La',    3500.0,   1000.0, 100.0,  0.002, 3.0,  2,  # C_La
                                                'V_Hr',   'e_C_Hr',   70.0,   0.01,   0.2,  0.05, 3.0,  2,  # C_La
                                                'V_Hl',   'e_C_Hl',   70.0,   0.01,   0.1,  0.05, 3.0,  2,  # C_La
                                                'a_Min',  'e_C_Thx', 2500.0,   3000, 5000, 0.01,  3.0,  0,  # E_Thx
                                                'a_Max',  'E_C_Thx', 3000.0,   3000, 5000, 0.01,  3.0,  0,  # E_Thx
                            ]).reshape(9, 8)

        states, modelObjects, structures = modelGenerator.modelInit(modelStructure)
        pressureRes, runsRes, trun, states = runSimulation(states, modelObjects, runTime=runTime, totalRunsToIgnore=totalRunsToIgnore, totalRuns=totalRuns, pressureRes={}, runsRes={}, structures=structures)
        plots.plotSigmoid(5,pressureRes | runsRes, totalTime, sampPeriod, states['P_Atm'], structures)
        plots.plotSigmoid(6,pressureRes | runsRes, totalTime, sampPeriod, states['P_Atm'], structures) 
        plots.plotSigmoid(4,pressureRes | runsRes, totalTime, sampPeriod, states['P_Atm'], structures)
        # Run 2
        modelStructure = modelStructureAndParameters(gasExchange=False,control=True,dcComponent=0.0,amplitude=0.0)
        
        modelStructure['controllers'] = np.array([ 'a_As',   'R_AsCs',   85.0,    0.20,  2.00,  0.01, 3.0,  0,  # As
                                                'a_Ap',   'R_ApCp',   17.0,    0.01,  0.60,  0.01, 3.0,  0,  # Ap
                                                'a_COl',  'E_C_Hl',   70.0,    0.02,  40.0,  0.01, 3.0,  0,  # E_Hl
                                                'a_COr',  'E_C_Hr',   70.0,    0.02,  20.0,  0.01, 3.0,  0,  # E_Hr
                                                'V_La',   'C_La',    3500.0,   1000.0, 100.0,  0.002, 3.0,  2,  # C_La
                                                #'V_Hr',   'e_C_Hr',   100.0,   0.03,   0.3,  0.05, 3.0,  2,  # C_La
                                                #'V_Hl',   'e_C_Hl',   120.0,   0.03,   0.2,  0.05, 3.0,  2,  # C_La
                                                'a_Min',  'e_C_Thx', 2500.0,   3000, 5000, 0.01,  3.0,  0,  # E_Thx
                                                'a_Max',  'E_C_Thx', 3000.0,   3000, 5000, 0.01,  3.0,  0,  # E_Thx
                        ]).reshape(7, 8)
        
        states, modelObjects, structures = modelGenerator.modelInit(modelStructure)
        pressureRes1, runsRes1, trun, states = runSimulation(states, modelObjects, runTime=runTime, totalRunsToIgnore=totalRunsToIgnore1, totalRuns=totalRuns1, pressureRes={}, runsRes={}, structures=structures)
        
    
    results = pressureRes | runsRes
    results1 = pressureRes1 | runsRes1
    plots.compareRunsCardioPV(results, results1, modelObjects, totalTime1, sampPeriod, states['P_Atm'], 'Self-Breathing Pressure-Volume Curves', 'Ventilator Pressure-Volume Curves')

       

# █████  ██      ██    ██ ███████  ██████  ██      ██      ██████  ██████  ███    ███ ██████  ██       █████  ██ ███    ██  ██████ ███████ 
#██   ██ ██      ██    ██ ██      ██    ██ ██      ██     ██      ██    ██ ████  ████ ██   ██ ██      ██   ██ ██ ████   ██ ██      ██      
#███████ ██      ██    ██ █████   ██    ██ ██      ██     ██      ██    ██ ██ ████ ██ ██████  ██      ███████ ██ ██ ██  ██ ██      █████   
#██   ██ ██       ██  ██  ██      ██    ██ ██      ██     ██      ██    ██ ██  ██  ██ ██      ██      ██   ██ ██ ██  ██ ██ ██      ██      
#██   ██ ███████   ████   ███████  ██████  ███████ ██      ██████  ██████  ██      ██ ██      ███████ ██   ██ ██ ██   ████  ██████ ███████
def runAlveoliComplianceTest():

    with jax.default_device(jax.devices("cpu")[0]):
        jax.config.update("jax_enable_x64", True)

        runTime = 5
        totalRunsToIgnore = 100
        totalRuns = 0
        
        totalRunsToIgnore1 = 20
        totalRuns1 = 1

        totalTime = runTime*(totalRuns)
        totalTime1 = runTime*(totalRuns1)
        sampPeriod = 0.01
        
        # Run 1
        modelStructure = modelStructureAndParameters(gasExchange=False,control=True,dcComponent=0.0,amplitude=0.0)
        #                                         TargetVar ToCtrllVar TargetVal MinVal MaxVal  Kp    Kd  type
        modelStructure['controllers'] = np.array([ 'a_As',   'R_AsCs',   85.0,    0.20,  2.00,  0.01, 3.0,  0,  # As
                                                'a_Ap',   'R_ApCp',   17.0,    0.01,  0.60,  0.01, 3.0,  0,  # Ap
                                                'a_COl',  'E_C_Hl',   70.0,    0.02,  40.0,  0.01, 3.0,  0,  # E_Hl
                                                'a_COr',  'E_C_Hr',   70.0,    0.02,  20.0,  0.01, 3.0,  0,  # E_Hr
                                                'V_La',   'C_La',    3500.0,   1000.0, 100.0,  0.002, 3.0,  2,  # C_La
                                                'V_Hr',   'e_C_Hr',   100.0,   0.03,   0.3,  0.05, 3.0,  2,  # C_La
                                                'V_Hl',   'e_C_Hl',   120.0,   0.03,   0.2,  0.05, 3.0,  2,  # C_La
                                                'a_Min',  'e_C_Thx', 2500.0,   3000, 5000, 0.01,  3.0,  0,  # E_Thx
                                                'a_Max',  'E_C_Thx', 3000.0,   3000, 5000, 0.01,  3.0,  0,  # E_Thx
                            ]).reshape(9, 8)

        states, modelObjects, structures = modelGenerator.modelInit(modelStructure)
        pressureRes, runsRes, trun, states = runSimulation(states, modelObjects, runTime=runTime, totalRunsToIgnore=totalRunsToIgnore, totalRuns=totalRuns, pressureRes={}, runsRes={}, structures=structures)

        # Run 2
        modelStructure = modelStructureAndParameters(gasExchange=False,control=True,dcComponent=3.67,amplitude=8.8)
        
        modelStructure['controllers'] = np.array([ #'a_As',   'R_AsCs',   85.0,    0.20,  2.00,  0.01, 3.0,  0,  # As
                                                #'a_Ap',   'R_ApCp',   17.0,    0.01,  0.60,  0.01, 3.0,  0,  # Ap
                                                #'a_COl',  'E_C_Hl',   70.0,    0.02,  40.0,  0.01, 3.0,  0,  # E_Hl
                                                #'a_COr',  'E_C_Hr',   70.0,    0.02,  20.0,  0.01, 3.0,  0,  # E_Hr
                                                'V_La',   'C_La',    3500.0,   1000.0, 100.0,  0.002, 3.0,  2,  # C_La
                                                'V_Hr',   'e_C_Hr',   100.0,   0.03,   0.3,  0.05, 3.0,  2,  # C_La
                                                'V_Hl',   'e_C_Hl',   120.0,   0.03,   0.2,  0.05, 3.0,  2,  # C_La
                                                #'a_Min',  'e_C_Thx', 2500.0,   3000, 5000, 0.01,  3.0,  0,  # E_Thx
                                                #'a_Max',  'E_C_Thx', 3000.0,   3000, 5000, 0.01,  3.0,  0,  # E_Thx
                        ]).reshape(3, 8)
        
        states['E_C_Thx'] = states['e_C_Thx']
        states['C_Vs'] = 225.0
        ignoredStates, modelObjects, structures = modelGenerator.modelInit(modelStructure)
        pressureRes, runsRes, trun, states = runSimulation(states, modelObjects, runTime=runTime, totalRunsToIgnore=totalRunsToIgnore1, totalRuns=totalRuns1, pressureRes={}, runsRes={}, structures=structures)
        plots.plotSigmoid(0,pressureRes | runsRes, totalTime, sampPeriod, states['P_Atm'], structures)
        # Run 3
        modelStructure = modelStructureAndParameters(gasExchange=False,control=True,dcComponent=3.67,amplitude=8.8)
        
        modelStructure['controllers'] = np.array([#'a_As',   'R_AsCs',   85.0,    0.20,  2.00,  0.01, 3.0,  0,  # As
                                                #'a_Ap',   'R_ApCp',   17.0,    0.01,  0.60,  0.01, 3.0,  0,  # Ap
                                                #'a_COl',  'E_C_Hl',   70.0,    0.02,  40.0,  0.01, 3.0,  0,  # E_Hl
                                                #'a_COr',  'E_C_Hr',   70.0,    0.02,  20.0,  0.01, 3.0,  0,  # E_Hr
                                                #'V_La',   'C_La',    3500.0,   1000.0, 100.0,  0.002, 3.0,  2,  # C_La
                                                'V_Hr',   'e_C_Hr',   100.0,   0.03,   0.3,  0.05, 3.0,  2,  # C_La
                                                'V_Hl',   'e_C_Hl',   120.0,   0.03,   0.2,  0.05, 3.0,  2,  # C_La
                                                #'a_Min',  'e_C_Thx', 2500.0,   3000, 5000, 0.01,  3.0,  0,  # E_Thx
                                                #'a_Max',  'E_C_Thx', 3000.0,   3000, 5000, 0.01,  3.0,  0,  # E_Thx
                        ]).reshape(2, 8)
        
        ignoredStates, modelObjects, structures = modelGenerator.modelInit(modelStructure)
        pressureRes1, runsRes1, trun, states = runSimulation(states, modelObjects, runTime=runTime, totalRunsToIgnore=totalRunsToIgnore1, totalRuns=totalRuns1, pressureRes={}, runsRes={}, structures=structures)


    results = pressureRes | runsRes
    results1 = pressureRes1 | runsRes1

    plots.compareRunsLungPV(results, results1, modelObjects, totalTime1, sampPeriod, states['P_Atm'], 'Self-Breathing Pressure-Volume Curves', 'Ventilator Pressure-Volume Curves')




'''
███    ███  █████  ██ ███    ██ 
████  ████ ██   ██ ██ ████   ██ 
██ ████ ██ ███████ ██ ██ ██  ██ 
██  ██  ██ ██   ██ ██ ██  ██ ██ 
██      ██ ██   ██ ██ ██   ████
'''

t0 = time.time()
states,modelObjects,modelStructure = runSelfBreathingVentilatorTest()
#states,modelObjects,modelStructure = runPEEPVentilatorTest()
#runVentricleComplianceTest()
#runAlveoliComplianceTest()


# Example on how to read the data from the hdf5 file1 ###########################
#HDF5API.hdf5Example()                                                          #
#print(str(np.round(time.time() - t0, 4)) + ' -> HDF5 read and plot done!')     #
#t0 = time.time()                                                               #
#################################################################################

