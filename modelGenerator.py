import numpy as np
import equations as eq

#██    ██ ███    ██ ██ ████████      ██████  ██████  ███    ██ ██    ██ ███████ ██████  ███████ ██  ██████  ███    ██ ███████ 
#██    ██ ████   ██ ██    ██        ██      ██    ██ ████   ██ ██    ██ ██      ██   ██ ██      ██ ██    ██ ████   ██ ██      
#██    ██ ██ ██  ██ ██    ██        ██      ██    ██ ██ ██  ██ ██    ██ █████   ██████  ███████ ██ ██    ██ ██ ██  ██ ███████ 
#██    ██ ██  ██ ██ ██    ██        ██      ██    ██ ██  ██ ██  ██  ██  ██      ██   ██      ██ ██ ██    ██ ██  ██ ██      ██ 
# ██████  ██   ████ ██    ██         ██████  ██████  ██   ████   ████   ███████ ██   ██ ███████ ██  ██████  ██   ████ ███████
                                                                      
def pa2mmHg (value):
    return value * 0.0075006156130264

def mmHg2Pa (value):
    return value * 133.3223900000007

#███    ███  ██████  ██████  ███████ ██           ██████  ██████  ███    ██ ███████     ███████ ████████ ██    ██  ██████ ████████ 
#████  ████ ██    ██ ██   ██ ██      ██          ██      ██    ██ ████   ██ ██          ██         ██    ██    ██ ██         ██    
#██ ████ ██ ██    ██ ██   ██ █████   ██          ██      ██    ██ ██ ██  ██ █████       ███████    ██    ██    ██ ██         ██    
#██  ██  ██ ██    ██ ██   ██ ██      ██          ██      ██    ██ ██  ██ ██ ██               ██    ██    ██    ██ ██         ██    
#██      ██  ██████  ██████  ███████ ███████      ██████  ██████  ██   ████ ██          ███████    ██     ██████   ██████    ██    

def modelStructureAndParametersDefault(gasExchange=True,control=True,dcComponent=0.0,amplitude=0.0,E_minThx=0.005):

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
        'airViscosity': pa2mmHg(1.81e-5),  # kg/(m*s)  Pa*s (converted to mmHg)
        'bloodViscosity': pa2mmHg(3.5e-3), # kg/(m*s)  Pa*s (converted to mmHg)
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
    compartmentNames = ['As', 'Cs', 'Vs', 'Vt', 'Hr', 'Ap', 'Cp', 'Vp', 'Hl', 'Lt', 'Lb', 'La', 'Atm', 'Thx', 'Plr', 'Tis', 'Ven']

    ############################################### Connectivity Matrix  ####################################################
    #                               As Cs Vs Vt Hr Ap Cp Vp Hl Lt Lb La Atm Thx Plr Tis Ven col/row
    connectivityMatrix = np.array([ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # As
                                    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Cs
                                    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Vs
                                    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Vt
                                    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Hr
                                    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Ap
                                    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Cp
                                    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,  0,  0,  0,  0,  0,  # Vp
                                    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Hl
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,  0,  0,  0,  0,  0,  # Lt
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,  0,  0,  0,  0,  0,  # Lb
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # La
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Atm
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Thx
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Plr
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Tis
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,  0,  0,  0,  0,  0,  # Ven
                                        ]).reshape(17, 17)


    ############################################### Resistor Parameters  ####################################################
    # Reisitors parameters (Names and number of resistors should be calculated by the connectivityMatrix) 
    resistorNames = ['AsCs', 'CsVs', 'VsVt', 'VtHr', 'HrAp', 'ApCp', 'CpVp', 'VpHl', 'HlAs', 'VenLt', 'LtLb', 'LbLa']

    #                                     R       L       y0  type
    resistorParamsMatrix = np.array([   0.539, 0.00000, 0.000, 1,  # AsCs
                                        0.080, 0.00000, 0.000, 1,  # CsVs
                                        0.060, 0.00000, 0.000, 1,  # VsVt
                                        0.017, 0.00001, 0.000, 0,  # VtHr
                                        0.003, 0.00010, 0.000, 0,  # HrAp
                                        0.106, 0.00000, 0.000, 1,  # ApCp
                                        0.010, 0.00000, 0.000, 1,  # CpVp
                                        0.010, 0.00001, 0.000, 0,  # VpHl
                                        0.006, 0.00010, 0.000, 0,  # HlAs
                                        0.005, 0.00100, 0.000, 1,  # VenLt
                                        0.007, 0.00000, 0.000, 1,  # LtLb
                                        0.003, 0.00000, 0.000, 1,  # LbLa
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
                                        0.001, 0.00000, 0.000, 1,  # VenLt
                                        0.002, 0.00000, 0.000, 1,  # LtLb
                                        0.001, 0.00000, 0.000, 1,  # LbLa
                                        ]).reshape(12, 4)
    '''


    ############################################### Capacitor Parameters  ####################################################
    #                                      C       V0     Emax    Emin   Trise    y0   type
    capacitorsParamsMatrix = np.array([ 1.0000, 650.00, 0.0000, 0.0000, 0.0000, 772.80, 2,  # As
                                        10.000, 0.0000, 0.0000, 0.0000, 0.0000, 279.00, 1,  # Cs
                                        125.00, 0.0000, 0.0000, 0.0000, 0.0000, 2369.0, 1,  # Vs
                                        13.000, 000.00, 0.0000, 0.0000, 0.0000, 237.23, 1,  # Vt
                                        0.0000, 0.0000, 1.4990, 0.1500, 0.0000, 30.290, 0,  # Hr ***
                                        2.0000, 090.00, 0.0000, 0.0000, 0.0000, 163.63, 2,  # Ap
                                        8.0000, 0.0000, 0.0000, 0.0000, 0.0000, 102.82, 1,  # Cp
                                        16.000, 00.000, 0.0000, 0.0000, 0.0000, 182.21, 1,  # Vp
                                        0.0000, 0.0000, 4.2280, 0.1000, 0.0000, 36.800, 0,  # Hl ***
                                        0.5000, 75.000, 0.0000, 0.0000, 0.0000, 75.210, 2,  # Lt ####
                                        1.0000, 75.000, 0.0000, 0.0000, 0.0000, 75.780, 2,  # Lb ####
                                        0125.0, 1000.0, 0.0000, 0.0000, 0.0000, 1325.7, 2,  # La ####
                                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 5,  # Atm
                                        000.00, 4500.0, 0.0060, E_minThx, 0.0000, 0.0000, 4,  # Thx 
                                        1000.0, 0100.0, 0.0000, 0.0000, 0.0000, 0.0000, 3,  # Plr
                                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 5,  # Tis
                                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 6,  # Ven

                                        ]).reshape(17, 7)
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
    connectivityPresBiasMatrix = np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # As
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Cs
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Vs
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Vt
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Hr
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Ap
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Cp
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Vp
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Hl
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Lt
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Lb
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # La
                                            1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0,  1,  1,  0,  1,  1,  # Atm
                                            0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0,  0,  0,  1,  0,  0,  # Thx
                                            0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1,  0,  0,  0,  0,  0,  # Plr
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Tis
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Ven
                                        ]).reshape(17, 17)


    ############################################### Region Volume Map  ####################################################
    #                                       As Cs Vs Vt Hr Ap Cp Vp Hl Lt Lb La Atm Thx Plr Tis Ven
    connectivityRegionVolMatrix = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # As
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Cs
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Vs
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Vt
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Hr
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Ap
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Cp
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Vp
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Hl
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Lt
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Lb
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # La
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Atm
                                            0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1,  0,  0,  0,  0,  0,  # Thx
                                            0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1,  0,  0,  0,  0,  0,  # Plr
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Tis
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Ven
                                        ]).reshape(17, 17)


    ############################################### Membrane Resisros Map Map  ############################################
    #                                           As Cs Vs Vt Hr Ap Cp Vp Hl Lt Lb La Atm Thx Plr Tis Ven
    connectivityMemResistorsMatrix = np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # As
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Cs
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Vs
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Vt
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Hr
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Ap
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Cp
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Vp
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Hl
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Lt
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Lb
                                                0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # La
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Atm
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Thx
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Plr
                                                0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Tis
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Ven
                                        ]).reshape(17, 17)

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
    gasDistributionMatrix = np.array([  0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,  1,  0,  0,  0,  1,  # Atmosphere
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,  0,  0,  0,  0,  0,  # Alveoli
                                        1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,  0,  0,  0,  0,  0,  # Arterial
                                        0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Venous
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  1,  0,  # Tissues
                                    ]).reshape(5, 17)
    
    ############################################### Controllers  ##########################################################
    #                      period  varName varInt Type 
    integrators = np.array(['HC', 'i_Hl',  'Q_HlAs',  0,  # i_Hl
                            'HC', 'i_Hr',  'Q_HrAp',  0,  # i_Hr
                            'HC', 'i_COl', 'i_Hl',  1,  # i_COl
                            'HC', 'i_COr', 'i_Hr',  1,  # i_COr
                    ]).reshape(4, 4)
    
    #                 AverageVar  ofVar    AtmPress Period type
    averages = np.array(['a_As', 'P_As',   'P_Atm',  3.0,   0,  # As
                         'a_Ap', 'P_Ap',   'P_Atm',  3.0,   0,  # Ap
                         'a_COl', 'i_COl', 92.0,  1.0,   1,  # COl
                         'a_COr', 'i_COr', 92.0,  1.0,   1,  # COr
                         'a_VLa', 'V_La' , 1500.0,  10.0,   1,  # COr
                    ]).reshape(5, 5)
    
    #                       TargetVar ToCtrllVar TargetVal MinVal MaxVal  Kp    Kd  type
    controllers = np.array([ 'a_As',   'R_AsCs',   90.0,    0.20,  2.00,  0.01, 3.0,  0,  # As
                             'a_Ap',   'R_ApCp',   20.0,    0.01,  0.60,  0.01, 3.0,  0,  # Ap
                             'a_COl',  'E_C_Hl',   92.0,    0.02,  40.0,  0.01, 3.0,  1,  # E_Hl
                             'a_COr',  'E_C_Hr',   92.0,    0.02,  20.0,  0.01, 3.0,  1,  # E_Hr
                             'V_La',   'C_La',    2000.0,   500.0,  55.0, 0.006,  3.0,  2,  # C_La
                             #'a_VLa',   'C_La',    1500.0,   75.0,  250.0, 0.004,  3.0,  2,  # C_La
                        ]).reshape(5, 8)

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




#################################################################################################################################################
'''##############################################################################################################################################
                                                                                                                                               ##
███    ███  ██████  ██████  ███████ ██          ██ ███    ██ ██ ████████ ██  █████  ██      ██ ███████  █████  ████████ ██  ██████  ███    ██  ## 
████  ████ ██    ██ ██   ██ ██      ██          ██ ████   ██ ██    ██    ██ ██   ██ ██      ██    ███  ██   ██    ██    ██ ██    ██ ████   ██  ## 
██ ████ ██ ██    ██ ██   ██ █████   ██          ██ ██ ██  ██ ██    ██    ██ ███████ ██      ██   ███   ███████    ██    ██ ██    ██ ██ ██  ██  ## 
██  ██  ██ ██    ██ ██   ██ ██      ██          ██ ██  ██ ██ ██    ██    ██ ██   ██ ██      ██  ███    ██   ██    ██    ██ ██    ██ ██  ██ ██  ## 
██      ██  ██████  ██████  ███████ ███████     ██ ██   ████ ██    ██    ██ ██   ██ ███████ ██ ███████ ██   ██    ██    ██  ██████  ██   ████  ## 
                                                                                                                                               ##
##############################################################################################################################################
'''#################################################################################################################################################



def modelInit(modelStructure):

    #modelStructure = modelStructureAndParameters(gasExchange,control,dcComponent,amplitude,E_minThx)
    
    resistors = initResistors(modelStructure)

    capacitors = initCapacitors(modelStructure)

    modelCompartments = initModelCompartments(modelStructure, resistors, capacitors)

    memResistors  = initGasExchange(modelCompartments, resistors, modelStructure)

    states, modelObjects, other, statesDict = initModelObjects(resistors, capacitors, modelCompartments, memResistors, modelStructure)

    structures = {
        'resistors': resistors,
        'capacitors': capacitors,
        'modelCompartments': modelCompartments,
        'memResistors': memResistors,
        'other': other,
        'statesDict': statesDict,
        'modelStructure': modelStructure,
        
    }
    
    newStates = {}
    for key,value in states.items():
        #print(key,value)
        newStates[key] = float(value)
    
    return newStates, modelObjects, structures




#██ ███    ██ ██ ████████     ███    ███ ███████ ████████ ██   ██  ██████  ██████  ███████ 
#██ ████   ██ ██    ██        ████  ████ ██         ██    ██   ██ ██    ██ ██   ██ ██      
#██ ██ ██  ██ ██    ██        ██ ████ ██ █████      ██    ███████ ██    ██ ██   ██ ███████ 
#██ ██  ██ ██ ██    ██        ██  ██  ██ ██         ██    ██   ██ ██    ██ ██   ██      ██ 
#██ ██   ████ ██    ██        ██      ██ ███████    ██    ██   ██  ██████  ██████  ███████

# Looks into the connectivity matrix, then creates and parametrises the resistors
def initResistors(modelStructure):
    compartmentNames =  modelStructure['compartmentNames']
    connectivityMatrix = modelStructure['connectivityMatrix']
    resistorParamsMatrix = modelStructure['resistorParamsMatrix']
    resistorNames = modelStructure['resistorNames']
    prefixes = modelStructure['prefixes']

    resistors = {}
    for row in range(len(compartmentNames)):
        for col in range(len(compartmentNames)):
            value = connectivityMatrix[row][col]
            if value == 1:
                for resRow in range(len(resistorNames)):
                    if resistorNames[resRow] == compartmentNames[row]+compartmentNames[col]:
                        resistor = {
                            'id': prefixes['resistor'] + compartmentNames[row]+compartmentNames[col],
                            'flow': prefixes['flow'] + compartmentNames[row]+compartmentNames[col],
                            'pressureIn': prefixes['pressure'] + compartmentNames[row],
                            'pressureOut': prefixes['pressure'] + compartmentNames[col],
                            'typeR': resistorParamsMatrix[resRow][3],
                            'R': resistorParamsMatrix[resRow][0],
                            'L': resistorParamsMatrix[resRow][1],
                            'y0': resistorParamsMatrix[resRow][2],
                        }
                        resistors[compartmentNames[row]+compartmentNames[col]] = resistor
    
    return resistors

# Creates and parametrises the capacitors
def initCapacitors(modelStructure):
    compartmentNames = modelStructure['compartmentNames'] 
    connectivityPresBiasMatrix = modelStructure['connectivityPresBiasMatrix']
    capacitorsParamsMatrix = modelStructure['capacitorsParamsMatrix']
    connectivityRegionVolMatrix = modelStructure['connectivityRegionVolMatrix']
    prefixes = modelStructure['prefixes']

    capacitors = {}
    for col in range(len(compartmentNames)):
        regions = []
        for row in range(len(compartmentNames)):
            value = connectivityRegionVolMatrix[col][row]
            if value == 1:
                regions.append(prefixes['volume'] + compartmentNames[row])

        for row in range(len(compartmentNames)):
            value = connectivityPresBiasMatrix[row][col]
            if value == 1:
                capacitor = {
                    'id': prefixes['capacitor'] + compartmentNames[col],
                    'pressure': prefixes['pressure'] + compartmentNames[col],
                    'volume': prefixes['volume'] + compartmentNames[col],
                    'typeC': capacitorsParamsMatrix[col][6],
                    'C': capacitorsParamsMatrix[col][0],
                    'biasPressure': prefixes['pressure'] + compartmentNames[row],
                    'unstressedvolume': capacitorsParamsMatrix[col][1],
                    'Emax': capacitorsParamsMatrix[col][2],
                    'Emin': capacitorsParamsMatrix[col][3],
                    'Trise': capacitorsParamsMatrix[col][4],
                    'y0': capacitorsParamsMatrix[col][5],
                    'regions': regions,
                }
        capacitors[compartmentNames[col]] = capacitor
    return capacitors

# Creates the model compartments, and adds the resistors and capacitors to them
def initModelCompartments(modelStructure, resistors, capacitors):
    compartmentNames = modelStructure['compartmentNames'] 
    connectivityMatrix = modelStructure['connectivityMatrix']
    prefixes = modelStructure['prefixes']

    modelCompartments = []
    for col in range(len(compartmentNames)):
        compartment = {
            'id': compartmentNames[col],
            'level': 0,
            'isLeaf': False,
            'flowsIn': [],
            'flowsOut': [],
            'pressure': prefixes['pressure'] + compartmentNames[col],
            'volume': prefixes['volume'] + compartmentNames[col],
            'capacitor': capacitors[compartmentNames[col]],
            'resistorsIn': [],
            'resistorsOut': [],
            'fluidType' : 0,
            'partialPressures' : [],
            'partialPressuresY0' : [],
            'gasExchangeResistors' : [],
            'gasExchange' : {},
        }
        for row in range(len(compartmentNames)):
            value = connectivityMatrix[row][col]
            if value == 1:
                compartment['flowsIn'].append(prefixes['flow'] + compartmentNames[row] + compartmentNames[col])
                compartment['resistorsIn'].append(resistors[compartmentNames[row] + compartmentNames[col]])

            outValue = connectivityMatrix[col][row]
            if outValue == 1:
                compartment['flowsOut'].append(prefixes['flow'] + compartmentNames[col] + compartmentNames[row])
                compartment['resistorsOut'].append(resistors[compartmentNames[col] + compartmentNames[row]])    

        modelCompartments.append(compartment)
    return modelCompartments


def initGasExchange(modelCompartments, resistors, modelStructure):
    memResistors = {}

    resistorMemParamsMatrix = modelStructure['resistorMemParamsMatrix']
    connectivityMatrix = modelStructure['connectivityMatrix']
    connectivityMemResistorsMatrix = modelStructure['connectivityMemResistorsMatrix'] 
    compartmentNames = modelStructure['compartmentNames']
    memResistorNames = modelStructure['memResistorNames']
    gases = modelStructure['gases']
    gasRegions = modelStructure['gasRegions']
    gasPartialPressures = modelStructure['gasPartialPressures']
    gasDistributionMatrix = modelStructure['gasDistributionMatrix']
    prefixes = modelStructure['prefixes']
    
    # Adds all the partial pressure identifiers to the compartment, and the initial values
    # Creates the membrane resistors and if the compartment is a gas compartment, adds the resistor to the gasExchangeResistors list
    for col in range(len(compartmentNames)):
        for regionRow in range(len(gasRegions)):
            gasRegion = gasDistributionMatrix[regionRow][col]
            if gasRegion == 1:
                for speciesCol in range(len(gases)):
                    gasValue = gasPartialPressures[regionRow][speciesCol]
                    if gasValue > 0.0:
                        name = prefixes['pressure'] + gases[speciesCol] + '_' + compartmentNames[col]
                        modelCompartments[col]['partialPressures'].append(name)
                        modelCompartments[col]['partialPressuresY0'].append(gasValue)
                        modelCompartments[col]['fluidType'] = gasPartialPressures[regionRow][4]
        
                        # Looks for the resistor with the right name gets the resistor parameters
                        if modelStructure['modelSwitches']['gasExchange']:
                            for row in range(len(compartmentNames)):
                                    isResistorOut = connectivityMemResistorsMatrix[row][col]
                                    isResistorIn = connectivityMemResistorsMatrix[col][row]
                                    if isResistorIn == 1:
                                        for resRow in range(len(memResistorNames)):
                                            if gases[speciesCol] + '_' + compartmentNames[row]+compartmentNames[col] in memResistorNames[resRow]:
                                                resistor = {
                                                    'id': prefixes['resistor'] + gases[speciesCol] + '_' + compartmentNames[row]+compartmentNames[col],
                                                    'flow': prefixes['flow'] + gases[speciesCol] + '_' + compartmentNames[row]+compartmentNames[col],
                                                    'pressureIn': prefixes['pressure'] + gases[speciesCol] + '_' + compartmentNames[row],
                                                    'pressureOut': prefixes['pressure'] +  gases[speciesCol] + '_' + compartmentNames[col],
                                                    'typeR': resistorMemParamsMatrix[resRow][3],
                                                    'R': resistorMemParamsMatrix[resRow][0],
                                                    'L': resistorMemParamsMatrix[resRow][1],
                                                    'y0': resistorMemParamsMatrix[resRow][2],
                                                }
                                                memResistors[gases[speciesCol] + '_' + compartmentNames[row]+compartmentNames[col]] = resistor
                                                if modelCompartments[col]['fluidType'] == 1:
                                                    modelCompartments[col]['resistorsIn'].append(resistor) 
                                                    modelCompartments[col]['flowsIn'].append(resistor['flow']) 
                                                if modelCompartments[row]['fluidType'] == 1:
                                                    modelCompartments[row]['resistorsOut'].append(resistor) 
                                                    modelCompartments[row]['flowsOut'].append(resistor['flow']) 


    # Generates the map needed for the gas exchange calculation

    # Loop through every compartment
    for row in range(len(compartmentNames)):
        # loop through every gas species on that compartment
        for idx, species in enumerate(modelCompartments[row]['partialPressures']):
            # Create the skeleton for the gas exchange equation map
            
            modelCompartments[row]['gasExchange'][species] = {
                'in': {
                    'flows': [],
                    'positive': [],
                    'negative': []
                },
                'out': {
                    'flows': [],
                    'positive': [],
                    'negative': []
                },
            }
            speciesStructure = modelCompartments[row]['gasExchange'][species]

            # look into the connectivity matrix for resistors in and out
            for col in range(len(compartmentNames)):
                isResistorOut = connectivityMatrix[row][col]
                isResistorIn = connectivityMatrix[col][row]
                if isResistorIn == 1:
                    speciesStructure['in']['flows'].append(resistors[compartmentNames[col] + compartmentNames[row]]['flow'])
                    parentPartPressureName = modelCompartments[col]['partialPressures'][idx]
                    
                    speciesStructure['in']['positive'].append(parentPartPressureName)
                    speciesStructure['in']['negative'].append(species)
                    
                if isResistorOut == 1:
                    speciesStructure['out']['flows'].append(resistors[compartmentNames[row] + compartmentNames[col]]['flow'])
                    childPartPressureName = modelCompartments[col]['partialPressures'][idx]

                    speciesStructure['out']['positive'].append(species)
                    speciesStructure['out']['negative'].append(childPartPressureName)  


            # look into the connectivity matrix for resistors in and out
            for resistor in memResistors.values():
                if resistor['pressureOut'] in species:
                    speciesStructure['in']['flows'].append(resistor['flow'])
                    parentPartPressureName = resistor['pressureIn']
                
                    speciesStructure['in']['positive'].append(parentPartPressureName)
                    speciesStructure['in']['negative'].append(species)
                if resistor['pressureIn'] in species:
                    speciesStructure['out']['flows'].append(resistor['flow'])
                    childPartPressureName = resistor['pressureOut']

                    speciesStructure['out']['positive'].append(species)
                    speciesStructure['out']['negative'].append(childPartPressureName)


    return memResistors            


#██ ███    ██ ██ ████████     ███████  ██████  ██    ██  █████  ████████ ██  ██████  ███    ██      ██████  ██████       ██ 
#██ ████   ██ ██    ██        ██      ██    ██ ██    ██ ██   ██    ██    ██ ██    ██ ████   ██     ██    ██ ██   ██      ██ 
#██ ██ ██  ██ ██    ██        █████   ██    ██ ██    ██ ███████    ██    ██ ██    ██ ██ ██  ██     ██    ██ ██████       ██ 
#██ ██  ██ ██ ██    ██        ██      ██ ▄▄ ██ ██    ██ ██   ██    ██    ██ ██    ██ ██  ██ ██     ██    ██ ██   ██ ██   ██ 
#██ ██   ████ ██    ██        ███████  ██████   ██████  ██   ██    ██    ██  ██████  ██   ████      ██████  ██████   █████  

def initModelObjects(resistors, capacitors, modelCompartments, memResistors, modelStructure):

    resistorObjects, inductorObjects, flows, dQ, dR = initResisotorObjects(resistors)

    membraneResistorObjects, membraneFlows, dMQ, dMR = initMembraneResisotorObjects(memResistors)

    capacitorObjects, regionObjects, connectionObjects, dVdt, dC, dE, dCycles, statePressures, pressures   = initCapacitorObjects(modelCompartments, modelStructure)

    gasExchangeObjects, dPp = initGasExchangeObjects(modelCompartments, modelStructure)

    timekeepingObjects, cycleObjects, dT = initTimekeepingObjects(dCycles, modelStructure)

    resistorVariationObjects = initResistorVariationObjects(resistors, memResistors, dR, modelStructure)

    capacityVariationObjects = initCapacitorVariationObjects(capacitors, dC, modelStructure)

    elastanceVariationObjects = initElastanceVariationObjects(dE, modelStructure)

    integratorObjects, dI = initIntegratorObjects(modelStructure)

    averageObjects, dA = initAverageObjects(modelStructure)


    if modelStructure['modelSwitches']['gasExchange']:
        modelObjects = {
            'capacitors': capacitorObjects,                          # Calculates the pressures (P) for the lungs and heart
            'regions': regionObjects,                                # Calculates the pressures (P) for the regions ie Pleura, Thorax, Atmosphere, Tissues
            'inductors': inductorObjects,                            # Calculates the flows (Q) through the inertial resistors
            'resistors': resistorObjects,                            # Calculates the flows (Q) through the resistors
            'membraneResistors': membraneResistorObjects,            # Calculates the gas flows through membranes (Q)
            'partialPressures': gasExchangeObjects,                  # Calculates the partial pressures (P) of the gases in every compartment
            'connections': connectionObjects,                        # Calculates the volume variations (dVdt) in the compartments
            'cycles': cycleObjects,                                  # Calculates the cycle durations 
            'timekeeping': timekeepingObjects,                       # Keeps track of the time
            'resistanceVariations': resistorVariationObjects,        # Houses the resistance variations (dR) in the compartments
            'capacityVariations': capacityVariationObjects,          # Houses the capacitance variations (dC) in the compartments
            'elastanceVariations' : elastanceVariationObjects,       # Houses the elastance variations (dE) in the compartments
            'integrators': integratorObjects,
            'averages': averageObjects,
        }

        statesDict = {
            'dVdt': dVdt,                     # Houses the volume variations (dVdt) in the compartments
            'dC': dC,                         # Houses the capacitance variations (dC) in the compartments
            'dE': dE,                         # Houses the elastance variations (dE) in the compartments
            'dQ': dQ | dMQ,                   # Houses the flow variations (dQ) in the compartments
            'dR': dR | dMR,                   # Houses the resistance variations (dR) in the compartments
            'dPp': dPp,                       # Houses the partial pressure variations (dPp) in the compartments
            'statePressures': statePressures, # Houses the pressures (P) in the compartments that are state variables TODO: check if this is needed
            'dCycles': dCycles,               # Houses the cycle duration variations (dCycles) in the compartments
            'time': dT,                       # Houses the time variations (dT) in the compartments
            'integrators': dI,
            'averages': dA,
        }

        other = {
            'flows': flows,
            'pressures': pressures,
            'membraneFlows': membraneFlows,
        }

        states = dVdt | dC | dE | dQ | dMQ | dR | dMR | dPp | statePressures | dCycles | dT | dA | dI
        
    else:
        modelObjects = {
            'capacitors': capacitorObjects,                          # Calculates the pressures (P) for the lungs and heart
            'regions': regionObjects,                                # Calculates the pressures (P) for the regions ie Pleura, Thorax, Atmosphere, Tissues
            'inductors': inductorObjects,                            # Calculates the flows (Q) through the inertial resistors
            'resistors': resistorObjects,                            # Calculates the flows (Q) through the resistors
            'membraneResistors': {},            # Calculates the gas flows through membranes (Q)
            'partialPressures': {},                  # Calculates the partial pressures (P) of the gases in every compartment
            'connections': connectionObjects,                        # Calculates the volume variations (dVdt) in the compartments
            'cycles': cycleObjects,                                  # Calculates the cycle durations 
            'timekeeping': timekeepingObjects,                       # Keeps track of the time
            'resistanceVariations': resistorVariationObjects,        # Houses the resistance variations (dR) in the compartments
            'capacityVariations': capacityVariationObjects,          # Houses the capacitance variations (dC) in the compartments
            'elastanceVariations' : elastanceVariationObjects,       # Houses the elastance variations (dE) in the compartments
            'integrators': integratorObjects,
            'averages': averageObjects,
        }

        statesDict = {
            'dVdt': dVdt,                     # Houses the volume variations (dVdt) in the compartments
            'dC': dC,                         # Houses the capacitance variations (dC) in the compartments
            'dE': dE,                         # Houses the elastance variations (dE) in the compartments
            'dQ': dQ | dMQ,                   # Houses the flow variations (dQ) in the compartments
            'dR': dR | dMR,                   # Houses the resistance variations (dR) in the compartments
            'dPp': dPp,                       # Houses the partial pressure variations (dPp) in the compartments
            'statePressures': statePressures, # Houses the pressures (P) in the compartments that are state variables TODO: check if this is needed
            'dCycles': dCycles,               # Houses the cycle duration variations (dCycles) in the compartments
            'time': dT,                       # Houses the time variations (dT) in the compartments
            'integrators': dI,
            'averages': dA,
        }

        other = {
            'flows': flows,
            'pressures': pressures,
            'membraneFlows': membraneFlows,
        }

    
        states = dVdt | dC | dE | dQ | dR | statePressures | dCycles | dT | dA | dI
    
    return states, modelObjects, other, statesDict


# Initialises the resistor Objects
def initResisotorObjects(resistors):
    flows = []
    dQ = {}
    dR = {}
    resistorObjects = {}
    inductorObjects ={}
    for resistor in resistors.values():
        if resistor['typeR'] == 0: #Diode
            resistorObjects[resistor['flow']] = eq.Diode(
                resistor['id'],
                resistor['L'],
                resistor['pressureIn'],
                resistor['pressureOut'],
                resistor['flow'],
                False
            )
            flows.append(resistor['flow'])
            dR[resistor['id']] = resistor['R']

        elif resistor['typeR'] == 1: #Resistor
            resistorObjects[resistor['flow']] = eq.Resistor(
                resistor['id'],
                resistor['L'],
                resistor['pressureIn'],
                resistor['pressureOut'],
                resistor['flow'],
                False
            )
            flows.append(resistor['flow'])
            dR[resistor['id']] = resistor['R']
            
        elif resistor['typeR'] == 2: #DiodeInertial
            inductorObjects[resistor['flow']] = eq.Diode(
                resistor['id'],
                resistor['L'],
                resistor['pressureIn'],
                resistor['pressureOut'],
                resistor['flow'],
                True
            )
            dQ[resistor['flow']] = resistor['y0']
            dR[resistor['id']] = resistor['R']

        elif resistor['typeR'] == 3: #ResistorInertial
            inductorObjects[resistor['flow']] = eq.Resistor(
                resistor['id'],
                resistor['L'],
                resistor['pressureIn'],
                resistor['pressureOut'],
                resistor['flow'],
                True
            )
            dQ[resistor['flow']] = resistor['y0']
            dR[resistor['id']] = resistor['R']

    return resistorObjects, inductorObjects, flows, dQ, dR

# Initialises the membrane resistor Objects
def initMembraneResisotorObjects(resistors):
    flows = []
    dQ = {}
    dR = {}
    resistorObjects = {}

    for resistor in resistors.values():
        if resistor['typeR'] == 0: #Diode
            resistorObjects[resistor['flow']] = eq.Diode(
                resistor['id'],
                resistor['L'],
                resistor['pressureIn'],
                [resistor['pressureOut']],
                resistor['flow'],
                False
            )
            flows.append(resistor['flow'])
            dR[resistor['id']] = resistor['R']

        elif resistor['typeR'] == 1: #Resistor
            resistorObjects[resistor['flow']] = eq.Resistor(
                resistor['id'],
                resistor['L'],
                resistor['pressureIn'],
                resistor['pressureOut'],
                resistor['flow'],
                False
            )
            flows.append(resistor['flow'])
            dR[resistor['id']] = resistor['R']
            
        elif resistor['typeR'] == 2: #DiodeInertial
            resistorObjects[resistor['flow']] = eq.Diode(
                resistor['id'],
                resistor['L'],
                resistor['pressureIn'],
                resistor['pressureOut'],
                resistor['flow'],
                True
            )
            flows.append(resistor['flow'])
            dQ[resistor['flow']] = resistor['y0']
            dR[resistor['id']] = resistor['R']

        elif resistor['typeR'] == 3: #ResistorInertial
            resistorObjects[resistor['flow']] = eq.Resistor(
                resistor['id'],
                resistor['L'],
                resistor['pressureIn'],
                resistor['pressureOut'],
                resistor['flow'],
                True
            )
            flows.append(resistor['flow'])
            dQ[resistor['flow']] = resistor['y0']
            dR[resistor['id']] = resistor['R']

    return resistorObjects, flows, dQ, dR

# Initialises the capacitor Objects
# TODO: Finda a solution for the 'HC and 'RC' parameters
# TODO: only capacitors 0,1,2 are considered for dVdt calculation -> find a better way to do this as this is quite hardcoded
def initCapacitorObjects(modelCompartments, modelStructure):

    ventilatorParams = modelStructure['ventilatorParams']
    gasDistributionMatrix = modelStructure['gasDistributionMatrix']
    compartmentNames = modelStructure['compartmentNames']
    gasRegions = modelStructure['gasRegions']
    gases = modelStructure['gases']
    gasPartialPressures = modelStructure['gasPartialPressures']

    cycles = modelStructure['cycles']
    cyclesDuration = modelStructure['cyclesDuration']
    cyclesDistributionMatrix = modelStructure['cyclesDistributionMatrix']

    simulationParameters = modelStructure['simulationParameters']


    dVdt = {}
    dE = {}
    dC = {}
    statePressures = {}
    dCycles = {}

    capacitorObjects = {}
    regionObjects = {}
    connectionObjects = {}

    pressures = []

    for compartment in modelCompartments:
        capacitor = compartment['capacitor']

        # builds the connections objects for dVdt calculation
        if capacitor['typeC'] in [0,1,2]:
            fIn = []
            for resIn in compartment['resistorsIn']:
                fIn.append(resIn['flow'])
            fOut = []
            for resOut in compartment['resistorsOut']:
                fOut.append(resOut['flow'])
                
            connectionObjects[compartment['volume']] = eq.Connections(
                fIn,
                fOut,
                compartment['volume'],

            )

        # Elastance Capacitor
        if capacitor['typeC'] == 0:

            for col in range(len(compartmentNames)):
                if compartmentNames[col] == compartment['id']:
                    for row in range(len(cycles)):
                        if cyclesDistributionMatrix[row][col] == 1:
                            cycleDuration = cyclesDuration[row]
                            break

            capacitorObjects[capacitor['pressure']] = eq.ElastanceCapacitor(
                capacitor['volume'],
                capacitor['biasPressure'],
                'E_' + capacitor['id'], #Emax
                'e_' + capacitor['id'], #Emin
                cycles[row], #HC 
            )
            dE['E_' + capacitor['id']] = capacitor['Emax']
            dE['e_' + capacitor['id']] = capacitor['Emin']
            dVdt[capacitor['volume']] = capacitor['y0']
            pressures.append(capacitor['pressure'])
            dCycles[cycles[row]] = cycleDuration
            
        # Passive Capacitor
        elif capacitor['typeC'] == 1:
            if 'Cp' in capacitor['id']:
                capacitorObjects[capacitor['pressure']] = eq.Capacitor(
                    capacitor['volume'],
                    capacitor['id'],
                    'P_La|0|0',
                )
            else:
                capacitorObjects[capacitor['pressure']] = eq.Capacitor(
                    capacitor['volume'],
                    capacitor['id'],
                    capacitor['biasPressure'],
                )
            dVdt[capacitor['volume']] = capacitor['y0']
            dC[capacitor['id']] = capacitor['C']
            pressures.append(capacitor['pressure'])

        # Semi-Compliant Capacitor
        elif capacitor['typeC'] == 2:
            capacitorObjects[capacitor['pressure']] = eq.CapacitorDCVolume(
                capacitor['volume'],
                capacitor['id'],
                capacitor['biasPressure'],
                capacitor['unstressedvolume'],
            )
            dVdt[capacitor['volume']] = capacitor['y0']
            dC[capacitor['id']] = capacitor['C']
            pressures.append(capacitor['pressure'])

        # Pleural Capacitor
        elif capacitor['typeC'] == 3:
            regionObjects[capacitor['pressure']] = eq.CapacitorPleura(
                capacitor['pressure'],
                capacitor['id'],
                capacitor['biasPressure'],
                capacitor['regions'],
                capacitor['pressure'],
                capacitor['unstressedvolume'],
                simulationParameters['dt'],
                )
            #statePressures[capacitor['pressure']] = capacitor['y0']
            statePressures[capacitor['pressure']] = 760.0
            dC[capacitor['id']] = capacitor['C']
        
        # Thorax Capacitor
        elif capacitor['typeC'] == 4:

            for col in range(len(compartmentNames)):
                if compartmentNames[col] == compartment['id']:
                    for cycleRow in range(len(cycles)):
                        if cyclesDistributionMatrix[cycleRow][col] == 1:
                            cycleDuration = cyclesDuration[cycleRow]
                            break

            dCycles[cycles[cycleRow]] = cycleDuration
            regionObjects[capacitor['pressure']] = eq.CapacitorSelfBreathingThorax(
                capacitor['id'],
                #capacitor['C'],
                'E_' + capacitor['id'], #Emax
                'e_' + capacitor['id'], #Emin
                capacitor['biasPressure'],
                capacitor['regions'],
                cycles[cycleRow],
                capacitor['pressure'],
                capacitor['unstressedvolume'],
                simulationParameters['dt'],
                )
            #statePressures[capacitor['pressure']] = capacitor['y0']
            statePressures[capacitor['pressure']] = 760.0

            dC[capacitor['id']] = capacitor['C']
            dE['E_' + capacitor['id']] = capacitor['Emax']
            dE['e_' + capacitor['id']] = capacitor['Emin']
            

        # Contant pressure Capacitor
        elif capacitor['typeC'] == 5:
            regionObjects[capacitor['pressure']] = eq.ConstantPressure(
                capacitor['volume'],
                capacitor['C'],
                capacitor['biasPressure'],
                )
            
            for col in range(len(compartmentNames)):
                if compartmentNames[col] == compartment['id']:
                    for row in range(len(gasRegions)):
                        if gasDistributionMatrix[row][col] == 1:
                            totalPressure = gasPartialPressures[row][len(gases)]
                            break

            statePressures[capacitor['pressure']] = totalPressure

        # Ventilator pressure Capacitor       
        elif capacitor['typeC'] == 6:

            for col in range(len(compartmentNames)):
                if compartmentNames[col] == compartment['id']:
                    for cycleRow in range(len(cycles)):
                        if cyclesDistributionMatrix[cycleRow][col] == 1:
                            cycleDuration = cyclesDuration[cycleRow]
                            break
            dCycles[cycles[cycleRow]] = cycleDuration

            for col in range(len(compartmentNames)):
                if compartmentNames[col] == compartment['id']:
                    for row in range(len(gasRegions)):
                        if gasDistributionMatrix[row][col] == 1:
                            totalPressure = gasPartialPressures[row][len(gases)]
                            break

            ie = ventilatorParams['I'] + ventilatorParams['E']
            ieRatio = ventilatorParams['I'] / ie
            slopeFreq = ventilatorParams['slopeFraction'] / ie
            regionObjects[capacitor['pressure']] = eq.VentilatorPressure(
                #totalPressure + ventilatorParams['dcComponent'] + ventilatorParams['amplitude'], #Emax
                ventilatorParams['amplitude'], #Emax
                #totalPressure + ventilatorParams['dcComponent'], #Emin
                'PEEP',
                cycles[cycleRow],
                ieRatio,
                slopeFreq,
                capacitor['pressure'],
                simulationParameters['dt'],
                '',
                '',   
            )
            statePressures['PEEP'] = totalPressure + ventilatorParams['dcComponent']
            statePressures[capacitor['pressure']] = totalPressure + ventilatorParams['dcComponent']

        # Ventilator pressure from file     
        elif capacitor['typeC'] == 7:

            for col in range(len(compartmentNames)):
                if compartmentNames[col] == compartment['id']:
                    for cycleRow in range(len(cycles)):
                        if cyclesDistributionMatrix[cycleRow][col] == 1:
                            cycleDuration = cyclesDuration[cycleRow]
                            break
            dCycles[cycles[cycleRow]] = cycleDuration

            for col in range(len(compartmentNames)):
                if compartmentNames[col] == compartment['id']:
                    for row in range(len(gasRegions)):
                        if gasDistributionMatrix[row][col] == 1:
                            totalPressure = gasPartialPressures[row][len(gases)]
                            break


            regionObjects[capacitor['pressure']] = eq.FilePressure(
                capacitor['pressure'],
                modelStructure['ventilatorPressure'],
                simulationParameters['dt']
            )
            statePressures['PEEP'] = modelStructure['ventilatorPressure'][0]
            statePressures[capacitor['pressure']] = modelStructure['ventilatorPressure'][0]
    


    return capacitorObjects, regionObjects, connectionObjects, dVdt, dC, dE, dCycles, statePressures, pressures       

    #                                      C       V0     Emax    Emin   Trise    y0   type
    #                                    100.00, 0000.0, 4000.0, 5000.0, 0.0000, 0.0000, 4,  # Thx


# Initialises the gas exchange Objects
# TODO: solve the issue for RC
def initGasExchangeObjects(modelCompartments, modelStructure):
    ventilatorParams = modelStructure['ventilatorParams']
    gasDistributionMatrix = modelStructure['gasDistributionMatrix']
    compartmentNames = modelStructure['compartmentNames']
    gasRegions = modelStructure['gasRegions']
    gases = modelStructure['gases']
    gasPartialPressures = modelStructure['gasPartialPressures']
    simulationParameters = modelStructure['simulationParameters']


    gasExchangeObjects = {}
    dPp = {}

    for compartment in modelCompartments:
        pps = {pp:y0 for pp, y0 in zip(compartment['partialPressures'], compartment['partialPressuresY0'])}
        dPp.update(pps)

        # creates the objects for gas exchange calculations
        for species, values in compartment['gasExchange'].items():
            # all other compartments
            if (compartment['capacitor']['typeC'] != 5) and (compartment['capacitor']['typeC'] != 6):
                dV = {
                    'volume': compartment['volume'],
                    'partialPressure': species
                    }

                ppObject = eq.GasTransport(species, values, dV)
                gasExchangeObjects[species] = ppObject
            
            # Compartments that do not change pressures
            elif compartment['capacitor']['typeC'] == 5:
                dV = {
                    'volume': compartment['volume'],
                    'partialPressure': species
                    }

                ppObject = eq.GasTransportTissue(species, dV)
                gasExchangeObjects[species] = ppObject
            
            # Compartments that change pressures (ventilator)
            elif compartment['capacitor']['typeC'] == 6:
                for col in range(len(compartmentNames)):
                    if compartmentNames[col] == compartment['id']:
                        for row in range(len(gasRegions)):
                            if gasDistributionMatrix[row][col] == 1:
                                totalPressure = gasPartialPressures[row][len(gases)]
                                for gas in range(len(gases)):
                                    if gases[gas] in species:
                                        partialPressure = gasPartialPressures[row][gas]
                                        break
                
                ie = ventilatorParams['I'] + ventilatorParams['E']
                ieRatio = ventilatorParams['I'] / ie
                slopeFreq = ventilatorParams['slopeFraction'] / ie

                multiplierMax = (ventilatorParams['amplitude'] + ventilatorParams['dcComponent'] + totalPressure) / totalPressure
                multiplierMin = (ventilatorParams['dcComponent'] + totalPressure) / totalPressure


                gasExchangeObjects[species] = eq.VentilatorPartialPressure(
                    partialPressure * multiplierMax, #Emax
                    partialPressure * multiplierMin, #Emin
                    'RC', #HC
                    ieRatio,
                    slopeFreq,
                    species,
                    simulationParameters['dt'],
                    '',
                    '',  
                )


    return gasExchangeObjects, dPp

#
def initIntegratorObjects(modelStructure):
    integratorConf = modelStructure['integrator']
    step = modelStructure['simulationParameters']['dt']

    dI = {}
    integratorObjects = {}
    for i in range(len(integratorConf)):
        if int(integratorConf[i,3]) == 0:
            integratorObjects[integratorConf[i,1]] = eq.CycleIntegrator(str(integratorConf[i,0]), str(integratorConf[i,1]), str(integratorConf[i,2]), step)
        elif int(integratorConf[i,3]) == 1:
            integratorObjects[integratorConf[i,1]] = eq.CycleKeeper(str(integratorConf[i,0]), str(integratorConf[i,1]), str(integratorConf[i,2]), step)
        elif int(integratorConf[i,3]) == 2:
            integratorObjects[integratorConf[i,1]] = eq.CycleMax(str(integratorConf[i,0]), str(integratorConf[i,1]), str(integratorConf[i,2]), step)
        elif int(integratorConf[i,3]) == 3:
            integratorObjects[integratorConf[i,1]] = eq.CycleMin(str(integratorConf[i,0]), str(integratorConf[i,1]), str(integratorConf[i,2]), step)
        
        dI[integratorConf[i,1]] = integratorConf[i,4]

    return integratorObjects, dI

# Initialises the average Objects
# TODO: Types 0 do not have a good way to initialise
def initAverageObjects(modelStructure):
    averageconf = modelStructure['averages']
    step = modelStructure['simulationParameters']['dt']

    dA = {}
    averageObjects = {}

    for i in range(len(averageconf)):
        if int(averageconf[i,4]) == 0: 
            averageObjects[averageconf[i,0]] = eq.PressureAverage(str(averageconf[i,0]), str(averageconf[i,1]), str(averageconf[i,2]), float(averageconf[i,3]), step)
            dA[averageconf[i,0]] = float(averageconf[i,3])
        elif int(averageconf[i,4]) == 1:
            averageObjects[averageconf[i,0]] = eq.PeriodAverage(str(averageconf[i,0]), str(averageconf[i,1]), float(averageconf[i,3]), step)
            dA[averageconf[i,0]] = float(averageconf[i,2])

    return averageObjects, dA


# Initialises the timer objects for the cycles
def initTimekeepingObjects(dCycles, modelStructure):

    dT = {
        'T': 0.0,
        'T0': 0.0,
    }
    timekeepingObjects = {}
    cycleObjects = {}

    for key,value in dCycles.items():
        tiggerName = 'trigger' + key
        timerName = 'timer' + key
        dT[tiggerName] = 0.0
        dT[timerName] = value
        timekeepingObjects[key] = eq.PeriodicTrigger(
            key, 
            tiggerName,
            timerName,
            )
        cycleObjects[key] = eq.Cycle(
            key,
            )
    '''
    # controlled cycle
    cycleObjects['HC'] = c.CycleControlled(
        cyclePeriodIdx = 'HC', 
        varTargetIdx = 'V_O2_Cs', 
        targetValue = 90.0,
        #targetValue = 'V_O2_Cs', 
        minValueToControl = 0.35, 
        maxValueToControl = 1.5, 
        proportionalConstant = 0.01, 
        derivativeConstant = 0.2
    )
    '''    

    return timekeepingObjects, cycleObjects, dT

def initResistorVariationObjects(resistors, memResistors, dR, modelStructure):
    resistorVariationObjects = {}

    resControllers = modelStructure['controllers']
    step = modelStructure['simulationParameters']['dt']

    for key,value in resistors.items():
        resistorVariationObjects[value['id']] = eq.NoController(
                                    value['id'],#varToControlIdx
                                )
        if modelStructure['modelSwitches']['control']:
            for i in range(len(resControllers)):
                if value['id'] == str(resControllers[i,1]) :
                    if resControllers[i,7] == '0':
                        resistorVariationObjects[value['id']] = eq.LocalController(
                            str(resControllers[i,0]),#varTargetIdx
                            str(resControllers[i,1]),#varToControlIdx
                            float(resControllers[i,2]),#targetValue
                            float(resControllers[i,3]),#minValueToControl
                            float(resControllers[i,4]),#maxValueToControl
                            float(resControllers[i,5]),#proportionalConstant
                            float(resControllers[i,6]),#derivativeConstant
                            )
                    elif resControllers[i,7] == '1':
                        resistorVariationObjects[value['id']] = eq.LocalControllerAnti(
                            str(resControllers[i,0]),#varTargetIdx
                            str(resControllers[i,1]),#varToControlIdx
                            float(resControllers[i,2]),#targetValue
                            float(resControllers[i,3]),#minValueToControl
                            float(resControllers[i,4]),#maxValueToControl
                            float(resControllers[i,5]),#proportionalConstant
                            float(resControllers[i,6]),#derivativeConstant
                                        )
                    elif resControllers[i,7] == '6':
                        resistorVariationObjects[value['id']] = eq.SigmoidControllerResistor(
                            str(resControllers[i,0]),#varTargetIdx
                            str(resControllers[i,1]),#varToControlIdx
                            float(resControllers[i,2]),#targetValue
                            float(resControllers[i,3]),#minValueToControl
                            float(resControllers[i,4]),#maxValueToControl
                            float(resControllers[i,5]),#proportionalConstant
                            float(resControllers[i,6]),#derivativeConstant
                                        )
                    elif resControllers[i,7] == '4':
                        resistorVariationObjects[value['id']] = eq.SigmoidControllerCompliance(
                            str(resControllers[i,0]),#varTargetIdx
                            str(resControllers[i,1]),#varToControlIdx
                            float(resControllers[i,2]),#targetValue
                            float(resControllers[i,3]),#minValueToControl
                            float(resControllers[i,4]),#maxValueToControl
                            float(resControllers[i,5]),#proportionalConstant
                            float(resControllers[i,6]),#derivativeConstant
                            step
                            )
                    elif resControllers[i,7] == '5':
                        resistorVariationObjects[value['id']] = eq.RampController(
                            str(resControllers[i,1]),#varToControlIdx
                            float(resControllers[i,2]),#targetValue
                            )
                    break
        #capacitorVariationObjects['PEEP'] = eq.RampController(
    #                            'PEEP',#varToControlIdx
    #                            1.0,#targetValue
    #                        )


    if modelStructure['modelSwitches']['gasExchange']:
        for key,value in memResistors.items():
            resistorVariationObjects[value['id']] = eq.NoController(
                                value['id'],#varToControlIdx
                            )
        

    return resistorVariationObjects

def initCapacitorVariationObjects(capacitors, dC, modelStructure):
    capacitorVariationObjects = {}
    resControllers = modelStructure['controllers']
    step = modelStructure['simulationParameters']['dt']

    for key,value in capacitors.items():
        if value['typeC'] in [1,2,3,4]:
            capacitorVariationObjects[value['id']] = eq.NoController(
                                value['id'],#varToControlIdx
                            )
        
        if modelStructure['modelSwitches']['control']:
            for i in range(len(resControllers)):
                if value['id'] == str(resControllers[i,1]) :
                    if resControllers[i,7] == '2':
                        capacitorVariationObjects[value['id']] = eq.SigmoidController(
                            str(resControllers[i,0]),#varTargetIdx
                            str(resControllers[i,1]),#varToControlIdx
                            float(resControllers[i,2]),#targetValue
                            float(resControllers[i,3]),#minValueToControl
                            float(resControllers[i,4]),#maxValueToControl
                            float(resControllers[i,5]),#proportionalConstant
                            float(resControllers[i,6]),#derivativeConstant
                            step
                            )
                    elif resControllers[i,7] == '3':
                        capacitorVariationObjects[value['id']] = eq.DoubleSigmoidController(
                            str(resControllers[i,0]),#varTargetIdx
                            str(resControllers[i,1]),#varToControlIdx
                            float(resControllers[i,2]),#inflectionPoint
                            float(resControllers[i,3]),#maxCompliance
                            float(resControllers[i,4]),# separation
                            float(resControllers[i,5]),#k
                            float(resControllers[i,6]),#cMin
                            step
                            )
                    elif resControllers[i,7] == '4':
                        capacitorVariationObjects[value['id']] = eq.SigmoidControllerCompliance(
                            str(resControllers[i,0]),#varTargetIdx
                            str(resControllers[i,1]),#varToControlIdx
                            float(resControllers[i,2]),#targetValue
                            float(resControllers[i,3]),#minValueToControl
                            float(resControllers[i,4]),#maxValueToControl
                            float(resControllers[i,5]),#proportionalConstant
                            float(resControllers[i,6]),#derivativeConstant
                            step
                            )
                    elif resControllers[i,7] == '0':
                        capacitorVariationObjects[value['id']] = eq.LocalController(
                            str(resControllers[i,0]),#varTargetIdx
                            str(resControllers[i,1]),#varToControlIdx
                            float(resControllers[i,2]),#targetValue
                            float(resControllers[i,3]),#minValueToControl
                            float(resControllers[i,4]),#maxValueToControl
                            float(resControllers[i,5]),#proportionalConstant
                            float(resControllers[i,6]),#derivativeConstant
                            )
                    elif resControllers[i,7] == '1':
                        capacitorVariationObjects[value['id']] = eq.LocalControllerAnti(
                            str(resControllers[i,0]),#varTargetIdx
                            str(resControllers[i,1]),#varToControlIdx
                            float(resControllers[i,2]),#targetValue
                            float(resControllers[i,3]),#minValueToControl
                            float(resControllers[i,4]),#maxValueToControl
                            float(resControllers[i,5]),#proportionalConstant
                            float(resControllers[i,6]),#derivativeConstant
                                        )
                    elif resControllers[i,7] == '5':
                        capacitorVariationObjects[value['id']] = eq.RampController(
                            str(resControllers[i,1]),#varToControlIdx
                            float(resControllers[i,2]),#targetValue
                            )
                    break  
    capacitorVariationObjects['PEEP'] = eq.NoController(
                                'PEEP',#varToControlIdx
                            )
    
    #capacitorVariationObjects['PEEP'] = eq.RampController(
    #                            'PEEP',#varToControlIdx
    #                            1.0,#targetValue
    #                        )
    return capacitorVariationObjects

def initElastanceVariationObjects(dE, modelStructure):
    elastanceVariationObjects = {}

    resControllers = modelStructure['controllers']

    for key,value in dE.items():
        elastanceVariationObjects[key] = eq.NoController(
                            key,#varToControlIdx
                        )
        if modelStructure['modelSwitches']['control']:
            for i in range(len(resControllers)):
                if key == str(resControllers[i,1]) :
                    if resControllers[i,7] == '0':
                        elastanceVariationObjects[key] = eq.LocalController(
                            str(resControllers[i,0]),#varTargetIdx
                            str(resControllers[i,1]),#varToControlIdx
                            float(resControllers[i,2]),#targetValue
                            float(resControllers[i,3]),#minValueToControl
                            float(resControllers[i,4]),#maxValueToControl
                            float(resControllers[i,5]),#proportionalConstant
                            float(resControllers[i,6]),#derivativeConstant
                            )
                    elif resControllers[i,7] == '1':
                        elastanceVariationObjects[key] = eq.LocalControllerAnti(
                            str(resControllers[i,0]),#varTargetIdx
                            str(resControllers[i,1]),#varToControlIdx
                            float(resControllers[i,2]),#targetValue
                            float(resControllers[i,3]),#minValueToControl
                            float(resControllers[i,4]),#maxValueToControl
                            float(resControllers[i,5]),#proportionalConstant
                            float(resControllers[i,6]),#derivativeConstant
                            )
                    elif resControllers[i,7] == '2':
                        elastanceVariationObjects[key] = eq.SigmoidController(
                            str(resControllers[i,0]),#varTargetIdx
                            str(resControllers[i,1]),#varToControlIdx
                            float(resControllers[i,2]),#targetValue
                            float(resControllers[i,3]),#minValueToControl
                            float(resControllers[i,4]),#maxValueToControl
                            float(resControllers[i,5]),#proportionalConstant
                            float(resControllers[i,6]),#derivativeConstant
                            )
                    elif resControllers[i,7] == '3':
                        elastanceVariationObjects[key] = eq.DoubleSigmoidController(
                            str(resControllers[i,0]),#varTargetIdx
                            str(resControllers[i,1]),#varToControlIdx
                            float(resControllers[i,2]),#targetValue
                            float(resControllers[i,3]),#minValueToControl
                            float(resControllers[i,4]),#maxValueToControl
                            float(resControllers[i,5]),#proportionalConstant
                            float(resControllers[i,6]),#derivativeConstant
                            )
                    break
        
    return elastanceVariationObjects


