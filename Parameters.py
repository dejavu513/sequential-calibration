import modelLibrary as mL
import numpy as np

# Contains the parameters for the simulation of the model like sampling period, simulation time, etc.
def modelSimulationParameters():
    samplingPeriod = 0.01  # seconds
    simulationTime = 9  # seconds
    initTime = 9  # seconds
    return samplingPeriod, simulationTime, initTime

# Contains the physical properties of the model like viscosity, etc. and other physiological parameters
def modelParameters():
    hr:float = 60.0
    rr:float = 20.0
    totalBloodVolume:float = 4000.0 # Total Blood Volume
    totalLungVolume:float = 2000.0  # mL
    airViscosity:float = mL.pa2mmHg(1.81e-5)  # kg/(m*s)  Pa*s (converted to mmHg)
    bloodViscosity:float = mL.pa2mmHg(3.5e-3) # kg/(m*s)  Pa*s (converted to mmHg)

    return hr, rr, totalBloodVolume, totalLungVolume, airViscosity, bloodViscosity

# Contains the parameters for the settings of the ventilator, this governs the ventilator function
def modelVentilatorParameters():
    ventilatorParams = {
        'inputName': 'u',
        'dcComponent': 12,  # baseline prassure mmHg
        'amplitude': 7,  # breath amplitude mmHg
        'I': 1,
        'E': 2,
        'slopeFraction': 0.2,
    }
    return ventilatorParams

####################################### Settings to plot the model (not in use for now)
def modelPlottingParameters():
    params = {
        'printData': True,
        'save': False,
        'plot': True,
        'legend': False,
        'dpi': 500,
        'lWidth': 0.8,
        'legendSize': 4,
        'plot2nd': False,
    }
    return params

# Contains the  parameters for the tree building algorithm that will give the model a visual representation
def modelTreeParams():
    treeParams = {
        ####################################### Starting Parameters
        'start_coordinate': [0, 0], # starting coordinate
        'angle': (3 * np.pi) / 2,   # first angle -> points down
        ####################################### Tree parameters
        'branchAngle': np.pi / 2.5, # angle between branches
        'lengthDecay': 1.6,         # length decay factor
        'plane': 'xz',              # plane of the tree
    }
    return treeParams

# Contains the parameters for the gas exchange model like initial partial pressures and membrane resistances
def modelGasExchangeParameters():
    gasParams = {
        'partialPressures': ['O2', 'C2', 'N2'],
        
        # Membrane Resistances
        'R_Tissues_C2': 1.5,
        'R_Tissues_O2': 4,
        'R_Alveoli_C2': 0.05/8,
        'R_Alveoli_O2': 4/8,
        'R_Alveoli_N2': 0.625/8,

        # Total Pressures
        'atmosphericPressure': 760,
        'arterialGasPressure': 755,
        'venousGasPressure': 706,
        'tissueGasPressure': 700,
        
        # Atmosphere partial pressures of all 3 gases
        'atmosphericPressuresY0': [
            145,
            7,
            608
        ], 
        # Alveoli partial pressures of all 3 gases
        'alveoliPressuresY0': [
            87,
            67,
            523
        ],
        # Arterial partial pressures of all 3 gases
        'arterialPartialPressuresY0': [
            81.6,
            66,
            524
        ], 
        # Venous partial pressures of all 3 gases
        'venousPartialPressuresY0': [
            71.5,
            91,
            524
        ], 
        # Tissues partial pressures of all 3 gases
        'tissuePartialPressuresY0': [
            30,
            122,
            608
        ], 
    }
    return gasParams

# Contains the parameters and structure of the 3 level simple lung model
def modelSimpleLungParameters(atmosphericPressure):
    
    lungParams = {
        
        #################### Trachea
        'C_Lu|0': 2.4,
        'C_y0_Lu|0': 28,

        'TypeR_Lu|0': 3,
        'R_Lu|0': 0.001,
        'L_Lu|0': 0.0001,
        'L_y0_Lu|0': 0,

        #################### Bronchia
        'C_Lu|0|0': 3,
        'C_y0_Lu|0|0': 36,

        'TypeR_Lu|0|0': 1,
        'R_Lu|0|0': 0.002,
        'L_Lu|0|0': 0.0001,
        'L_y0_Lu|0|0': 0,

        #################### Alveoli
        'C_Lu|0|0|0': 100,
        'C_y0_Lu|0|0|0': 1400,

        'TypeR_Lu|0|0|0': 1,
        'R_Lu|0|0|0': 0.003,
        'L_Lu|0|0|0': 0.0001,
        'L_y0_Lu|0|0|0': 0,
    }

    capacitorsResp = {
        'Lu|0': {
            'id': 'C_Lu|0',
            'pressure': 'V_Lu|0',
            'typeC': 1,
            'typeC_params': {
                'C': lungParams['C_Lu|0'],
                'biasPressure': atmosphericPressure,
            },
            'y0': lungParams['C_y0_Lu|0'],
        },
        'Lu|0|0': {
            'id': 'C_Lu|0|0',
            'pressure': 'V_Lu|0|0',
            'typeC': 1,
            'typeC_params': {
                'C': lungParams['C_Lu|0|0'],
                'biasPressure': atmosphericPressure,
            },
            'y0': lungParams['C_y0_Lu|0|0'],
        },
        'Lu|0|0|0': {
            'id': 'C_Lu|0|0|0',
            'pressure': 'V_Lu|0|0|0',
            'typeC': 1,
            'typeC_params': {
                'C': lungParams['C_Lu|0|0|0'],
                'biasPressure': atmosphericPressure,
            },
            'y0': lungParams['C_y0_Lu|0|0|0'],
        },
    }

    resistorsResp = {
        'Lu|0': {
            'id': 'R_Lu|0',
            'current': 'I_Lu|0',
            'pressureIn': 'u',
            'pressureOut': 'V_Lu|0',
            'typeR': lungParams['TypeR_Lu|0'],
            'typeR_params': {
                'R': lungParams['R_Lu|0'],
                'L': lungParams['L_Lu|0'],
                'y0': lungParams['L_y0_Lu|0'],
            },
        },
        'Lu|0|0': {
            'id': 'R_Lu|0|0',
            'current': 'I_Lu|0|0',
            'pressureIn': 'V_Lu|0',
            'pressureOut': 'V_Lu|0|0',
            'typeR': lungParams['TypeR_Lu|0|0'],
            'typeR': 1,
            'typeR_params': {
                'R': lungParams['R_Lu|0|0'],
                'L': lungParams['L_Lu|0|0'],
                'y0': lungParams['L_y0_Lu|0|0'],
            },
        },
        'Lu|0|0|0': {
            'id': 'R_Lu|0|0|0',
            'current': 'I_Lu|0|0|0',
            'pressureIn': 'V_Lu|0|0',
            'pressureOut': 'V_Lu|0|0|0',
            'typeR': lungParams['TypeR_Lu|0|0|0'],
            'typeR': 1,
            'typeR_params': {
                'R': lungParams['R_Lu|0|0|0'],
                'L': lungParams['L_Lu|0|0|0'],
                'y0': lungParams['L_y0_Lu|0|0|0'],
            },
        },
    }

    modelComponentsResp = [
        {
            'id': 'Lu|0',
            'level': 0,
            'isLeaf': False,
            'currentsIn': ['I_Lu|0'],
            'currentsOut': ['I_Lu|0|0'],
            'pressure': 'V_Lu|0',
            'capacitor': capacitorsResp['Lu|0'],
            'resistorsIn': [resistorsResp['Lu|0']],
            'resistorsOut': [resistorsResp['Lu|0|0']],
            'parent': '',
            'children': ['Lu|0|0'],
            'partialPressures': [],
            'partialPressuresY0': [],
        },
        {
            'id': 'Lu|0|0',
            'level': 1,
            'isLeaf': False,
            'currentsIn': ['I_Lu|0|0'],
            'currentsOut': ['I_Lu|0|0|0'],
            'pressure': 'V_Lu|0|0',
            'capacitor': capacitorsResp['Lu|0|0'],
            'resistorsIn': [resistorsResp['Lu|0|0']],
            'resistorsOut': [resistorsResp['Lu|0|0|0']],
            'parent': 'Lu|0',
            'children': ['Lu|0|0|0'],
            'partialPressures': [],
            'partialPressuresY0': [],
        },
        {
            'id': 'Lu|0|0|0',
            'level': 2,
            'isLeaf': True,
            'currentsIn': ['I_Lu|0|0|0'],
            'currentsOut': [],
            'pressure': 'V_Lu|0|0|0',
            'capacitor': capacitorsResp['Lu|0|0|0'],
            'resistorsIn': [resistorsResp['Lu|0|0|0']],
            'resistorsOut': [],
            'parent': 'Lu|0|0',
            'children': [],
            'partialPressures': [],
            'partialPressuresY0': [],
        },
    ]

    return modelComponentsResp

# Contains the parameters and structure of the 3 level simple lung model
def modelSimpleLungParametersCustom(atmosphericPressure, ParamArray):
    
    lungParams = {
        
        #################### Trachea
        'C_Lu|0': 2.4,
        'C_y0_Lu|0': 28,

        'TypeR_Lu|0': 3,
        'R_Lu|0': 0.001,
        'L_Lu|0': 0.0001,
        'L_y0_Lu|0': 0,

        #################### Bronchia
        'C_Lu|0|0': 3,
        'C_y0_Lu|0|0': 36,

        'TypeR_Lu|0|0': 1,
        'R_Lu|0|0': 0.002,
        'L_Lu|0|0': 0.0001,
        'L_y0_Lu|0|0': 0,

        #################### Alveoli
        'C_Lu|0|0|0': 100,
        'C_y0_Lu|0|0|0': 1400,

        'TypeR_Lu|0|0|0': 1,
        'R_Lu|0|0|0': 0.003,
        'L_Lu|0|0|0': 0.0001,
        'L_y0_Lu|0|0|0': 0,
    }

    lungParams['R_Lu|0'] = ParamArray[0]
    lungParams['R_Lu|0|0'] = ParamArray[1]
    lungParams['R_Lu|0|0|0'] = ParamArray[2]
    lungParams['C_Lu|0'] = ParamArray[3]
    lungParams['C_Lu|0|0'] = ParamArray[4]
    lungParams['C_Lu|0|0|0'] = ParamArray[5]


    capacitorsResp = {
        'Lu|0': {
            'id': 'C_Lu|0',
            'pressure': 'V_Lu|0',
            'typeC': 1,
            'typeC_params': {
                'C': lungParams['C_Lu|0'],
                'biasPressure': atmosphericPressure,
            },
            'y0': lungParams['C_y0_Lu|0'],
        },
        'Lu|0|0': {
            'id': 'C_Lu|0|0',
            'pressure': 'V_Lu|0|0',
            'typeC': 1,
            'typeC_params': {
                'C': lungParams['C_Lu|0|0'],
                'biasPressure': atmosphericPressure,
            },
            'y0': lungParams['C_y0_Lu|0|0'],
        },
        'Lu|0|0|0': {
            'id': 'C_Lu|0|0|0',
            'pressure': 'V_Lu|0|0|0',
            'typeC': 1,
            'typeC_params': {
                'C': lungParams['C_Lu|0|0|0'],
                'biasPressure': atmosphericPressure,
            },
            'y0': lungParams['C_y0_Lu|0|0|0'],
        },
    }

    resistorsResp = {
        'Lu|0': {
            'id': 'R_Lu|0',
            'current': 'I_Lu|0',
            'pressureIn': 'u',
            'pressureOut': 'V_Lu|0',
            'typeR': lungParams['TypeR_Lu|0'],
            'typeR_params': {
                'R': lungParams['R_Lu|0'],
                'L': lungParams['L_Lu|0'],
                'y0': lungParams['L_y0_Lu|0'],
            },
        },
        'Lu|0|0': {
            'id': 'R_Lu|0|0',
            'current': 'I_Lu|0|0',
            'pressureIn': 'V_Lu|0',
            'pressureOut': 'V_Lu|0|0',
            'typeR': lungParams['TypeR_Lu|0|0'],
            'typeR': 1,
            'typeR_params': {
                'R': lungParams['R_Lu|0|0'],
                'L': lungParams['L_Lu|0|0'],
                'y0': lungParams['L_y0_Lu|0|0'],
            },
        },
        'Lu|0|0|0': {
            'id': 'R_Lu|0|0|0',
            'current': 'I_Lu|0|0|0',
            'pressureIn': 'V_Lu|0|0',
            'pressureOut': 'V_Lu|0|0|0',
            'typeR': lungParams['TypeR_Lu|0|0|0'],
            'typeR': 1,
            'typeR_params': {
                'R': lungParams['R_Lu|0|0|0'],
                'L': lungParams['L_Lu|0|0|0'],
                'y0': lungParams['L_y0_Lu|0|0|0'],
            },
        },
    }

    modelComponentsResp = [
        {
            'id': 'Lu|0',
            'level': 0,
            'isLeaf': False,
            'currentsIn': ['I_Lu|0'],
            'currentsOut': ['I_Lu|0|0'],
            'pressure': 'V_Lu|0',
            'capacitor': capacitorsResp['Lu|0'],
            'resistorsIn': [resistorsResp['Lu|0']],
            'resistorsOut': [resistorsResp['Lu|0|0']],
            'parent': '',
            'children': ['Lu|0|0'],
            'partialPressures': [],
            'partialPressuresY0': [],
        },
        {
            'id': 'Lu|0|0',
            'level': 1,
            'isLeaf': False,
            'currentsIn': ['I_Lu|0|0'],
            'currentsOut': ['I_Lu|0|0|0'],
            'pressure': 'V_Lu|0|0',
            'capacitor': capacitorsResp['Lu|0|0'],
            'resistorsIn': [resistorsResp['Lu|0|0']],
            'resistorsOut': [resistorsResp['Lu|0|0|0']],
            'parent': 'Lu|0',
            'children': ['Lu|0|0|0'],
            'partialPressures': [],
            'partialPressuresY0': [],
        },
        {
            'id': 'Lu|0|0|0',
            'level': 2,
            'isLeaf': True,
            'currentsIn': ['I_Lu|0|0|0'],
            'currentsOut': [],
            'pressure': 'V_Lu|0|0|0',
            'capacitor': capacitorsResp['Lu|0|0|0'],
            'resistorsIn': [resistorsResp['Lu|0|0|0']],
            'resistorsOut': [],
            'parent': 'Lu|0|0',
            'children': [],
            'partialPressures': [],
            'partialPressuresY0': [],
        },
    ]

    return modelComponentsResp


# Contains the parameters and structure of the 6 level simple cardiovascular model
def modelSimpleHeartParameters(totalVolume:float, atmosphericPressure:float):
    # Contains the distribution of the volumes of the different parts of the model
    #   to use as initial conditions for the cardiovascular model
    volDistro = {
        'Hl': 0.03 * totalVolume,
        'Hr': 0.03 * totalVolume,

        'As': 0.10 * totalVolume,
        'Cs': 0.07 * totalVolume,
        'Vs': 0.66 * totalVolume,
        
        'Ap': 0.03 * totalVolume,
        'Cp': 0.03 * totalVolume,
        'Vp': 0.03 * totalVolume,

        'Ar': 0.01 * totalVolume,
        'Al': 0.01 * totalVolume,
    }

    # Contains the distribution of the capacities of the different parts of the cardiovascular model
    capDistro = {
        'Hl_Emax': 1/0.05, #15.2,
        'Hl_Emin': 1/20, #0.03,
        
        'Hr_Emax': 1/4,#5,
        'Hr_Emin': 1/20, #0.17,

        'As': 1.5,
        'Cs': 6.5,
        'Vs': 300,
        
        'Ap': 7,
        'Cp': 6.5,
        'Vp': 15,

        'Ar': 6,
        'Al': 6,
    }

    # Contains the distribution of the resistances of the different parts of the cardiovascular model
    resDistro = {
        'Hl': 0.0010,
        'Hr': 0.0010,

        'As': 0.9660,
        'Cs': 0.0300,
        'Vs': 0.00020,

        'Ap': 0.0200,
        'Cp': 0.0005,
        'Vp': 0.0002,

        'Ar': 0.00010,
        'Al': 0.00010,
    }

    # Contains the distribution of the inductances of the different parts of the cardiovascular model
    indDistro = {
        'As_L': 0.005,  # Inductance
        'As_I': 00,  # Start Current

        'Ap_L': 0.00001,  # Inductance
        'Ap_I': 000,  # Start Current

        'Vs_L': 0.0003,  # Inductance
        'Vs_I': 000,  # Start Current

        'Vp_L': 0.0003,  # Inductance
        'Vp_I': 00,  # Start Current
    }

    capacitorsCardio = {
        'Hl': {
            'id': 'C_Hl',
            'pressure': 'V_Hl',
            'typeC': 0,
            'typeC_params': {
                'Emax': capDistro['Hl_Emax'],
                'Emin': capDistro['Hl_Emin'],
                'biasPressure': atmosphericPressure,
            },
            'y0': volDistro['Hl'],
        },
        'As': {
            'id': 'C_As',
            'pressure': 'V_As',
            'typeC': 1,
            'typeC_params': {
                'C': capDistro['As'],
                'biasPressure': atmosphericPressure,
            },
            'y0': volDistro['As'],
        },
        'Cs': {
            'id': 'C_Cs',
            'pressure': 'V_Cs',
            'typeC': 1,
            'typeC_params': {
                'C': capDistro['Cs'],
                'biasPressure': atmosphericPressure,
            },
            'y0': volDistro['Cs'],
        },
        'Vs': {
            'id': 'C_Vs',
            'pressure': 'V_Vs',
            'typeC': 1,
            'typeC_params': {
                'C': capDistro['Vs'],
                'biasPressure': atmosphericPressure,
            },
            'y0': volDistro['Vs'],
        },
        'Ar': {
            'id': 'C_Ar',
            'pressure': 'V_Ar',
            'typeC': 1,
            'typeC_params': {
                'C': capDistro['Ar'],
                'biasPressure': atmosphericPressure,
            },
            'y0': volDistro['Ar'],
        },
        'Hr': {
            'id': 'C_Hr',
            'pressure': 'V_Hr',
            'typeC': 0,
            'typeC_params': {
                'Emax': capDistro['Hr_Emax'],
                'Emin': capDistro['Hr_Emin'],
                'biasPressure': atmosphericPressure,
            },
            'y0': volDistro['Hr'],
        },
        'Ap': {
            'id': 'C_Ap',
            'pressure': 'V_Ap',
            'typeC': 1,
            'typeC_params': {
                'C': capDistro['Ap'],
                'biasPressure': atmosphericPressure,
            },
            'y0': volDistro['Ap'],
        },
        'Cp': {
            'id': 'C_Cp',
            'pressure': 'V_Cp',
            'typeC': 1,
            'typeC_params': {
                'C': capDistro['Cp'],
                'biasPressure': atmosphericPressure,
            },
            'y0': volDistro['Cp'],
        },
        'Vp': {
            'id': 'C_Vp',
            'pressure': 'V_Vp',
            'typeC': 1,
            'typeC_params': {
                'C': capDistro['Vp'],
                'biasPressure': atmosphericPressure,
            },
            'y0': volDistro['Vp'],
        },
        'Al': {
            'id': 'C_Al',
            'pressure': 'V_Al',
            'typeC': 1,
            'typeC_params': {
                'C': capDistro['Al'],
                'biasPressure': atmosphericPressure,
            },
            'y0': volDistro['Al'],
        },
    }

    resistorsCardio = {
        'Hl': {
            'id': 'R_Hl',
            'current': 'I_Hl',
            'pressureIn': 'V_Hl',
            'pressureOut': 'V_As',
            'typeR': 2,
            'typeR_params': {
                'R': resDistro['Hl'],
                'L': indDistro['As_L'],
                'y0': indDistro['As_I'],
                'threshold': 0,
            },
        },
        'As': {
            'id': 'R_As',
            'current': 'I_As',
            'pressureIn': 'V_As',
            'pressureOut': 'V_Cs',
            'typeR': 1,
            'typeR_params': {
                'R': resDistro['As'],
                'L': indDistro['As_L'],
                'y0': indDistro['As_I']
            },
        },
        'Cs': {
            'id': 'R_Cs',
            'current': 'I_Cs',
            'pressureIn': 'V_Cs',
            'pressureOut': 'V_Vs',
            'typeR': 1,
            'typeR_params': {
                'R': resDistro['Cs'],
            },
        },
        'Vs': {
            'id': 'R_Vs',
            'current': 'I_Vs',
            'pressureIn': 'V_Vs',
            'pressureOut': 'V_Ar',
            'typeR': 1,
            'typeR_params': {
                'R': resDistro['Vs'],
                'L': indDistro['Vs_L'],
                'y0': indDistro['Vs_I'],
            },
        },
        'Ar': {
            'id': 'R_Ar',
            'current': 'I_Ar',
            'pressureIn': 'V_Ar',
            'pressureOut': 'V_Hr',
            'typeR': 2,
            'typeR_params': {
                'R': resDistro['Ar'],
                'L': indDistro['As_L'],
                'y0': indDistro['As_I'],
                'threshold': 0
            },
        },
        'Hr': {
            'id': 'R_Hr',
            'current': 'I_Hr',
            'pressureIn': 'V_Hr',
            'pressureOut': 'V_Ap',
            'typeR': 2,
            'typeR_params': {
                'R': resDistro['Hr'],
                'L': indDistro['As_L'],
                'y0': indDistro['As_I'],
                'threshold': 0,
            },
        },
        'Ap': {
            'id': 'R_Ap',
            'current': 'I_Ap',
            'pressureIn': 'V_Ap',
            'pressureOut': 'V_Cp',
            'typeR': 3,
            'typeR_params': {
                'R': resDistro['Ap'],
                'L': indDistro['Ap_L'],
                'y0': indDistro['Ap_I'],
            },
        },
        'Cp': {
            'id': 'R_Cp',
            'current': 'I_Cp',
            'pressureIn': 'V_Cp',
            'pressureOut': 'V_Vp',
            'typeR': 1,
            'typeR_params': {
                'R': resDistro['Cp'],
            },
        },
        'Vp': {
            'id': 'R_Vp',
            'current': 'I_Vp',
            'pressureIn': 'V_Vp',
            'pressureOut': 'V_Al',
            'typeR': 1,
            'typeR_params': {
                'R': resDistro['Vp'],
                'L': indDistro['Vp_L'],
                'y0': indDistro['Vp_I']
            },
        },
        'Al': {
            'id': 'R_Al',
            'current': 'I_Al',
            'pressureIn': 'V_Al',
            'pressureOut': 'V_Hl',
            'typeR': 2,
            'typeR_params': {
                'R': resDistro['Al'],
                'L': indDistro['As_L'],
                'y0': indDistro['As_I'],
                'threshold': 0
            },
        },
    }

    modelComponents = [
        {
            'id': 'Hl',
            'level': 0,
            'isLeaf': False,
            'currentsIn': ['I_Al'],
            'currentsOut': ['I_Hl'],
            'pressure': 'V_Hl',
            'capacitor': capacitorsCardio['Hl'],
            'resistorsIn': [resistorsCardio['Al']],
            'resistorsOut': [resistorsCardio['Hl']],
            'bloodType' : 'arterial',
        },
        {
            'id': 'As',
            'level': 0,
            'isLeaf': False,
            'currentsIn': ['I_Hl'],
            'currentsOut': ['I_As'],
            'pressure': 'V_As',
            'capacitor': capacitorsCardio['As'],
            'resistorsIn': [resistorsCardio['Hl']],
            'resistorsOut': [resistorsCardio['As']],
            'bloodType' : 'arterial',
        },
        {
            'id': 'Cs',
            'level': 0,
            'isLeaf': False,
            'currentsIn': ['I_As'],
            'currentsOut': ['I_Cs'],
            'pressure': 'V_Cs',
            'capacitor': capacitorsCardio['Cs'],
            'resistorsIn': [resistorsCardio['As']],
            'resistorsOut': [resistorsCardio['Cs']],
            'bloodType' : 'arterial',
        },
        {
            'id': 'Vs',
            'level': 0,
            'isLeaf': False,
            'currentsIn': ['I_Cs'],
            'currentsOut': ['I_Vs'],
            'pressure': 'V_Vs',
            'capacitor': capacitorsCardio['Vs'],
            'resistorsIn': [resistorsCardio['Cs']],
            'resistorsOut': [resistorsCardio['Vs']],
            'bloodType' : 'venous',
        },
        {
            'id': 'Ar',
            'level': 0,
            'isLeaf': False,
            'currentsIn': ['I_Vs'],
            'currentsOut': ['I_Ar'],
            'pressure': 'V_Ar',
            'capacitor': capacitorsCardio['Ar'],
            'resistorsIn': [resistorsCardio['Vs']],
            'resistorsOut': [resistorsCardio['Ar']],
            'bloodType' : 'venous',
        },
        {
            'id': 'Hr',
            'level': 0,
            'isLeaf': False,
            'currentsIn': ['I_Ar'],
            'currentsOut': ['I_Hr'],
            'pressure': 'V_Hr',
            'capacitor': capacitorsCardio['Hr'],
            'resistorsIn': [resistorsCardio['Ar']],
            'resistorsOut': [resistorsCardio['Hr']],
            'bloodType' : 'venous',
        },
        {
            'id': 'Ap',
            'level': 0,
            'isLeaf': False,
            'currentsIn': ['I_Hr'],
            'currentsOut': ['I_Ap'],
            'pressure': 'V_Ap',
            'capacitor': capacitorsCardio['Ap'],
            'resistorsIn': [resistorsCardio['Hr']],
            'resistorsOut': [resistorsCardio['Ap']],
            'bloodType' : 'venous',
        },
        {
            'id': 'Cp',
            'level': 0,
            'isLeaf': False,
            'currentsIn': ['I_Ap'],
            'currentsOut': ['I_Cp'],
            'pressure': 'V_Cp',
            'capacitor': capacitorsCardio['Cp'],
            'resistorsIn': [resistorsCardio['Ap']],
            'resistorsOut': [resistorsCardio['Cp']],
            'bloodType' : 'venous',
        },
        {
            'id': 'Vp',
            'level': 0,
            'isLeaf': False,
            'currentsIn': ['I_Cp'],
            'currentsOut': ['I_Vp'],
            'pressure': 'V_Vp',
            'capacitor': capacitorsCardio['Vp'],
            'resistorsIn': [resistorsCardio['Cp']],
            'resistorsOut': [resistorsCardio['Vp']],
            'bloodType' : 'arterial',
        },
        {
            'id': 'Al',
            'level': 0,
            'isLeaf': False,
            'currentsIn': ['I_Vp'],
            'currentsOut': ['I_Al'],
            'pressure': 'V_Al',
            'capacitor': capacitorsCardio['Al'],
            'resistorsIn': [resistorsCardio['Vp']],
            'resistorsOut': [resistorsCardio['Al']],
            'bloodType' : 'arterial',
        },
    ]

    return modelComponents



# Contains the parameters for the geometric progession of the Arterial pulmonary tree
def modelArterialTreeParameters():
    Parameters = {
        # parameters for the geometric progession of the radius of the vessels
        'radius': 0.037/2, 
        'rFactor':1.18,
        # parameters for the geometric progession of the length of the vessels
        'length': 0.09, 
        'lFactor': 1.28, 
        # parameters for the geometric progession of the thickness of the vessels
        'thickness': 6e-5, 
        'tFactor': 1.13,
        # parameters for the geometric progession of the young Modulus of the vessels
        'youngM': 1.5*1000000, 
        'yFactor': 1.15,

        'nrGenerations': 17,                         # Number of generations of the arterial tree
        'cardioRegions2' : [0, 1, 12, 18],           # Definition of the regions for the 2 generation model    
        'cardioRegions5' : [0, 1, 6, 9, 12, 17, 18], # Definition of the regions for the 5 generation model
        
        # Contains extra parameters for the estimation of the resistances and capacitances of the arterial tree
        'resistors': 1, #Multiplier to bias the Resistors
        'capacitors':1, #Multiplier to bias the Capacitors
        'nrBranches':[2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,6,6,6],
    }
    return Parameters

# Contains the parameters for the geometric progession of the Venous pulmonary tree
def modelVenousTreeParameters():
    Parameters = {
        # parameters for the geometric progession of the radius of the vessels
        'radius': 0.05/2, 
        'rFactor':1.20, 
        # parameters for the geometric progession of the length of the vessels
        'length': 0.04, 
        'lFactor': 1.3, 
        # parameters for the geometric progession of the thickness of the vessels
        'thickness': 4e-5, 
        'tFactor': 1.1, 
        # parameters for the geometric progession of the young Modulus of the vessels
        'youngM': 1.5*1000000, 
        'yFactor': 1.2,

        'nrGenerations': 16,                         # Number of generations of the arterial tree
        'cardioRegions2' : [0, 6, 12, 17],           # Definition of the regions for the 2 generation model    
        'cardioRegions5' : [0, 3, 6, 9, 12, 16, 17],  # Definition of the regions for the 5 generation model
   
        # Contains extra parameters for the estimation of the resistances and capacitances of the venous tree
        'resistors': 1, # Multiplier to bias the Resistors
        'capacitors':1, # Multiplier to bias the Capacitors
        'nrBranches': [4,2,2,2,2,2,2,2,2,4,4,4,4,4,6,6,6],
    }
    return Parameters

# Contains the parameters for the geometric progession of the Lung pulmonary tree
def modelLungTreeParameters():
    Parameters = {
        # parameters for the geometric progession of the radius of the vessels
        'radius': 0.05/2, 
        'rFactor':1.20, 
        # parameters for the geometric progession of the length of the vessels
        'length': 0.04, 
        'lFactor': 1.3, 
        # parameters for the geometric progession of the thickness of the vessels
        'thickness': 4e-5, 
        'tFactor': 1.1, 
        # parameters for the geometric progession of the young Modulus of the vessels
        'youngM': 1.5*1000000, 
        'yFactor': 1.2,

        'nrGenerations': 16,                         # Number of generations of the arterial tree
        'cardioRegions2' : [0, 6, 12, 17],           # Definition of the regions for the 2 generation model    
        'cardioRegions5' : [0, 3, 6, 9, 12, 16, 17],  # Definition of the regions for the 5 generation model
        
        # Contains extra parameters for the estimation of the resistances and capacitances of the venous tree
        'resistors': 1, # Multiplier to bias the Resistors
        'capacitors':1, # Multiplier to bias the Capacitors
        'nrBranches': [4,2,2,2,2,2,2,2,2,4,4,4,4,4,6,6,6],
    }
    return Parameters





