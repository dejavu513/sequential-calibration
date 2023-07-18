import numpy as np
import Parameters
import modelLibrary as mL
import treeClass_1 as tree
import components as c


class ModelStructure:

    ############################################ Simulation Params
    hasLung: bool = False
    hasHeart: bool = False
    hasGasExchange : bool= False
    nrLevels: int = 0
    samplingPeriod: float = 0.01
    simulationTime: float = 3.0
    initTime: float = 3.0

    ############################################ Plotting Params
    plotParameters: dict = {}

    ############################################ Model Params
    hr: float = 60.0
    rr: float = 20.0
    totalBloodVolume: float = 4000.0
    totalLungVolume: float = 2000.0
    airViscosity: float = mL.pa2mmHg(1.81e-5)
    bloodViscosity: float = mL.pa2mmHg(3.5e-3)
    atmosphericPressure: float = 760.0

    gasExchangeParameters: dict = {}
    ventilatorParams: dict = {}

    ############################################ Simple Model Compartemnts
    baseLungCompartments: dict = {}
    baseHeartCompartments: dict = {}

    ############################################ Tree Model Compartemnts
    treeParameters : dict = {}
    arterialTreeParameters : dict = {}
    venousTreeParameters : dict = {}
    lungTreeParameters : dict = {}

    ############################################ Model Compartments, Resistors and Capacitors
    resistors = []
    capacitors = []
    compartments = []

    ############################################ Equation Model Variables
    states = {}

    '''
    ██ ███    ██ ██ ████████ ██  █████  ██      ██ ███████  █████  ████████ ██  ██████  ███    ██ 
    ██ ████   ██ ██    ██    ██ ██   ██ ██      ██    ███  ██   ██    ██    ██ ██    ██ ████   ██ 
    ██ ██ ██  ██ ██    ██    ██ ███████ ██      ██   ███   ███████    ██    ██ ██    ██ ██ ██  ██ 
    ██ ██  ██ ██ ██    ██    ██ ██   ██ ██      ██  ███    ██   ██    ██    ██ ██    ██ ██  ██ ██ 
    ██ ██   ████ ██    ██    ██ ██   ██ ███████ ██ ███████ ██   ██    ██    ██  ██████  ██   ████ 
    '''

    def __init__(self,lung, heart, gasExchange, nrLevels, ParamArray) -> None:
        ############################################ Simulation Params
        self.hasLung = lung
        self.hasHeart = heart
        self.hasGasExchange = gasExchange
        self.nrLevels = nrLevels
        self.samplingPeriod, self.simulationTime, self.initTime = Parameters.modelSimulationParameters()

        ############################################ Plotting Params
        self.plotParameters = Parameters.modelPlottingParameters()

        ############################################ Model Params
        self.hr, self.rr, self.totalBloodVolume, self.totalLungVolume, self.airViscosity, self.bloodViscosity = Parameters.modelParameters()
        
        self.gasExchangeParameters = Parameters.modelGasExchangeParameters()
        self.atmosphericPressure = self.gasExchangeParameters['atmosphericPressure']
        self.ventilatorParams = Parameters.modelVentilatorParameters()

        ############################################ Base Model Compartemnts
        self.baseLungCompartments = Parameters.modelSimpleLungParametersCustom(self.atmosphericPressure,ParamArray)
        self.baseHeartCompartments = Parameters.modelSimpleHeartParameters(self.totalBloodVolume, self.atmosphericPressure)

        ############################################ Tree Model Compartemnts
        self.treeParameters = Parameters.modelTreeParams()
        self.lungTreeParameters = Parameters.modelLungTreeParameters()
        self.arterialTreeParameters = Parameters.modelArterialTreeParameters()
        self.venousTreeParameters = Parameters.modelVenousTreeParameters()

        if self.nrLevels < 2:
            if self.hasLung:
                self.init_Simple_Lung()
            if self.hasHeart:
                self.init_Simple_Heart()
        else:
            if self.hasLung:
                self.init_Tree_Lung()
            if self.hasHeart:
                self.init_Tree_Heart()




    '''
    ██████   █████  ███████ ███████     ███    ███  ██████  ██████  ███████ ██      ███████ 
    ██   ██ ██   ██ ██      ██          ████  ████ ██    ██ ██   ██ ██      ██      ██      
    ██████  ███████ ███████ █████       ██ ████ ██ ██    ██ ██   ██ █████   ██      ███████ 
    ██   ██ ██   ██      ██ ██          ██  ██  ██ ██    ██ ██   ██ ██      ██           ██ 
    ██████  ██   ██ ███████ ███████     ██      ██  ██████  ██████  ███████ ███████ ███████
    '''
    # adds the compartments of the basic cardiovascular model to the list of compartments
    # adds the resistors to the list of resistors
    # adds the capacitors to the list of capacitors
    # adds the partial pressures to the list of partial pressures if gas exchange is enabled
    # adds the gas exchange resistors to the list of resistors
    def init_Simple_Heart(self):
        # Appends the cardiovascular compartments to the list of compartments
        self.compartments = self.compartments + self.baseHeartCompartments

        for idx, compartment in enumerate(self.baseHeartCompartments):
            # Appends the capacitors and resistors to the list of capacitors and resistors
            self.capacitors.append(compartment['capacitor'])
            self.resistors.append(compartment['resistorsIn'][0])

            # If gas exchange is enabled, add the partial pressures and create the resistors for the gas exchange
            if self.hasGasExchange:
                compartment['partialPressures'] = []
                compartment['gasExchangeResistors'] = []

                # Set the initial partial pressures depending on the blood type
                if compartment['bloodType'] == 'venous':
                    compartment['partialPressuresY0'] = self.gasExchangeParameters['venousPartialPressuresY0']
                elif compartment['bloodType'] == 'arterial':
                    compartment['partialPressuresY0'] = self.gasExchangeParameters['arterialPartialPressuresY0']

                # Run for every gas species
                for idx, pp in enumerate(self.gasExchangeParameters['partialPressures']):
                    name = pp + '_' + compartment['id']
                    compartment['partialPressures'].append('V_' + name)

                    # Create and add the resistors for the gas exchange on the tissues only
                    if (compartment['id'] == 'Cs') and ('N2' not in pp):
                        resistor = {
                            'id': 'R_' + name + '_Tissues',
                            'current': 'I_' + name + '_Tissues',
                            'pressureIn': 'V_' + name,
                            'pressureOut': 'V_' + name + '_Tissues',
                            'typeR': 1,
                            'typeR_params': {
                                'R': 0.0
                            },
                        }
                        # Set the resistor value of the membrane depending on the gas species
                        if pp == 'O2':
                            resistor['typeR_params']['R'] = self.gasExchangeParameters['R_Tissues_O2']
                        elif pp == 'C2':
                            resistor['typeR_params']['R'] = self.gasExchangeParameters['R_Tissues_C2']
                        
                        compartment['gasExchangeResistors'].append(resistor)
                        self.resistors.append(resistor)

        print('Simple Heart initialised')

    # adds the compartments of the basic lung model to the list of compartments
    # adds the resistors to the list of resistors
    # adds the capacitors to the list of capacitors
    # adds the partial pressures to the list of partial pressures if gas exchange is enabled
    # adds the gas exchange resistors to the list of resistors
    def init_Simple_Lung(self):
        # Appends the lung compartments to the list of compartments
        self.compartments = self.compartments + self.baseLungCompartments

        for idx, compartment in enumerate(self.baseLungCompartments):
            # Appends the capacitors and resistors to the list of capacitors and resistors
            self.capacitors.append(compartment['capacitor'])
            self.resistors.append(compartment['resistorsIn'][0])

            # If gas exchange is enabled, add the partial pressures and create the resistors for the gas exchange
            if self.hasGasExchange:
                # Set the initial partial pressures as the alveoli partial pressures
                compartment['partialPressuresY0'] = self.gasExchangeParameters['alveoliPressuresY0']
                compartment['partialPressures'] = []
                compartment['gasExchangeResistors'] = []

                # Run for every gas species
                for idx, pp in enumerate(self.gasExchangeParameters['partialPressures']):
                    name = pp + '_' + compartment['id']
                    compartment['partialPressures'].append('V_' + name)

                    if compartment['isLeaf']:
                        compartment['currentsOut'].append('I' + name)
                        resistor = {
                            'id': 'R_' + name,
                            'current': 'I_' + name,
                            'pressureIn': 'V_' + name,
                            'pressureOut': 'V_' + name + '_Blood',
                            'typeR': 1,
                            'typeR_params': {
                                'R': 0.0
                            },
                        }
                        # Set the resistor value of the membrane depending on the gas species
                        if pp == 'O2':
                            resistor['typeR_params']['R'] = self.gasExchangeParameters['R_Alveoli_O2']
                        elif pp == 'C2':
                            resistor['typeR_params']['R'] = self.gasExchangeParameters['R_Alveoli_C2']
                        elif pp == 'N2':
                            resistor['typeR_params']['R'] = self.gasExchangeParameters['R_Alveoli_N2']


                        if self.hasHeart:
                            # V_O2_Cp
                            resistor['pressureOut'] = 'V_' + pp + '_Cp'


                        compartment['gasExchangeResistors'].append(resistor)
                        self.resistors.append(resistor)
                        compartment['resistorsOut'].append(resistor)
        print('Simple Lung initialised')
    
    '''
    ████████ ██████  ███████ ███████     ███    ███  ██████  ██████  ███████ ██      ███████ 
       ██    ██   ██ ██      ██          ████  ████ ██    ██ ██   ██ ██      ██      ██      
       ██    ██████  █████   █████       ██ ████ ██ ██    ██ ██   ██ █████   ██      ███████ 
       ██    ██   ██ ██      ██          ██  ██  ██ ██    ██ ██   ██ ██      ██           ██ 
       ██    ██   ██ ███████ ███████     ██      ██  ██████  ██████  ███████ ███████ ███████
    '''
    # TODO build method
    def init_Tree_Heart(self):
        return 0

    # TODO build method
    def init_Tree_Lung(self):
        res, cap = self.getEquivalentResistors(self.lungTreeParameters, self.airViscosity)
        return 0
           

    def generateVesselConf(self, parameters, viscosity):

        # Parameter calculation formulas
        def calculateResistorValue(length, radius, viscosity):
            result = (8 * viscosity * length) / ((radius ** 4) * np.pi)
            return result
        def calculateCapacitorValue(length, radius, E, thickness):
            result = (3 * length * (radius ** 3) * np.pi) / (2 * E * thickness)
            return result
        def calculateInductorValue(length, radius, density):
            result = (density * length) / (np.pi * (radius ** 2))
            return result



        pa2mmHg = 0.00750062
        conf = [
            {
                'gen': 0,
                'radius': parameters['radius'],
                'length': parameters['length'],
                'ymodulus': parameters['youngM'],
                'thickness': parameters['thickness'],
                'resistance': calculateResistorValue(
                                parameters['length'], 
                                parameters['radius'], 
                                viscosity) / 1000,
                'capacity': calculateCapacitorValue(
                                parameters['length'], 
                                parameters['radius'], 
                                parameters['youngM'], 
                                parameters['thickness']) * (1000 / pa2mmHg)
            },
        ]
        for i in np.arange(1, parameters['nrGenerations'] + 1):
            radius = conf[i - 1]['radius'] / parameters['rFactor']
            length = conf[i - 1]['length'] / parameters['lFactor']
            ymodulus = conf[i - 1]['ymodulus'] / parameters['yFactor']
            thickness = conf[i - 1]['thickness'] / parameters['tFactor']

            genData = {
                'gen': i,
                'radius': radius,
                'length': length,
                'ymodulus': ymodulus,
                'thickness': thickness,
                'resistance': calculateResistorValue(length, radius, viscosity) / 1000,
                'capacity': calculateCapacitorValue(length, radius, ymodulus, thickness) * (1000 / pa2mmHg),
            }
            conf.append(genData)


        if self.nrLevels == 2:
            regions = parameters['cardioRegions2']
        else:
            regions = parameters['cardioRegions5']

        return conf, regions
    
    def getEquivalentResistors(self, parameters, viscosity):

        conf, regions = self.generateVesselConf(parameters,viscosity)
        
        last = regions[-1]
        resistorsArray = []
        capacitorsArray = []

        nrOriginalVessels = [1]
        nrBranches = parameters['nrBranches']


        for i, value in enumerate(nrBranches[1:]):
            newVal = nrOriginalVessels[i]*nrBranches[i]
            nrOriginalVessels.append(newVal)
        

        if self.nrLevels == 2:
            resDivisors = [1,1,4096]
        else:
            resDivisors = [1, 1, 16, 64, 1024, 32768]

        for i, value in enumerate(regions):
            if value != last:
                nrGen = regions[i + 1] - regions[i]
                genConf = conf[value: regions[i + 1]]

                treeParams = {
                    # -------------------------------- Coordinate Parameters
                    'level': 0,
                    'branch': 0,
                    'start_coordinate': [0, 0],
                    'angle': (3 * np.pi) / 2,  # first angle -> points down
                    'branchAngle': np.pi / 2.5,
                    'plane': 'xz',
                    # -------------------------------------- Tree parameters
                    'name': 'Lu',
                    'conf': genConf,
                    'inputName': 'u',
                    # --------------------------------------- Gas Parameters
                    'gasExchange': False,
                    'gases': ['O2', 'C2', 'N2'],
                    'gasConcentrations': [144.4, 6.84, 608.76],  # mmHg
                    'nrBranches': nrBranches[regions[i]:regions[i + 1]],
                    'nrOriginalVessels': nrOriginalVessels[regions[i]:regions[i + 1]]
                }
                trachea = tree.baseNode(treeParams)
                trachea.buildTreeForParameterEstimation(nrGen - 1, treeParams['branchAngle'])
                sub_tree = trachea.getTreeArray()

                nrTreeVessels = 2 ** i
                resVal = resDivisors[i]

                resistorsArray.append((sub_tree[0]['resistorsIn'][0]['typeR_params']['R'] / resVal) * parameters['resistors'])
                capacitorsArray.append((sub_tree[0]['capacitor']['typeC_params']['C'] / nrTreeVessels) * parameters['capacitors'])
        
        print('Cap ->' + str(capacitorsArray))
        print('Res ->' + str(resistorsArray))
        return resistorsArray, capacitorsArray

    '''
    ███████  ██████  ██    ██  █████  ████████ ██  ██████  ███    ██     ███    ███  █████  ██████  ██████  ██ ███    ██  ██████  
    ██      ██    ██ ██    ██ ██   ██    ██    ██ ██    ██ ████   ██     ████  ████ ██   ██ ██   ██ ██   ██ ██ ████   ██ ██       
    █████   ██    ██ ██    ██ ███████    ██    ██ ██    ██ ██ ██  ██     ██ ████ ██ ███████ ██████  ██████  ██ ██ ██  ██ ██   ███ 
    ██      ██ ▄▄ ██ ██    ██ ██   ██    ██    ██ ██    ██ ██  ██ ██     ██  ██  ██ ██   ██ ██      ██      ██ ██  ██ ██ ██    ██ 
    ███████  ██████   ██████  ██   ██    ██    ██  ██████  ██   ████     ██      ██ ██   ██ ██      ██      ██ ██   ████  ██████ 
    '''
    

    # Creates the maps to be used by the integrator to solve the equation model
    def getEquationMaps(self):
        pressures = {}
        resistorObjects = {}
        capacitorObjects = {}
        connectionbjects = {}
        dVdt = {}
        capacitors = {}
        flows = {}
        resistors = {}

        # creates the volume map
        dVdt = {compartments['id']:float(compartments['capacitor']['y0']) for compartments in self.compartments}

        # creates the pressure map
        pressures = {compartment['capacitor']['pressure']:i for i, compartment in enumerate(self.compartments)}
        
        # If there is a lung adds the ventilator pressure to the pressure map
        if self.hasLung:
            pressures['u'] = len(pressures)
            
        # creates the capacitor map
        def capacitorSelector(capacitor):
            if capacitor['typeC'] == 0:
                return  0.0
            elif capacitor['typeC'] == 1:
                return float(capacitor['typeC_params']['C'])
        capacitors = {capacitor['id']:capacitorSelector(capacitor) for capacitor in self.capacitors}
        
        # creates the flow map
        #flows = {resistor['current']:0.0 for resistor in self.resistors if resistor['typeR'] == 3}
        flows = {resistor['current']:0.0 for resistor in self.resistors}

        # creates the resistor map
        resistors = {resistor['id']:resistor['typeR_params']['R'] for resistor in self.resistors}
            
        # State variables are -> pressures['u'] | dVdT | capacitor | flows | resistors
        pPleural = {'Pleural': float(self.atmosphericPressure)}
        states = {**pPleural, **dVdt, **capacitors, **flows, **resistors}

        # Initialises the capacitor Objects
        hc = 1 / self.hr * 60  # Heart cycle duration (s)
        tSys = 0.3 * np.sqrt(hc)  # Systole duration (seconds)
        for compartment in self.compartments:
            capacitor = compartment['capacitor']

            if capacitor['typeC'] == 0:
                capacitorObjects[capacitor['pressure']] = c.ElastanceCapacitor(
                    capacitor['id'],
                    'Pleural',
                    capacitor['typeC_params']['Emax'], #Emax
                    capacitor['typeC_params']['Emin'], #Emin
                    hc, #HC
                    tSys, #Tsys
                    compartment['id'],
                    0.0 #slopeFreq
                    
                )
            elif capacitor['typeC'] == 1:
                capacitorObjects[capacitor['pressure']] = c.Capacitor(
                    capacitor['id'],
                    'Pleural',
                    0.0, #Emax
                    0.0, #Emin
                    0.0, #HC
                    0.0, #Tsys
                    compartment['id'],
                    0.0 #slopeFreq

                )
        
        # If there is a lung initialises the ventilator object
        if self.hasLung:
            ie = self.ventilatorParams['I'] + self.ventilatorParams['E']
            rc = 1 / self.rr * 60  # Heart cycle duration (s)
            tInsp = (self.ventilatorParams['I'] / ie) * rc  # Inspiration duration (seconds)
            slopeTime = (self.ventilatorParams['slopeFraction'] / ie) * rc
            pMax = self.ventilatorParams['dcComponent'] + self.ventilatorParams['amplitude'] + self.atmosphericPressure
            pMin = self.ventilatorParams['dcComponent'] + self.atmosphericPressure
            
            capacitorObjects['u'] = c.VentilatorPressure(
                'u',
                0,
                pMax, #Emax
                pMin, #Emin
                rc, #HC
                tInsp, #Tsys
                0,
                slopeTime #slopeFreq
            )

        # Initialises the resistor Objects
        for resistor in self.resistors:
            if resistor['typeR'] == 0: #Diode
                resistorObjects[resistor['current']] = c.Diode(
                    resistor['id'],
                    0.0,
                    resistor['pressureIn'],
                    [resistor['pressureOut']],
                    resistor['current'],
                    False
                        )
            elif resistor['typeR'] == 1: #Resistor
                resistorObjects[resistor['current']] = c.Resistor(
                    resistor['id'],
                    0.0,
                    resistor['pressureIn'],
                    resistor['pressureOut'],
                    resistor['current'],
                    False
                )
                
            elif resistor['typeR'] == 2: #DiodeInertial
                resistorObjects[resistor['current']] = c.Diode(
                    resistor['id'],
                    resistor['typeR_params']['L'],
                    resistor['pressureIn'],
                    resistor['pressureOut'],
                    resistor['current'],
                    True
                )
            elif resistor['typeR'] == 3: #ResistorInertial
                resistorObjects[resistor['current']] = c.Resistor(
                    resistor['id'],
                    resistor['typeR_params']['L'],
                    resistor['pressureIn'],
                    resistor['pressureOut'],
                    resistor['current'],
                    True
                )
        
        # Initialises the connection Objects
        for comp in self.compartments:
            fIn = []
            for resIn in comp['resistorsIn']:
                fIn.append(resIn['current'])
            fOut = []
            for resOut in comp['resistorsOut']:
                fOut.append(resOut['current'])
            connectionbjects[comp['id']] = c.Connections(
                fIn,
                fOut
            )

        return states, capacitorObjects, resistorObjects, connectionbjects, dVdt, capacitors, flows, resistors, pressures 
