import equinox as eqx
import jax.numpy as jnp

'''
 ██████  █████  ██████   █████   ██████ ██ ████████  ██████  ██████  ███████
██      ██   ██ ██   ██ ██   ██ ██      ██    ██    ██    ██ ██   ██ ██
██      ███████ ██████  ███████ ██      ██    ██    ██    ██ ██████  ███████
██      ██   ██ ██      ██   ██ ██      ██    ██    ██    ██ ██   ██      ██
 ██████ ██   ██ ██      ██   ██  ██████ ██    ██     ██████  ██   ██ ███████
'''

# This is the super class for the components of the model that can be considered as a capacitor
# or components that calculate the pressure in the system. all the components that inherit from this class
# must have the pressure function implemented.
# The pressure function is the function that calculates the pressure in the system.
# Eventually a method to calculate the variation of the capacity of the capacitor will be implemented here.
class Capacitor(eqx.Module):
    volIdx: str
    eMaxIdx: str # Here means just the capacity of the capacitor
    biasPIdx: str

    def __init__(self,
                 volIdx: jnp.ndarray,
                 eMaxIdx: jnp.ndarray,
                 biasPIdx: jnp.ndarray,
                 ):
        self.volIdx = volIdx
        self.eMaxIdx = eMaxIdx
        self.biasPIdx = biasPIdx

    def pressure(self,
                 t: jnp.ndarray,
                 y:dict[str,float],
                 ) -> jnp.ndarray:
        pressure = (y[self.volIdx] + (y[self.eMaxIdx] * y[self.biasPIdx])) / y[self.eMaxIdx]
        return pressure


class CapacitorDCVolume(Capacitor):
    volIdx: str
    eMaxIdx: str # Here means just the capacity of the capacitor
    biasPIdx: str
    unstressedVolume: float = 0.0

    def __init__(self,
                 volIdx: jnp.ndarray,
                 eMaxIdx: jnp.ndarray,
                 biasPIdx: jnp.ndarray,
                 unstressedVolume: jnp.ndarray = 0.0):
        self.volIdx = volIdx
        self.eMaxIdx = eMaxIdx
        self.biasPIdx = biasPIdx
        self.unstressedVolume = unstressedVolume

    def pressure(self,
                 t: jnp.ndarray,
                 y:dict[str,float]) -> jnp.ndarray:
        pressure = ((y[self.volIdx] - self.unstressedVolume) + (y[self.eMaxIdx] * y[self.biasPIdx])) / y[self.eMaxIdx]
        return pressure

#TODO change to absolute pressure output, not variation of pressure
class CapacitorPleura(Capacitor):
    volIdx: str
    eMaxIdx: str # Here means just the capacity of the capacitor
    biasPIdx: str
    children : list
    unstressedVolume: float = 0.0
    pressureName:str
    step:float = 1e-3

    def __init__(self,
                 volIdx: jnp.ndarray,
                 eMaxIdx: jnp.ndarray,
                 biasPIdx: jnp.ndarray,
                 children : list,
                 pressureName:str,
                 unstressedVolume: float = 0.0,
                 step:float = 1e-3,
                 ):
        self.volIdx = volIdx
        self.eMaxIdx = eMaxIdx
        self.biasPIdx = biasPIdx
        self.children = children
        self.unstressedVolume = unstressedVolume
        self.pressureName = pressureName
        self.step = step

    def pressure(self,
                 t: jnp.ndarray,
                 y:dict[str,float]) -> jnp.ndarray:

        volumeChildren = 0.0
        for child in self.children:
            volumeChildren += y[child]

        volumeCavity = volumeChildren - self.unstressedVolume
        #volumeCavity = volumeChildren

        #pressure = (volumeCavity) / y[self.eMaxIdx]
        pressure = (volumeCavity + (y[self.eMaxIdx] * y[self.biasPIdx])) / y[self.eMaxIdx]
        return (pressure - y[self.pressureName])/self.step

#TODO change to absolute pressure output, not variation of pressure
class CapacitorThorax(Capacitor):
    volIdx: str
    eMaxIdx: str # Here means just the capacity of the capacitor
    biasPIdx: str
    children : list
    unstressedVolume: float = 0.0

    def __init__(self,
                 volIdx: jnp.ndarray,
                 eMaxIdx: jnp.ndarray,
                 biasPIdx: jnp.ndarray,
                 children : list,
                 unstressedVolume: float = 0.0
                 ):
        self.volIdx = volIdx
        self.eMaxIdx = eMaxIdx
        self.biasPIdx = biasPIdx
        self.children = children
        self.unstressedVolume = unstressedVolume

    def pressure(self,
                 t: jnp.ndarray,
                 y:dict[str,float]) -> jnp.ndarray:

        volumeChildren = 0.0
        for child in self.children:
            volumeChildren += y[child]

        volumeCavity = volumeChildren - self.unstressedVolume
        #volumeCavity = volumeChildren

        #pressure = (volumeCavity) / y[self.eMaxIdx]
        pressure = (volumeCavity + (y[self.eMaxIdx] * y[self.biasPIdx])) / y[self.eMaxIdx]
        return pressure

#TODO change to absolute pressure output, not variation of pressure
class CapacitorSelfBreathingThorax(Capacitor):
    volIdx: str
    eMaxIdx: str
    eMinIdx: str
    biasPIdx: str
    children : list
    unstressedVolume: float = 0.0
    cycle:str
    step:float = 1e-3
    pressureName:str

    def __init__(self,
                 volIdx: jnp.ndarray,
                 eMaxIdx: jnp.ndarray,
                 eMinIdx: jnp.ndarray,
                 biasPIdx: jnp.ndarray,
                 children : list,
                 cycle:str,
                 pressureName:str,
                 unstressedVolume: float = 0.0,
                 step:float = 1e-3,

                 ):
        self.volIdx = volIdx
        self.eMaxIdx = eMaxIdx
        self.eMinIdx = eMinIdx
        self.biasPIdx = biasPIdx
        self.children = children
        self.unstressedVolume = unstressedVolume
        self.cycle = cycle
        self.step = step
        self.pressureName = pressureName


    def pressure(self,
                 t: jnp.ndarray,
                 y:dict[str,float]) -> jnp.ndarray:

        volumeChildren = 0.0
        for child in self.children:
            volumeChildren += y[child]
        volumeCavity = volumeChildren


        #tSys = 0.3 * jnp.sqrt(y['SC'])
        tSys = 0.4 * y[self.cycle]
        t0 = y['timer' + self.cycle]
        multiplier = 0.5


        def cnd1(t00):
            cond1 = 1 - jnp.cos((jnp.pi * t00) / tSys)
            cond2 = 1 + jnp.cos((1/multiplier * jnp.pi * (t00-tSys)) / tSys)
            Esin = jnp.where(t00 <= tSys, cond1, cond2)
            el = (Esin * ((y[self.eMaxIdx] - y[self.eMinIdx]) / 2)) + y[self.eMinIdx]
            return el


        uVol = jnp.where(t0 <= (multiplier*tSys + tSys), cnd1(t0), y[self.eMinIdx])

        #pressure = (y[self.volIdx] + (newCapacity * y[self.biasPIdx])) / newCapacity
        pressure = (volumeCavity - uVol + (y[self.volIdx] * y[self.biasPIdx])) / y[self.volIdx]

        return (pressure - y[self.pressureName])/self.step
'''
class CapacitorSelfBreathingThorax(Capacitor):
    volIdx: str
    eMaxIdx: str
    eMinIdx: str
    biasPIdx: str
    children : list
    unstressedVolume: float = 0.0
    cycle:str
    step:float = 1e-3
    pressureName:str

    def __init__(self,
                 volIdx: jnp.ndarray,
                 eMaxIdx: jnp.ndarray,
                 eMinIdx: jnp.ndarray,
                 biasPIdx: jnp.ndarray,
                 children : list,
                 cycle:str,
                 pressureName:str,
                 unstressedVolume: float = 0.0,
                 step:float = 1e-3,

                 ):
        self.volIdx = volIdx
        self.eMaxIdx = eMaxIdx
        self.eMinIdx = eMinIdx
        self.biasPIdx = biasPIdx
        self.children = children
        self.unstressedVolume = unstressedVolume
        self.cycle = cycle
        self.step = step
        self.pressureName = pressureName


    def pressure(self,
                 t: jnp.ndarray,
                 y:dict[str,float]) -> jnp.ndarray:

        volumeChildren = 0.0
        for child in self.children:
            volumeChildren += y[child]
        volumeCavity = volumeChildren - self.unstressedVolume


        elastance = 0.0
        #tSys = 0.3 * jnp.sqrt(y['SC'])
        tSys = 0.3 * y[self.cycle]
        t0 = y['timer' + self.cycle]


        def cnd1(t00):
            cond1 = 1 - jnp.cos((jnp.pi * t00) / tSys)
            cond2 = 1 + jnp.cos((2 * jnp.pi * (t00 - tSys)) / tSys)
            Esin = jnp.where(t00 <= tSys, cond1, cond2)
            el = (Esin * ((y[self.eMaxIdx] - y[self.eMinIdx]) / 2)) + y[self.eMinIdx]
            return el


        elastance = jnp.where(t0 <= (1.5 * tSys), cnd1(t0), y[self.eMinIdx])
        #newCapacity = 1/y[self.eMaxIdx]
        newCapacity = 1/elastance

        #pressure = (y[self.volIdx] + (newCapacity * y[self.biasPIdx])) / newCapacity
        pressure = (volumeCavity + (newCapacity * y[self.biasPIdx])) / newCapacity

        return (pressure - y[self.pressureName])/self.step
'''

# Class for the elastance capacitor. The elastance capacitor is a capacitor that has a variable capacity.
# This is used to calculate the driving functions of the heart ventricles.
class ElastanceCapacitor(Capacitor):
    volIdx: str
    biasPIdx: str
    eMaxIdx: str
    eMinIdx: str
    hcIdx: float

    #cycle:str
    #step:float = 1e-3
    #pressureName:str

    def __init__(self,
                 volIdx: jnp.ndarray,
                 biasPIdx: jnp.ndarray,
                 eMaxIdx: jnp.ndarray,
                 eMinIdx: jnp.ndarray,
                 hcIdx: jnp.ndarray,
                 ):
        self.volIdx = volIdx
        self.biasPIdx = biasPIdx
        self.eMaxIdx = eMaxIdx
        self.eMinIdx = eMinIdx
        self.hcIdx = hcIdx


    def pressure(self,
                 t: jnp.ndarray,
                 y:dict[str,float]) -> jnp.ndarray:

        elastance = 0.0
        tSys = 0.3 * jnp.sqrt(y[self.hcIdx])
        t0 = y['timer' + self.hcIdx]

        def cnd1(t00):
            cond1 = 1 - jnp.cos((jnp.pi * t00) / tSys)
            cond2 = 1 + jnp.cos((2 * jnp.pi * (t00 - tSys)) / tSys)
            Esin = jnp.where(t00 <= tSys, cond1, cond2)
            el = (Esin * ((y[self.eMaxIdx] - y[self.eMinIdx]) / 2)) + y[self.eMinIdx]
            return el

        elastance = jnp.where(t0 <= (1.5 * tSys), cnd1(t0), y[self.eMinIdx]*1.0)



        newCapacity = 1/elastance
        pressure = (y[self.volIdx] + (newCapacity * y[self.biasPIdx])) / newCapacity

        return pressure

# Class for the ventilator pressure is of the capacitor type, even though the variables names do not match with
# the capacitor class. This is because the ventilator pressure is calculated in the same way as the capacitor pressure.
class VentilatorPressure(Capacitor):
    volIdx: str
    biasPIdx: str
    eMaxIdx: str # Here it means Pmax
    eMinIdx: str # Here it means Pmin

    rcIdx: float
    ieRatio: float
    slopeFreq: float

    pressureName: str
    step:  float

    def __init__(self,
                eMaxIdx: jnp.ndarray,
                eMinIdx: jnp.ndarray,
                rcIdx: jnp.ndarray,
                ieRatio: jnp.ndarray,
                slopeFreq: jnp.ndarray,
                pressureName: jnp.ndarray,
                step:  jnp.ndarray,
                volIdx: jnp.ndarray = '',
                biasPIdx: jnp.ndarray = '',

                 ):
        self.eMaxIdx = eMaxIdx
        self.eMinIdx = eMinIdx
        self.rcIdx = rcIdx
        self.ieRatio = ieRatio
        self.slopeFreq = slopeFreq
        self.volIdx = volIdx
        self.biasPIdx = biasPIdx
        self.pressureName = pressureName
        self.step = step

    def pressure(self,
                 t: jnp.ndarray,
                 y:dict[str,float]) -> jnp.ndarray:

        tInsp = (self.ieRatio) * y[self.rcIdx]  # Inspiration duration (seconds)
        slopeTime = (self.slopeFreq) * y[self.rcIdx]

        amp = self.eMaxIdx
        slopeFreq = 1 / slopeTime

        t0 = y['timer' + self.rcIdx]

        def cnd2(t000):
            pr1 = y[self.eMinIdx] + amp * jnp.sin(0.5 * slopeFreq * jnp.pi * t0)
            pr2 = y[self.eMinIdx] + amp
            return jnp.where(t000 < slopeTime, pr1, pr2)

        def cnd1(t00):
            pr2 = y[self.eMinIdx] + amp * jnp.cos(0.5 * slopeFreq * jnp.pi * (t0 - (tInsp)))
            return jnp.where(t00 < tInsp, cnd2(t00), pr2)

        pressure = jnp.where(t0 < (tInsp + slopeTime), cnd1(t0) , y[self.eMinIdx]*1.0)
        #return pressure
        return (pressure - y[self.pressureName])/self.step

class ConstantPressure(Capacitor):
    volIdx: str
    biasPIdx: str
    eMaxIdx: str # Here it means Pmax

    def __init__(self,
                 volIdx: jnp.ndarray,
                 eMaxIdx: jnp.ndarray,
                 biasPIdx: jnp.ndarray,
                 ):
        self.volIdx = volIdx
        self.eMaxIdx = eMaxIdx
        self.biasPIdx = biasPIdx

    def pressure(self,
                 t: jnp.ndarray,
                 y:dict[str,float]) -> jnp.ndarray:

        return 0.0

class FilePressure(eqx.Module):
    name:str
    array: tuple #= eqx.field(static=True)
    step: str


    def __init__(self,
                 name: str,
                 array: tuple,
                 step: str,
                 ):
        self.name = name
        self.array = array
        self.step = step


    def pressure(self,
                 t: jnp.ndarray,
                 y:dict[str,float]) -> jnp.ndarray:
        index = (t/self.step).astype(int)
        pressure = jnp.array(self.array)[index]
        return (pressure - y[self.name])/self.step

'''
██████  ███████ ███████ ██ ███████ ████████  ██████  ██████  ███████
██   ██ ██      ██      ██ ██         ██    ██    ██ ██   ██ ██
██████  █████   ███████ ██ ███████    ██    ██    ██ ██████  ███████
██   ██ ██           ██ ██      ██    ██    ██    ██ ██   ██      ██
██   ██ ███████ ███████ ██ ███████    ██     ██████  ██   ██ ███████
'''

# This is the super class for the components of the model that can be considered as a resistor
# or components that calculate the flow in the system. all the components that inherit from this class
# must have the flow_rate and flow_rate_deriv functions implemented.
# The flow_rate function is the function that calculates the flow in the system.
# The flow_rate_deriv function is the function that calculates the derivative of the flow in the system.
# Eventually a method to calculate the variation of the resistance of the resistor will be implemented here.

############################# Sigmoid code ####################
# u = V1-V2
# expCoef = (-u+0.3)*10
# sigmoid = 1/(1 + np.exp(expCoef))
# # result = ((V1-V2)/params['R']) * sigmoid
###############################################################

class Resistor(eqx.Module):
    rIdx: str
    l: float
    pInIdx: str
    pOutIdx: str
    flowIdx: str

    inertial: bool

    def __init__(self, rIdx,l, pInIdx, pOutIdx, flowIdx, inertial=False):
        self.rIdx = rIdx
        self.l = l
        self.pInIdx = pInIdx
        self.pOutIdx = pOutIdx
        self.flowIdx = flowIdx

        self.inertial = inertial



    def flow_rate(
        self,
        t: jnp.ndarray,
        y:dict[str,float],
        p:dict[str,float]
    ) -> jnp.ndarray:
        if not self.inertial:
            q_flow = (p[self.pInIdx] - p[self.pOutIdx]) / y[self.rIdx]
        else:
            q_flow = y[self.flowIdx]
        return q_flow

    def flow_rate_deriv(
        self,
        t: jnp.ndarray,
        y:dict[str,float],
        p:dict[str,float]
    ) -> jnp.ndarray:
        if not self.inertial:
            dq_dt = 0.0
        else:
            dq_dt = (p[self.pInIdx] - p[self.pOutIdx] - y[self.flowIdx] * y[self.rIdx]) / self.l
        return dq_dt

# Class for the diode. The diode is a resistor that has a flow in only one direction.
# This is used to model the heart valves.
# The diode can have enertance or not
#
class Diode(Resistor):
    allow_reverse_flow: bool

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allow_reverse_flow = False

    def open(
        self,
        y:dict[str,float],
        p:dict[str,float]
    ) -> jnp.ndarray:
        if self.inertial:
            return jnp.logical_or(p[self.pInIdx] > p[self.pOutIdx], y[self.flowIdx] > 0.0)
        else:
            return p[self.pInIdx] > p[self.pOutIdx]

    def flow_rate(
        self,
        t: jnp.ndarray,
        y:dict[str,float],
        p:dict[str,float]
    ) -> jnp.ndarray:

        q_flow = super().flow_rate(t, y, p)
        # Regardless of inertial valve or not, ignore inertia and consider steady state
        valve_open = p[self.pInIdx] > p[self.pOutIdx]
        return jnp.where(valve_open, q_flow, 0.0)


    def flow_rate_deriv(
        self,
        t: jnp.ndarray,
        y:dict[str,float],
        p:dict[str,float]
    ) -> jnp.ndarray:

        dq_dt = super().flow_rate_deriv(t, y, p)
        valve_open = self.open(y, p)
        return jnp.where(valve_open, dq_dt, 0.0)

'''
 ██████  ██████  ███    ██ ███    ██ ███████  ██████ ████████ ██  ██████  ███    ██ ███████
██      ██    ██ ████   ██ ████   ██ ██      ██         ██    ██ ██    ██ ████   ██ ██
██      ██    ██ ██ ██  ██ ██ ██  ██ █████   ██         ██    ██ ██    ██ ██ ██  ██ ███████
██      ██    ██ ██  ██ ██ ██  ██ ██ ██      ██         ██    ██ ██    ██ ██  ██ ██      ██
 ██████  ██████  ██   ████ ██   ████ ███████  ██████    ██    ██  ██████  ██   ████ ███████
'''

# Class to hold the connections between the components of the model.
# The connections are the components that calculate the variation of the volume in the system.
# The dV function is the function that calculates the variation of the volume in the system.
class Connections(eqx.Module):
    fInIdxs: list
    fOutIdxs: list
    volIdx: str

    def __init__(self, fInIdxs: list, fOutIdxs: list, volIdx: str):
        self.fInIdxs = fInIdxs
        self.fOutIdxs = fOutIdxs
        self.volIdx = volIdx

    def dV(self,y:dict[str,float]) -> jnp.ndarray:
        volume = y[self.volIdx]
        fInSum = 0.0
        fOutSum = 0.0
        for idx in self.fInIdxs:
            fInSum += y[idx]
        for idx in self.fOutIdxs:
            fOutSum += y[idx]
        #fIn = jax.vmap(lambda idx: y[idx], self.fInIdxs)
        #fOut = jax.vmap(lambda idx: y[idx], self.fOutIdxs)

        #fInSum = jnp.sum(fIn)
        #fOutSum = jnp.sum(fOut)
        dV = fInSum - fOutSum
        #dV1 = jnp.where((volume + dV) > 0.0, fInSum - fOutSum, 0.0)



        return dV



'''
 ██████   █████  ███████     ███████ ██   ██  ██████ ██   ██  █████  ███    ██  ██████  ███████
██       ██   ██ ██          ██       ██ ██  ██      ██   ██ ██   ██ ████   ██ ██       ██
██   ███ ███████ ███████     █████     ███   ██      ███████ ███████ ██ ██  ██ ██   ███ █████
██    ██ ██   ██      ██     ██       ██ ██  ██      ██   ██ ██   ██ ██  ██ ██ ██    ██ ██
 ██████  ██   ██ ███████     ███████ ██   ██  ██████ ██   ██ ██   ██ ██   ████  ██████  ███████
'''
# TODO New generalised gas exchange method
class GasTransport(eqx.Module):
    ppName: str
    forwardFlowIn: list
    forwardFlowOut: list
    dVConf: dict[str,float]

    def __init__(self, species, gasExchangeconf,dV):
        self.ppName = species
        self.forwardFlowIn = gasExchangeconf['in']
        self.forwardFlowOut = gasExchangeconf['out']
        self.dVConf = dV

    def dP(self,y:dict[str,float],flow:dict[str,float],dV:dict[str,float]) -> jnp.ndarray:
        cPin = 0
        for flowId, posPres, negPres in zip(self.forwardFlowIn['flows'], self.forwardFlowIn['positive'] , self.forwardFlowIn['negative'] ):
            posFLow = cPin + (flow[flowId] * y[posPres])
            negFLow = cPin + (flow[flowId] * y[negPres])

            cPin = jnp.where(flow[flowId] > 0.0, posFLow, negFLow)

        cPout = 0
        for flowId, posPres, negPres in zip(self.forwardFlowOut['flows'], self.forwardFlowOut['positive'] , self.forwardFlowOut['negative'] ):
            posFLow = cPout + (flow[flowId] * y[posPres])
            negFLow = cPout + (flow[flowId] * y[negPres])

            cPout = jnp.where(flow[flowId] > 0.0, posFLow, negFLow)


        dVP = dV[self.dVConf['volume']] * y[self.dVConf['partialPressure']]

        dPdt = (cPin - cPout - dVP) / y[self.dVConf['volume']]

        return dPdt

class GasTransportTissue(GasTransport):
    def __init__(self, species,dV):
        self.ppName = species
        self.forwardFlowIn = []
        self.forwardFlowOut = []
        self.dVConf = dV

    def dP(self,y:dict[str,float],flow:dict[str,float],dV:dict[str,float]) -> jnp.ndarray:
        return 0.0

class VentilatorPartialPressure(eqx.Module):
    volIdx: str
    biasPIdx: str
    eMaxIdx: str # Here it means Pmax
    eMinIdx: str # Here it means Pmin

    rcIdx: float
    ieRatio: float
    slopeFreq: float

    pressureName: str
    step:  float

    def __init__(self,
                eMaxIdx: jnp.ndarray,
                eMinIdx: jnp.ndarray,
                rcIdx: jnp.ndarray,
                ieRatio: jnp.ndarray,
                slopeFreq: jnp.ndarray,
                pressureName: jnp.ndarray,
                step:  jnp.ndarray,
                volIdx: jnp.ndarray = '',
                biasPIdx: jnp.ndarray = '',

                 ):
        self.eMaxIdx = eMaxIdx
        self.eMinIdx = eMinIdx
        self.rcIdx = rcIdx
        self.ieRatio = ieRatio
        self.slopeFreq = slopeFreq
        self.volIdx = volIdx
        self.biasPIdx = biasPIdx

        self.pressureName = pressureName
        self.step = step

    def dP(self, y:dict[str,float],flow:dict[str,float],dV:dict[str,float]) -> jnp.ndarray:

        tInsp = (self.ieRatio) * y[self.rcIdx]  # Inspiration duration (seconds)
        slopeTime = (self.slopeFreq) * y[self.rcIdx]

        amp = self.eMaxIdx - self.eMinIdx
        slopeFreq = 1 / slopeTime

        t0 = y['timer' + self.rcIdx]

        def cnd2(t000):
            pr1 = self.eMinIdx + amp * jnp.sin(0.5 * slopeFreq * jnp.pi * t0)
            pr2 = self.eMaxIdx*1.0
            return jnp.where(t000 < slopeTime, pr1, pr2)

        def cnd1(t00):
            pr2 = self.eMinIdx + amp * jnp.cos(0.5 * slopeFreq * jnp.pi * (t0 - (tInsp)))
            return jnp.where(t00 < tInsp, cnd2(t00), pr2)

        pressure = jnp.where(t0 < (tInsp + slopeTime), cnd1(t0) , self.eMinIdx*1.0)
        return (pressure - y[self.pressureName])/self.step


'''
██████  ███████ ██████  ██  ██████  ██████  ██  ██████     ████████ ██████  ██  ██████   ██████  ███████ ██████  ███████
██   ██ ██      ██   ██ ██ ██    ██ ██   ██ ██ ██             ██    ██   ██ ██ ██       ██       ██      ██   ██ ██
██████  █████   ██████  ██ ██    ██ ██   ██ ██ ██             ██    ██████  ██ ██   ███ ██   ███ █████   ██████  ███████
██      ██      ██   ██ ██ ██    ██ ██   ██ ██ ██             ██    ██   ██ ██ ██    ██ ██    ██ ██      ██   ██      ██
██      ███████ ██   ██ ██  ██████  ██████  ██  ██████        ██    ██   ██ ██  ██████   ██████  ███████ ██   ██ ███████
'''

class PeriodicTrigger(eqx.Module):
    cyclePeriodIdx: str # Index of the variable that holds the period of the cycle Ex: 'HC', 'RC'
    triggerIdx: str # Index of the variable that holds the trigger of the cycle Ex: 'triggerHC', 'triggerRC'
    timerIdx: str # Index of the variable that holds the timer of the cycle Ex: 'timerHC', 'timerRC'

    def __init__(self, cyclePeriodIdx, triggerIdx, timerIdx):
        self.cyclePeriodIdx = cyclePeriodIdx
        self.triggerIdx = triggerIdx
        self.timerIdx = timerIdx

    ###################################################################
    def caseTrueTrigger(self,y:dict[str,float]) -> jnp.ndarray:
        #return (2.36044/1e-3) # Misterious number to get the right period when using runge kutta 4 solver (Tsit5)
        return (1/1e-3) * y[self.cyclePeriodIdx]
    def caseFalseTrigger(self,y:dict[str,float]) -> jnp.ndarray:
        return 0.0

    def trigger(self,y:dict[str,float],t:jnp.ndarray) -> jnp.ndarray:
        currentAbsoluteTime = t + y['T0']
        condition = currentAbsoluteTime - y[self.triggerIdx] > 0.00
        return jnp.where(condition, self.caseTrueTrigger(y), self.caseFalseTrigger(y))

    ###################################################################
    # Timer for the cycle
    def caseTrueTimer(self,y:dict[str,float]) -> jnp.ndarray:
        return -(1/1e-3) * y[self.timerIdx]
    def caseFalseTimer(self,y:dict[str,float]) -> jnp.ndarray:
        return 1.0

    def timer(self,y:dict[str,float],t:jnp.ndarray) -> jnp.ndarray:
        currentAbsoluteTime = t + y['T0']
        condition = currentAbsoluteTime - y[self.triggerIdx] >= -0.001
        return jnp.where(condition, self.caseTrueTimer(y), self.caseFalseTimer(y))

'''
██ ███    ██ ████████ ███████  ██████  ██████   █████  ████████  ██████  ██████  ███████     ██  █████  ██    ██ ███████ ██████   ██████  ███████ ███████
██ ████   ██    ██    ██      ██       ██   ██ ██   ██    ██    ██    ██ ██   ██ ██         ██  ██   ██ ██    ██ ██      ██   ██ ██       ██      ██
██ ██ ██  ██    ██    █████   ██   ███ ██████  ███████    ██    ██    ██ ██████  ███████   ██   ███████ ██    ██ █████   ██████  ██   ███ █████   ███████
██ ██  ██ ██    ██    ██      ██    ██ ██   ██ ██   ██    ██    ██    ██ ██   ██      ██  ██    ██   ██  ██  ██  ██      ██   ██ ██    ██ ██           ██
██ ██   ████    ██    ███████  ██████  ██   ██ ██   ██    ██     ██████  ██   ██ ███████ ██     ██   ██   ████   ███████ ██   ██  ██████  ███████ ███████
'''

# TODO step is still a literal value, change to a variable
class CycleIntegrator(eqx.Module):
    cyclePeriodIdx: str # Index of the variable that holds the period of the cycle Ex: 'HC', 'RC'
    triggerIdx: str # Index of the variable that holds the trigger of the cycle Ex: 'triggerHC', 'triggerRC'
    timerIdx: str # Index of the variable that holds the timer of the cycle Ex: 'timerHC', 'timerRC'
    processedIdx: str # Index of the variable that holds the integrated value of the cycle Ex: 'Hl', 'Rl'
    varToProcessIdx: str = 'Hl' # Index of the variable to integrate
    step: float = 1e-3

    def __init__(self, cyclePeriodIdx, processedIdx='i_Hl', varToProcessIdx='Hl', step=1e-3):
        self.cyclePeriodIdx = cyclePeriodIdx
        self.triggerIdx = 'trigger' + cyclePeriodIdx
        self.timerIdx = 'timer' + cyclePeriodIdx
        self.processedIdx = processedIdx
        self.varToProcessIdx = varToProcessIdx
        self.step = step

    ###################################################################
    def caseTrueIntegrator(self,y:dict[str,float]) -> jnp.ndarray:
        return -(1/self.step) * y[self.processedIdx]

    def caseFalseIntegrator(self,y:dict[str,float]) -> jnp.ndarray:
        return y[self.varToProcessIdx] #* 1e-3

    def process(self,y:dict[str,float],t:jnp.ndarray) -> jnp.ndarray:
        currentAbsoluteTime = t + y['T0']
        condition = currentAbsoluteTime - y[self.triggerIdx] >= 0.0
        return jnp.where(condition, self.caseTrueIntegrator(y), self.caseFalseIntegrator(y))

# TODO step is still a literal value, change to a variable
class CycleMax(CycleIntegrator):
    cyclePeriodIdx: str # Index of the variable that holds the period of the cycle Ex: 'HC', 'RC'
    triggerIdx: str # Index of the variable that holds the trigger of the cycle Ex: 'triggerHC', 'triggerRC'
    timerIdx: str # Index of the variable that holds the timer of the cycle Ex: 'timerHC', 'timerRC'
    processedIdx: str # Index of the variable that holds the integrated value of the cycle Ex: 'Hl', 'Rl'
    varToProcessIdx: str = 'Hl' # Index of the variable to integrate
    step: float = 1e-3

    def __init__(self, cyclePeriodIdx, processedIdx='i_Hl', varToProcessIdx='Hl', step=1e-3):
        self.cyclePeriodIdx = cyclePeriodIdx
        self.triggerIdx = 'trigger' + cyclePeriodIdx
        self.timerIdx = 'timer' + cyclePeriodIdx
        self.processedIdx = processedIdx
        self.varToProcessIdx = varToProcessIdx
        self.step = step

    ###################################################################
    def caseTrue(self,y:dict[str,float]) -> jnp.ndarray:
        return (1/self.step) * (-y[self.processedIdx])

    def caseMax(self,y:dict[str,float]) -> jnp.ndarray:
        return (1/self.step) * (-y[self.processedIdx] + y[self.varToProcessIdx])

    def caseFalse(self,y:dict[str,float]) -> jnp.ndarray:
        condition = y[self.varToProcessIdx] > y[self.processedIdx]
        return jnp.where(condition, self.caseMax(y), 0.0)

    def process(self,y:dict[str,float],t:jnp.ndarray) -> jnp.ndarray:
        currentAbsoluteTime = t + y['T0']
        condition = currentAbsoluteTime - y[self.triggerIdx] >= 0.00#-self.step
        return jnp.where(condition, self.caseTrue(y), self.caseFalse(y))

# TODO step is still a literal value, change to a variable
class CycleMin(CycleIntegrator):
    cyclePeriodIdx: str # Index of the variable that holds the period of the cycle Ex: 'HC', 'RC'
    triggerIdx: str # Index of the variable that holds the trigger of the cycle Ex: 'triggerHC', 'triggerRC'
    timerIdx: str # Index of the variable that holds the timer of the cycle Ex: 'timerHC', 'timerRC'
    processedIdx: str # Index of the variable that holds the integrated value of the cycle Ex: 'Hl', 'Rl'
    varToProcessIdx: str = 'Hl' # Index of the variable to integrate
    step: float = 1e-3

    def __init__(self, cyclePeriodIdx, processedIdx='i_Hl', varToProcessIdx='Hl', step=1e-3):
        self.cyclePeriodIdx = cyclePeriodIdx
        self.triggerIdx = 'trigger' + cyclePeriodIdx
        self.timerIdx = 'timer' + cyclePeriodIdx
        self.processedIdx = processedIdx
        self.varToProcessIdx = varToProcessIdx
        self.step = step

    ###################################################################
    def caseTrue(self,y:dict[str,float]) -> jnp.ndarray:
        return (1/self.step) * (-y[self.processedIdx] + y[self.varToProcessIdx])

    def caseMin(self,y:dict[str,float]) -> jnp.ndarray:
        return (1/self.step) * (-y[self.processedIdx] + y[self.varToProcessIdx])

    def caseFalse(self,y:dict[str,float]) -> jnp.ndarray:
        condition = y[self.varToProcessIdx] < y[self.processedIdx]
        return jnp.where(condition, self.caseMin(y), 0.0)

    def process(self,y:dict[str,float],t:jnp.ndarray) -> jnp.ndarray:
        currentAbsoluteTime = t + y['T0']
        condition = currentAbsoluteTime - y[self.triggerIdx] >= 0.00#-self.step
        return jnp.where(condition, self.caseTrue(y), self.caseFalse(y))

# TODO step is still a literal value, change to a variable
class CycleKeeper(CycleIntegrator):
    cyclePeriodIdx: str # Index of the variable that holds the period of the cycle Ex: 'HC', 'RC'
    triggerIdx: str # Index of the variable that holds the trigger of the cycle Ex: 'triggerHC', 'triggerRC'
    timerIdx: str # Index of the variable that holds the timer of the cycle Ex: 'timerHC', 'timerRC'
    processedIdx: str # Index of the variable that holds the integrated value of the cycle Ex: 'Hl', 'Rl'
    varToProcessIdx: str = 'Hl' # Index of the variable to integrate
    step: float = 1e-3

    def __init__(self, cyclePeriodIdx, processedIdx='i_Hl', varToProcessIdx='Hl', step=1e-3):
        self.cyclePeriodIdx = cyclePeriodIdx
        self.triggerIdx = 'trigger' + cyclePeriodIdx
        self.timerIdx = 'timer' + cyclePeriodIdx
        self.processedIdx = processedIdx
        self.varToProcessIdx = varToProcessIdx
        self.step = step

    ###################################################################
    def caseTrueKeeper(self,y:dict[str,float]) -> jnp.ndarray:
        return (1/self.step) * (-y[self.processedIdx] + y[self.varToProcessIdx])

    def caseFalseKeeper(self,y:dict[str,float]) -> jnp.ndarray:
        return 0.0

    def process(self,y:dict[str,float],t:jnp.ndarray) -> jnp.ndarray:
        currentAbsoluteTime = t + y['T0']
        condition = currentAbsoluteTime - y[self.triggerIdx] >= -self.step
        return jnp.where(condition, self.caseTrueKeeper(y), self.caseFalseKeeper(y))

# TODO step is still a literal value, change to a variable
class CycleAverage(eqx.Module):
    cyclePeriodIdx: str # Index of the variable that holds the period of the cycle Ex: 'HC', 'RC'
    processedIdx: str # Index of the variable that holds the integrated value of the cycle Ex: 'Hl', 'Rl'
    varToProcessIdx: str = 'Hl' # Index of the variable to integrate
    step: float = 1e-3
    period: float = 3.0

    def __init__(self, cyclePeriodIdx, processedIdx, varToProcessIdx, period, step=1e-3):
        self.cyclePeriodIdx = cyclePeriodIdx
        self.processedIdx = processedIdx
        self.varToProcessIdx = varToProcessIdx
        self.step = step
        self.period = period

    ###################################################################
    # Average for the cycle
    def caseTrueAverage(self,y:dict[str,float], pressure:dict[str,float]) -> jnp.ndarray:
        return -(1/self.step) * y[self.processedIdx]

    def caseFalseAverage(self,y:dict[str,float], pressure:dict[str,float]) -> jnp.ndarray:
        return (pressure[self.varToProcessIdx] - y['AtmosphericPressure']) / (y[self.cyclePeriodIdx])

    def process(self, y:dict[str,float], pressure:dict[str,float], t:jnp.ndarray) -> jnp.ndarray:
        #currentAbsoluteTime = t + y['T0']
        #condition = currentAbsoluteTime - y[self.triggerIdx] >= 0.0
        #return 0.0
        #return jnp.where(condition, self.caseTrueAverage(y,pressure), self.caseFalseAverage(y,pressure))
        return ((pressure[self.varToProcessIdx] - y['AtmosphericPressure'])/self.period) - (y[self.processedIdx]/self.period)

class PeriodAverage(eqx.Module):
    processedIdx: str # Index of the variable that holds the integrated value of the cycle Ex: 'Hl', 'Rl'
    varToProcessIdx: str # Index of the variable to integrate
    period: float = 10
    step: float = 1e-3

    def __init__(self, processedIdx, varToProcessIdx, period=10, step=1e-3):
        self.processedIdx = processedIdx
        self.varToProcessIdx = varToProcessIdx
        self.step = step
        self.period = period

    def process(self, y:dict[str,float], t:jnp.ndarray) -> jnp.ndarray:
        #currentAbsoluteTime = t + y['T0']
        #condition = currentAbsoluteTime - y[self.triggerIdx] >= 0.0
        #return 0.0
        #return jnp.where(condition, self.caseTrueAverage(y,pressure), self.caseFalseAverage(y,pressure))
        return ((y[self.varToProcessIdx])/self.period) - (y[self.processedIdx]/self.period)

class PressureAverage(eqx.Module):
    processedIdx: str # Index of the variable that holds the integrated value of the cycle Ex: 'Hl', 'Rl'
    varToProcessIdx: str # Index of the variable to integrate
    atmPressureIdx: str
    period: float = 10
    step: float = 1e-3

    def __init__(self, processedIdx, varToProcessIdx, atmPressure, period=10, step=1e-3):
        self.processedIdx = processedIdx
        self.varToProcessIdx = varToProcessIdx
        self.atmPressureIdx = atmPressure
        self.step = step
        self.period = period

    def process(self, y:dict[str,float], t:jnp.ndarray) -> jnp.ndarray:
        #currentAbsoluteTime = t + y['T0']
        #condition = currentAbsoluteTime - y[self.triggerIdx] >= 0.0
        #return 0.0
        #return jnp.where(condition, self.caseTrueAverage(y,pressure), self.caseFalseAverage(y,pressure))
        return ((y[self.varToProcessIdx] - y[self.atmPressureIdx])/self.period) - (y[self.processedIdx]/self.period)

class CycleAverageGroup(CycleAverage):
    cyclePeriodIdx: str # Index of the variable that holds the period of the cycle Ex: 'HC', 'RC'
    processedIdx: str # Index of the variable that holds the integrated value of the cycle Ex: 'Hl', 'Rl'
    varToProcessIdx: list # Index of the variable to integrate

    def __init__(self, cyclePeriodIdx, processedIdx, varToProcessIdx):
        self.cyclePeriodIdx = cyclePeriodIdx
        self.processedIdx = processedIdx
        self.varToProcessIdx = varToProcessIdx

    def process(self, y:dict[str,float], pressure:dict[str,float], t:jnp.ndarray) -> jnp.ndarray:
        period = 3
        meanGroup = 0.0
        for key in self.varToProcessIdx:
            meanGroup += y[key]
        meanGroup = meanGroup / len(self.varToProcessIdx)
        newAverageIn = meanGroup/period
        oldAverageOut = y[self.processedIdx]/period
        return newAverageIn - oldAverageOut

'''
 ██████ ██    ██  ██████ ██      ███████     ████████ ██    ██ ██████  ███████ ███████
██       ██  ██  ██      ██      ██             ██     ██  ██  ██   ██ ██      ██
██        ████   ██      ██      █████          ██      ████   ██████  █████   ███████
██         ██    ██      ██      ██             ██       ██    ██      ██           ██
 ██████    ██     ██████ ███████ ███████        ██       ██    ██      ███████ ███████
'''

# Constant cycle rate
class Cycle(eqx.Module):
    cyclePeriodIdx: str # Index of the variable that holds the period of the cycle Ex: 'HC', 'RC'
    def __init__(self, cyclePeriodIdx):
        self.cyclePeriodIdx = cyclePeriodIdx

    def dCycle(self,y:dict[str,float],t:jnp.ndarray) -> jnp.ndarray:
        return 0.0

class CycleRamp(Cycle):
    cyclePeriodIdx: str # Index of the variable that holds the period of the cycle Ex: 'HC', 'RC'
    rate: float # Rate of the cycle change Ex: 0.1 for a change of 0.1 Hz
    def __init__(self, cyclePeriodIdx, rate=0.0):
        self.cyclePeriodIdx = cyclePeriodIdx
        self.rate = rate

    def dCycle(self,y:dict[str,float],t:jnp.ndarray) -> jnp.ndarray:
        return self.rate

class CyclePeriodic(Cycle):
    cyclePeriodIdx: str # Index of the variable that holds the period of the cycle Ex: 'HC', 'RC'
    rate: float # Rate of the cycle change Ex: 0.1 for a change of 0.1 Hz
    changePeriod: float # Frequency of the cycle change Ex: 0.1 for a change of 0.1 Hz
    def __init__(self, cyclePeriodIdx, rate=0.0, changePeriod=0.0):
        self.cyclePeriodIdx = cyclePeriodIdx
        self.rate = rate
        self.changePeriod = changePeriod

    def dCycle(self,y:dict[str,float],t:jnp.ndarray) -> jnp.ndarray:
        currentAbsoluteTime = t + y['T0']
        return self.rate*jnp.sin(2*jnp.pi*currentAbsoluteTime/self.changePeriod)

class CycleControlled(Cycle):
    cyclePeriodIdx: str # Index of the variable that holds the period of the cycle Ex: 'HC', 'RC'

    varTargetIdx: str # Ex: 'V_O2_Cs', 'V_O2_Cp'

    targetValue: str # Target value of the controller Ex: PpO2 = 100 mmHg
    minValueToControl: float # Minimum value of the variable to control Ex: R_Cs = 0.1 mmHg/(ml/min)
    maxValueToControl: float # Maximum value of the variable to control Ex: R_Cs = 0.1 mmHg/(ml/min)
    proportionalConstant: float # Slope of the controller Ex: 0.1 mmHg/(ml/min) / mmHg
    derivativeConstant: float # Slope of the controller Ex: 0.1 mmHg/(ml/min) / mmHg


    def __init__(self, cyclePeriodIdx, varTargetIdx, targetValue, minValueToControl, maxValueToControl, proportionalConstant=0.0, derivativeConstant=0.0):
        self.cyclePeriodIdx = cyclePeriodIdx

        self.varTargetIdx = varTargetIdx
        self.targetValue = targetValue
        self.minValueToControl = minValueToControl
        self.maxValueToControl = maxValueToControl
        self.proportionalConstant = proportionalConstant
        self.derivativeConstant = derivativeConstant

    def dCycle(self,y:dict[str,float],t:jnp.ndarray) -> jnp.ndarray:

        varToControl = y[self.cyclePeriodIdx]
        varTarget = y[self.varTargetIdx]

        #targetValue = y[self.targetValue]
        targetValue = self.targetValue

        def belowMin():
            return 0.0

        def aboveMax():
            return 0.0

        def betweenMinAndMax():
            diff = targetValue < varTarget
            meanError = jnp.abs(((targetValue - varTarget) / varTarget) * self.derivativeConstant)
            return jnp.where(diff, self.proportionalConstant * varToControl * meanError, -self.proportionalConstant * varToControl * meanError)

        return jnp.where(varToControl < self.minValueToControl, belowMin(), jnp.where(varToControl > self.maxValueToControl, aboveMax(), betweenMinAndMax()))


'''
 ██████  ██████  ███    ██ ████████ ██████   ██████  ██      ██      ███████ ██████  ███████
██      ██    ██ ████   ██    ██    ██   ██ ██    ██ ██      ██      ██      ██   ██ ██
██      ██    ██ ██ ██  ██    ██    ██████  ██    ██ ██      ██      █████   ██████  ███████
██      ██    ██ ██  ██ ██    ██    ██   ██ ██    ██ ██      ██      ██      ██   ██      ██
 ██████  ██████  ██   ████    ██    ██   ██  ██████  ███████ ███████ ███████ ██   ██ ███████
'''


class NoController(eqx.Module):
    # Index of the variable that holds the target value of the controller
    #varTargetIdx: str # Ex: 'V_O2_Cs', 'V_O2_Cp'
    # Index of the variable that holds the value to control
    varToControlIdx: str #Ex: 'R_Cs', 'R_Cp'

    #targetValue: float # Target value of the controller Ex: PpO2 = 100 mmHg
    #minValueToControl: float # Minimum value of the variable to control Ex: R_Cs = 0.1 mmHg/(ml/min)
    #maxValueToControl: float # Maximum value of the variable to control Ex: R_Cs = 0.1 mmHg/(ml/min)
    #slope: float # Slope of the controller Ex: 0.1 mmHg/(ml/min) / mmHg

    def __init__(self, varToControlIdx):
        self.varToControlIdx = varToControlIdx


    def dControl(self,y:dict[str,float]) -> jnp.ndarray:

        varToControl = y[self.varToControlIdx]

        return 0.0

class RampController(eqx.Module):
    varToControlIdx: str
    rate: float # Rate of the cycle change Ex: 0.1 for a change of 0.1 Hz

    def __init__(self, varToControlIdx, rate=0.0):
        self.varToControlIdx = varToControlIdx
        self.rate = rate


    def dControl(self,y:dict[str,float]) -> jnp.ndarray:
        return self.rate #* y[self.varToControlIdx]

class LocalController(eqx.Module):
    # Index of the variable that holds the target value of the controller
    varTargetIdx: str # Ex: 'V_O2_Cs', 'V_O2_Cp'
    # Index of the variable that holds the value to control
    varToControlIdx: str #Ex: 'R_Cs', 'R_Cp'

    targetValue: float # Target value of the controller Ex: PpO2 = 100 mmHg
    minValueToControl: float # Minimum value of the variable to control Ex: R_Cs = 0.1 mmHg/(ml/min)
    maxValueToControl: float # Maximum value of the variable to control Ex: R_Cs = 0.1 mmHg/(ml/min)
    proportionalConstant: float # Slope of the controller Ex: 0.1 mmHg/(ml/min) / mmHg
    derivativeConstant: float # Slope of the controller Ex: 0.1 mmHg/(ml/min) / mmHg

    def __init__(self, varTargetIdx, varToControlIdx, targetValue, minValueToControl, maxValueToControl, proportionalConstant=0.0, derivativeConstant=0.0):
        self.varTargetIdx = varTargetIdx
        self.varToControlIdx = varToControlIdx
        self.targetValue = targetValue
        self.minValueToControl = minValueToControl
        self.maxValueToControl = maxValueToControl
        self.proportionalConstant = proportionalConstant
        self.derivativeConstant = derivativeConstant

    def dControl(self,y:dict[str,float]) -> jnp.ndarray:

        varToControl = y[self.varToControlIdx]
        varTarget = y[self.varTargetIdx]

        def belowMin():
            return self.proportionalConstant * varToControl

        def aboveMax():
            return - self.proportionalConstant * varToControl

        def betweenMinAndMax():
            diff = self.targetValue - varTarget
            meanError = jnp.abs((diff / varTarget) * self.derivativeConstant)
            return jnp.where(diff < 0.0 , -self.proportionalConstant * varToControl * meanError, self.proportionalConstant * varToControl * meanError)

        return jnp.where(varToControl < self.minValueToControl, belowMin(), jnp.where(varToControl > self.maxValueToControl, aboveMax(), betweenMinAndMax()))

class LocalControllerAnti(eqx.Module):
    varToControlIdx: str

    # Index of the variable that holds the target value of the controller
    varTargetIdx: str # Ex: 'V_O2_Cs', 'V_O2_Cp'
    # Index of the variable that holds the value to control

    targetValue: str # Target value of the controller Ex: PpO2 = 100 mmHg
    minValueToControl: float # Minimum value of the variable to control Ex: R_Cs = 0.1 mmHg/(ml/min)
    maxValueToControl: float # Maximum value of the variable to control Ex: R_Cs = 0.1 mmHg/(ml/min)
    proportionalConstant: float # Slope of the controller Ex: 0.1 mmHg/(ml/min) / mmHg
    derivativeConstant: float # Slope of the controller Ex: 0.1 mmHg/(ml/min) / mmHg

    def __init__(self, varTargetIdx, varToControlIdx, targetValue, minValueToControl, maxValueToControl, proportionalConstant=0.0, derivativeConstant=0.0):
        self.varTargetIdx = varTargetIdx
        self.varToControlIdx = varToControlIdx
        self.targetValue = targetValue
        self.minValueToControl = minValueToControl
        self.maxValueToControl = maxValueToControl
        self.proportionalConstant = proportionalConstant
        self.derivativeConstant = derivativeConstant


    def dControl(self, y:dict[str,float]) -> jnp.ndarray:

        varToControl = y[self.varToControlIdx]
        varTarget = y[self.varTargetIdx]

        def belowMin():
            return self.proportionalConstant * varToControl

        def aboveMax():
            return - self.proportionalConstant * varToControl

        def betweenMinAndMax():
            diff = self.targetValue - varTarget
            meanError = jnp.abs(((diff) / varTarget) * self.derivativeConstant)

            caseTrue = +self.proportionalConstant * varToControl * meanError
            caseFalse = -self.proportionalConstant * varToControl * meanError
            return jnp.where(diff < 0.0, caseTrue, caseFalse)

        return jnp.where(varToControl < self.minValueToControl, belowMin(), jnp.where(varToControl > self.maxValueToControl, aboveMax(), betweenMinAndMax()))

class SigmoidController(eqx.Module):
    # Index of the variable that holds the target value of the controller
    varTargetIdx: str # Ex: 'V_O2_Cs', 'V_O2_Cp'
    # Index of the variable that holds the value to control
    varToControlIdx: str #Ex: 'R_Cs', 'R_Cp'

    targetValue: float # Target value of the controller Ex: PpO2 = 100 mmHg
    minValueToControl: float # Minimum value of the variable to control Ex: R_Cs = 0.1 mmHg/(ml/min)
    maxValueToControl: float # Maximum value of the variable to control Ex: R_Cs = 0.1 mmHg/(ml/min)
    proportionalConstant: float # Slope of the controller Ex: 0.1 mmHg/(ml/min) / mmHg
    derivativeConstant: float # Slope of the controller Ex: 0.1 mmHg/(ml/min) / mmHg
    step: float = 1e-3

    def __init__(self, varTargetIdx, varToControlIdx, targetValue, minValueToControl, maxValueToControl, proportionalConstant=0.0, derivativeConstant=0.0, step=1e-3):
        self.varTargetIdx = varTargetIdx
        self.varToControlIdx = varToControlIdx
        self.targetValue = targetValue
        self.minValueToControl = minValueToControl
        self.maxValueToControl = maxValueToControl
        self.proportionalConstant = proportionalConstant
        self.derivativeConstant = derivativeConstant
        self.step = step

    def dControl(self,y:dict[str,float]) -> jnp.ndarray:


        '''
        # for the Alveoli, La:
        varTargetIdx         -> V_La
        varToControlIdx      -> C_La
        targetValue          -> 1500
        minValueToControl    -> 75
        maxValueToControl    -> 200
        proportionalConstant -> 0.004
        derivativeConstant   -> ?
        '''

        amplitude = self.maxValueToControl - self.minValueToControl
        offset = self.minValueToControl
        inflectionPoint = self.targetValue
        slope = self.proportionalConstant

        sigmoid = amplitude / (1 + jnp.exp(-slope * (y[self.varTargetIdx] - inflectionPoint))) + offset
        return (-y[self.varToControlIdx] + (1/sigmoid))/self.step

class SigmoidControllerResistor(eqx.Module):
    # Index of the variable that holds the target value of the controller
    varTargetIdx: str # Ex: 'V_O2_Cs', 'V_O2_Cp'
    # Index of the variable that holds the value to control 
    varToControlIdx: str #Ex: 'R_Cs', 'R_Cp'
    
    targetValue: float # Target value of the controller Ex: PpO2 = 100 mmHg
    minValueToControl: float # Minimum value of the variable to control Ex: R_Cs = 0.1 mmHg/(ml/min)
    maxValueToControl: float # Maximum value of the variable to control Ex: R_Cs = 0.1 mmHg/(ml/min)
    proportionalConstant: float # Slope of the controller Ex: 0.1 mmHg/(ml/min) / mmHg
    derivativeConstant: float # Slope of the controller Ex: 0.1 mmHg/(ml/min) / mmHg
    step: float = 1e-3

    def __init__(self, varTargetIdx, varToControlIdx, targetValue, minValueToControl, maxValueToControl, proportionalConstant=0.0, derivativeConstant=0.0, step=1e-3):
        self.varTargetIdx = varTargetIdx
        self.varToControlIdx = varToControlIdx
        self.targetValue = targetValue
        self.minValueToControl = minValueToControl
        self.maxValueToControl = maxValueToControl
        self.proportionalConstant = proportionalConstant
        self.derivativeConstant = derivativeConstant
        self.step = step
    
    def dControl(self,y:dict[str,float]) -> jnp.ndarray:


        '''
        # for the Alveoli, La:
        varTargetIdx         -> V_La
        varToControlIdx      -> C_La
        targetValue          -> 1500
        minValueToControl    -> 75
        maxValueToControl    -> 200
        proportionalConstant -> 0.004
        derivativeConstant   -> ?
        '''

        amplitude = self.maxValueToControl - self.minValueToControl
        offset = self.minValueToControl
        inflectionPoint = self.targetValue
        slope = self.proportionalConstant

        sigmoid = amplitude / (1 + jnp.exp(-slope * (y[self.varTargetIdx] - inflectionPoint))) + offset
        return (-y[self.varToControlIdx] + (sigmoid))/self.step

class SigmoidControllerCompliance(eqx.Module):
    # Index of the variable that holds the target value of the controller
    varTargetIdx: str # Ex: 'V_O2_Cs', 'V_O2_Cp'
    # Index of the variable that holds the value to control
    varToControlIdx: str #Ex: 'R_Cs', 'R_Cp'

    targetValue: float # Target value of the controller Ex: PpO2 = 100 mmHg
    minValueToControl: float # Minimum value of the variable to control Ex: R_Cs = 0.1 mmHg/(ml/min)
    maxValueToControl: float # Maximum value of the variable to control Ex: R_Cs = 0.1 mmHg/(ml/min)
    proportionalConstant: float # Slope of the controller Ex: 0.1 mmHg/(ml/min) / mmHg
    derivativeConstant: float # Slope of the controller Ex: 0.1 mmHg/(ml/min) / mmHg
    step: float = 1e-3

    def __init__(self, varTargetIdx, varToControlIdx, targetValue, minValueToControl, maxValueToControl, proportionalConstant=0.0, derivativeConstant=0.0, step=1e-3):
        self.varTargetIdx = varTargetIdx
        self.varToControlIdx = varToControlIdx
        self.targetValue = targetValue
        self.minValueToControl = minValueToControl
        self.maxValueToControl = maxValueToControl
        self.proportionalConstant = proportionalConstant
        self.derivativeConstant = derivativeConstant
        self.step = step

    def dControl(self,y:dict[str,float]) -> jnp.ndarray:


        '''
        # for the Alveoli, La:
        varTargetIdx         -> V_La
        varToControlIdx      -> C_La
        targetValue          -> 1500
        minValueToControl    -> 75
        maxValueToControl    -> 200
        proportionalConstant -> 0.004
        derivativeConstant   -> ?
        '''

        amplitude = self.maxValueToControl - self.minValueToControl
        offset = self.minValueToControl
        inflectionPoint = self.targetValue
        slope = self.proportionalConstant
        dist = self.derivativeConstant

        flow = y['Q_' + self.varToControlIdx[2:]]

        sigmoid_in = amplitude / (1 + jnp.exp(-slope * (y[self.varTargetIdx] - inflectionPoint))) + offset
        sigmoid_out = amplitude / (1 + jnp.exp(-slope * (y[self.varTargetIdx] - (inflectionPoint-dist)))) + offset

        result = jnp.where(flow > 0.0 , sigmoid_in, sigmoid_out)


        return (-y[self.varToControlIdx] + (result))/self.step

class DoubleSigmoidController(eqx.Module):
    # Index of the variable that holds the target value of the controller
    varTargetIdx: str # Ex: 'V_O2_Cs', 'V_O2_Cp'
    # Index of the variable that holds the value to control
    varToControlIdx: str #Ex: 'R_Cs', 'R_Cp'

    inflectionPoint: float # Target value of the controller Ex: PpO2 = 100 mmHg
    maxCompliance: float # Minimum value of the variable to control Ex: R_Cs = 0.1 mmHg/(ml/min)

    separation: float # Maximum value of the variable to control Ex: R_Cs = 0.1 mmHg/(ml/min)
    slope: float # Slope of the controller Ex: 0.1 mmHg/(ml/min) / mmHg
    cMin: float # Slope of the controller Ex: 0.1 mmHg/(ml/min) / mmHg

    separation: float
    step: float = 1e-3

    def __init__(self, varTargetIdx, varToControlIdx, inflectionPoint, maxCompliance, separation, slope=0.3, cMin=0.5, step=1e-3):
        self.varTargetIdx = varTargetIdx
        self.varToControlIdx = varToControlIdx

        self.inflectionPoint = inflectionPoint
        self.maxCompliance = maxCompliance
        self.separation = separation
        self.slope = slope
        self.cMin = cMin
        self.step = step

    def dControl(self,y:dict[str,float]) -> jnp.ndarray:

        offset = (self.maxCompliance) + self.cMin
        amplitude = self.cMin - self.maxCompliance

        intersection = self.inflectionPoint + self.separation/2



        sigmoid_open = amplitude / (1 + jnp.exp(self.slope * (y[self.varTargetIdx] - self.inflectionPoint))) + offset
        sigmoid_strech = amplitude / (1 + jnp.exp(-self.slope * (y[self.varTargetIdx] - (self.inflectionPoint + self.separation)))) + offset

        result = jnp.where(y[self.varTargetIdx] < intersection, sigmoid_open, sigmoid_strech)

        #result = jnp.where(y[self.varTargetIdx] <= 1000, 48.99615327695676 , result1)

        return (-y[self.varToControlIdx] + result)/self.step

'''
███████ ██       █████  ███████ ████████  █████  ███    ██  ██████ ███████ ███████
██      ██      ██   ██ ██         ██    ██   ██ ████   ██ ██      ██      ██
█████   ██      ███████ ███████    ██    ███████ ██ ██  ██ ██      █████   ███████
██      ██      ██   ██      ██    ██    ██   ██ ██  ██ ██ ██      ██           ██
███████ ███████ ██   ██ ███████    ██    ██   ██ ██   ████  ██████ ███████ ███████
'''














'''
 █████   ██████ ████████ ██    ██  █████  ████████  ██████  ██████  ███████
██   ██ ██         ██    ██    ██ ██   ██    ██    ██    ██ ██   ██ ██
███████ ██         ██    ██    ██ ███████    ██    ██    ██ ██████  ███████
██   ██ ██         ██    ██    ██ ██   ██    ██    ██    ██ ██   ██      ██
██   ██  ██████    ██     ██████  ██   ██    ██     ██████  ██   ██ ███████
'''









'''
██████  ███████  ██████  ██    ██ ██       █████  ████████  ██████  ██████  ███████
██   ██ ██      ██       ██    ██ ██      ██   ██    ██    ██    ██ ██   ██ ██
██████  █████   ██   ███ ██    ██ ██      ███████    ██    ██    ██ ██████  ███████
██   ██ ██      ██    ██ ██    ██ ██      ██   ██    ██    ██    ██ ██   ██      ██
██   ██ ███████  ██████   ██████  ███████ ██   ██    ██     ██████  ██   ██ ███████
'''





