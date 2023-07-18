import equinox as eqx
import jax.numpy as jnp
import jax

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
    capIdx: str
    biasPIdx: str
    Emax: float
    Emin: float
    hc: float
    TSys: float
    volIdx: str
    slope: float

    def __init__(self, 
                 capIdx: jnp.ndarray,
                 biasPIdx: jnp.ndarray,
                 Emax: jnp.ndarray,
                 Emin: jnp.ndarray,
                 hc: jnp.ndarray,
                 TSys: jnp.ndarray,
                 volIdx: jnp.ndarray,
                 slope: jnp.ndarray = 0.0):
        self.capIdx = capIdx
        self.biasPIdx = biasPIdx
        self.Emax = Emax
        self.Emin = Emin
        self.hc = hc
        self.TSys = TSys
        self.volIdx = volIdx
        self.slope = slope
        pass

    def pressure(self, 
                 t: jnp.ndarray,
                 y:dict[str,float], 
                 ) -> jnp.ndarray:
        pressure = (y[self.volIdx] + (y[self.capIdx] * y[self.biasPIdx])) / y[self.capIdx]
        return pressure

# Class for the elastance capacitor. The elastance capacitor is a capacitor that has a variable capacity.
# This is used to calculate the driving functions of the heart ventricles.
class ElastanceCapacitor(Capacitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pressure(self, 
                 t: jnp.ndarray, 
                 y:dict[str,float]) -> jnp.ndarray:
        
        elastance = 0.0
        
        t0 = jnp.remainder(t, self.hc)
        
        def cnd1(t00):
            cond1 = 1 - jnp.cos((jnp.pi * t00) / self.TSys)
            cond2 = 1 + jnp.cos((2 * jnp.pi * (t00 - self.TSys)) / self.TSys)
            Esin = jnp.where(t00 <= self.TSys, cond1, cond2)
            el = (Esin * ((self.Emax - self.Emin) / 2)) + self.Emin
            return el

        elastance = jnp.where(t0 <= (1.5 * self.TSys), cnd1(t0), self.Emin*1.0)
        
        

        newCapacity = 1/elastance
        pressure = (y[self.volIdx] + (newCapacity * y[self.biasPIdx])) / newCapacity

        return pressure
    

# Class for the ventilator pressure is of the capacitor type, even though the variables names do not match with 
# the capacitor class. This is because the ventilator pressure is calculated in the same way as the capacitor pressure.
# Pmin = self.Emin
# Pmax = self.Emax
# rc = self.hc
# Tinsp = self.TSys
# slope = self.slope
# amp = Pmax - Pmin
# slopeFreq = 1 / slope
class VentilatorPressure(Capacitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pressure(self, 
                 t: jnp.ndarray, 
                 y:dict[str,float]) -> jnp.ndarray:
        
        amp = self.Emax - self.Emin
        slopeFreq = 1 / self.slope
        t0 = jnp.remainder(t, self.hc)

        def cnd2(t000):
            pr1 = self.Emin + amp * jnp.sin(0.5 * slopeFreq * jnp.pi * t0)
            pr2 = self.Emax*1.0
            return jnp.where(t000 < self.slope, pr1, pr2)

        def cnd1(t00):
            pr2 = self.Emin + amp * jnp.cos(0.5 * slopeFreq * jnp.pi * (t0 - (self.TSys)))
            return jnp.where(t00 < self.TSys, cnd2(t00), pr2)

        pressure = jnp.where(t0 < (self.TSys + self.slope), cnd1(t0) , self.Emin*1.0)
        return pressure

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

    def __init__(self, fInIdxs: list, fOutIdxs: list):
        self.fInIdxs = fInIdxs
        self.fOutIdxs = fOutIdxs

    def dV(self,y:dict[str,float]) -> jnp.ndarray:
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

        return dV
    


'''
 ██████   █████  ███████     ███████ ██   ██  ██████ ██   ██  █████  ███    ██  ██████  ███████ 
██       ██   ██ ██          ██       ██ ██  ██      ██   ██ ██   ██ ████   ██ ██       ██      
██   ███ ███████ ███████     █████     ███   ██      ███████ ███████ ██ ██  ██ ██   ███ █████   
██    ██ ██   ██      ██     ██       ██ ██  ██      ██   ██ ██   ██ ██  ██ ██ ██    ██ ██      
 ██████  ██   ██ ███████     ███████ ██   ██  ██████ ██   ██ ██   ██ ██   ████  ██████  ███████ 
'''