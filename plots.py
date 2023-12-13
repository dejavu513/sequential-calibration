import numpy as np
import matplotlib.pyplot as mpl
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
from adjustText import adjust_text
import jax
import jax.numpy as jnp

def calculateThoraxUnstressedVolume(modelObjects,results):
    equation = modelObjects['regions']['P_Plr']
    children = equation.children
    cycle = equation.cycle
    eMinIdx = equation.eMinIdx
    eMaxIdx = equation.eMaxIdx

    res = []
    for i,t in enumerate(results['T']):
        volumeChildren = 0.0
        for child in children:
            volumeChildren += results[child][i]
        
        
        #tSys = 0.3 * jnp.sqrt(y['SC'])
        tSys = 0.4 * results[cycle][i]
        t0 = results['timer' + cycle][i]
        multiplier = 0.5

        
        def cnd1(t00):
            cond1 = 1 - jnp.cos((jnp.pi * t00) / tSys)
            cond2 = 1 + jnp.cos((1/multiplier * jnp.pi * (t00-tSys)) / tSys)
            Esin = jnp.where(t00 <= tSys, cond1, cond2)
            el = (Esin * ((results[eMaxIdx][i] - results[eMinIdx][i]) / 2)) + results[eMinIdx][i]
            return el

        
        res.append(jnp.where(t0 <= (multiplier*tSys + tSys), cnd1(t0), results[eMinIdx][i]))


    return np.array(res)

def buildElastanceCurve(duration, eMin, eMax, step, param1=0.3, param2=1.5):
    def cnd1(t00):
        cond1 = 1 - jnp.cos((jnp.pi * t00) / tSys)
        cond2 = 1 + jnp.cos((2 * jnp.pi * (t00 - tSys)) / tSys)
        Esin = jnp.where(t00 <= tSys, cond1, cond2)
        el = (Esin * ((eMax - eMin) / 2)) + eMin
        return el
    
    res = []
    for i,t in enumerate(np.arange(0, duration, step)):
        tSys = param1 * duration
        res.append(jnp.where(t <= (param2 * tSys), cnd1(t), eMin))
    
    return np.array(res)


def calculateElastance(modelObjects,results,id):
    equation = modelObjects['capacitors'][id]
    volIdx = equation.volIdx
    hcIdx = equation.hcIdx
    biasPIdx = equation.biasPIdx
    eMinIdx = equation.eMinIdx
    eMaxIdx = equation.eMaxIdx

    res = []
    for i,t in enumerate(results['T']):

        tSys = 0.3 * results[hcIdx][i]
        t0 = results['timer' + hcIdx][i]

        
        def cnd1(t00):
            cond1 = 1 - jnp.cos((jnp.pi * t00) / tSys)
            cond2 = 1 + jnp.cos((2 * jnp.pi * (t00 - tSys)) / tSys)
            Esin = jnp.where(t00 <= tSys, cond1, cond2)
            el = (Esin * ((results[eMaxIdx][i] - results[eMinIdx][i]) / 2)) + results[eMinIdx][i]
            return el

        
        res.append(1/jnp.where(t0 <= (1.5 * tSys), cnd1(t0), results[eMinIdx][i]))


    return np.array(res)


def custom_format(x, pos):
    return f'{x:.3f}'

#██████  ██████  ███████       ██████  ██    ██ ██ ██      ████████      █████  ██   ██ ███████ ███████ 
#██   ██ ██   ██ ██            ██   ██ ██    ██ ██ ██         ██        ██   ██  ██ ██  ██      ██      
#██████  ██████  █████   █████ ██████  ██    ██ ██ ██         ██        ███████   ███   █████   ███████ 
#██      ██   ██ ██            ██   ██ ██    ██ ██ ██         ██        ██   ██  ██ ██  ██           ██ 
#██      ██   ██ ███████       ██████   ██████  ██ ███████    ██        ██   ██ ██   ██ ███████ ███████ 

def buildPressureAndVolumeAxis(ax,results,pressure,volume,title,atmPressure,t):
    formatter = ScalarFormatter(useMathText=False, useOffset=False)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_title(title)

    ax.plot(np.array(t),np.array(results[pressure]) - atmPressure, color='tab:blue', label = pressure)
    ax.set_ylabel('mmHg', color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    
    ax_11 = ax.twinx()
    ax_11.xaxis.set_major_formatter(formatter)
    ax_11.yaxis.set_major_formatter(formatter)
    ax_11.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_11.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_11.plot(np.array(t),np.array(results[volume]), color='tab:red', label = volume)
    ax_11.set_ylabel('mL', color='tab:red')
    ax_11.tick_params(axis='y', labelcolor='tab:red')
    
    ax.set_ylim(min(np.array(results[pressure]) - atmPressure), max(np.array(results[pressure]) - atmPressure))
    ax_11.set_ylim(min(np.array(results[volume])), max(np.array(results[volume])))
    ax.set_yticks([
        min(np.array(results[pressure]) - atmPressure), 
        max(np.array(results[pressure]) - atmPressure), 
        np.round(np.mean(np.array(results[pressure])) - atmPressure,4)
        ])
    ax_11.set_yticks([
        min(np.array(results[volume])), 
        max(np.array(results[volume])),
        np.round(np.mean(np.array(results[volume])),4)
        ])
    
# If one i spressure use Var 1 slot and add a non-zero bias pressure (atmPressure)
def buildLeftRightAxisIndependentScales(ax,results,var1,var2,title,label1,label2,color1,color2,atmPressure,t):
    formatter = ScalarFormatter(useMathText=False, useOffset=False)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    
    toIgone = 2
    ax.set_title(title)

    ax.plot(np.array(t[toIgone:]),np.array(results[var1][toIgone:] - atmPressure), color=color1, label = var1)
    ax.set_ylabel(label1, color=color1)
    ax.tick_params(axis='y', labelcolor=color1)
    
    ax_11 = ax.twinx()
    ax_11.xaxis.set_major_formatter(formatter)
    ax_11.yaxis.set_major_formatter(formatter)
    ax_11.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_11.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_11.plot(np.array(t[toIgone:]),np.array(results[var2][toIgone:]), color=color2, label = var2)
    ax_11.set_ylabel(label2, color=color2)
    ax_11.tick_params(axis='y', labelcolor=color2)
    
    #ax.set_ylim(min(np.array(results[var1] - atmPressure)), max(np.array(results[var1] - atmPressure)))
    #ax_11.set_ylim(min(np.array(results[var2])), max(np.array(results[var2])))
    ax.set_yticks([
        np.round(min(np.array(results[var1][toIgone:] - atmPressure)),4), 
        np.round(max(np.array(results[var1][toIgone:] - atmPressure)),4), 
        np.round(np.mean(np.array(results[var1] - atmPressure)),4)
        ])
    ax_11.set_yticks([
        np.round(min(np.array(results[var2][toIgone:])),4), 
        np.round(max(np.array(results[var2][toIgone:])),4),
        np.round(np.mean(np.array(results[var2][toIgone:])),4)
        ])

def buildLeftRightAxisIndependentPressureScales(ax,results,var1,var2,title,label1,label2,color1,color2,atmPressure,t):
    formatter = ScalarFormatter(useMathText=False, useOffset=False)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_title(title)

    ax.plot(np.array(t),np.array(results[var1] - atmPressure), color=color1, label = var1)
    ax.set_ylabel(label1, color=color1)
    ax.tick_params(axis='y', labelcolor=color1)
    
    ax_11 = ax.twinx()
    ax_11.xaxis.set_major_formatter(formatter)
    ax_11.yaxis.set_major_formatter(formatter)
    ax_11.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_11.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_11.plot(np.array(t),np.array(results[var2] - atmPressure), color=color2, label = var2)
    ax_11.set_ylabel(label2, color=color2)
    ax_11.tick_params(axis='y', labelcolor=color2)
    
    ax.set_ylim(min(np.array(results[var1]) - atmPressure), max(np.array(results[var1]) - atmPressure))
    ax_11.set_ylim(min(np.array(results[var2] - atmPressure)), max(np.array(results[var2] - atmPressure)))
    ax.set_yticks([
        np.round(min(np.array(results[var1] - atmPressure)),4), 
        np.round(max(np.array(results[var1] - atmPressure)),4), 
        np.round(np.mean(np.array(results[var1] - atmPressure)),4)
        ])
    ax_11.set_yticks([
        np.round(min(np.array(results[var2] - atmPressure)),4), 
        np.round(max(np.array(results[var2] - atmPressure)),4),
        np.round(np.mean(np.array(results[var2] - atmPressure)),4)
        ])

# If one i spressure use Var 1 slot and add a non-zero bias pressure (atmPressure)
def buildLeftRightAxisDependentScales(ax,results,var1,var2,title,label1,label2,color1,color2,atmPressure,t):
    formatter = ScalarFormatter(useMathText=False, useOffset=False)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    toIgone = 2
    ax.set_title(title)

    ax.plot(np.array(t[toIgone:]),np.array(results[var1][toIgone:] - atmPressure), color=color1, label = var1)
    ax.set_ylabel(label1, color=color1)
    ax.tick_params(axis='y', labelcolor=color1)
    
    ax_11 = ax.twinx()
    ax_11.xaxis.set_major_formatter(formatter)
    ax_11.yaxis.set_major_formatter(formatter)
    ax_11.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_11.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_11.plot(np.array(t[toIgone:]),np.array(results[var2][toIgone:]), color=color2, label = var2)
    ax_11.set_ylabel(label2, color=color2)
    ax_11.tick_params(axis='y', labelcolor=color2)
    
    ax.set_ylim(min(np.array(results[var1][toIgone:] - atmPressure)), max(np.array(results[var1][toIgone:] - atmPressure)))
    ax_11.set_ylim(min(np.array(results[var2][toIgone:])), max(np.array(results[var2][toIgone:])))
    ax.set_yticks([
        np.round(min(np.array(results[var1][toIgone:] - atmPressure))), 
        np.round(max(np.array(results[var1][toIgone:] - atmPressure))), 
        np.round(np.mean(np.array(results[var1])),4)
        ])
    ax_11.set_yticks([
        np.round(min(np.array(results[var2][toIgone:]))), 
        np.round(max(np.array(results[var2][toIgone:]))),
        np.round(np.mean(np.array(results[var2][toIgone:])),4)
        ])
    
    minP = min(min(np.array(results[var1][toIgone:] - atmPressure)), min(np.array(results[var2][toIgone:])))
    maxP = max(max(np.array(results[var1][toIgone:] - atmPressure)), max(np.array(results[var2][toIgone:])))
    
    ax.set_ylim([minP -1, maxP+1])
    ax_11.set_ylim([minP-1, maxP+1])

def buildLeftRightAxisDependentPressureScales(ax,results,var1,var2,title,label1,label2,color1,color2,atmPressure,t):
    formatter = ScalarFormatter(useMathText=False, useOffset=False)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_title(title)

    ax.plot(np.array(t),np.array(results[var1] - atmPressure), color=color1, label = var1)
    ax.set_ylabel(label1, color=color1)
    ax.tick_params(axis='y', labelcolor=color1)
    
    ax_11 = ax.twinx()
    ax_11.xaxis.set_major_formatter(formatter)
    ax_11.yaxis.set_major_formatter(formatter)
    ax_11.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_11.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_11.plot(np.array(t),np.array(results[var2] - atmPressure), color=color2, label = var2)
    ax_11.set_ylabel(label2, color=color2)
    ax_11.tick_params(axis='y', labelcolor=color2)
    
    ax.set_ylim(min(np.array(results[var1]) - atmPressure), max(np.array(results[var1]) - atmPressure))
    ax_11.set_ylim(min(np.array(results[var2] - atmPressure)), max(np.array(results[var2] - atmPressure)))
    ax.set_yticks([
        np.round(min(np.array(results[var1] - atmPressure)),4), 
        np.round(max(np.array(results[var1] - atmPressure)),4), 
        np.round(np.mean(np.array(results[var1])),4)
        ])
    ax_11.set_yticks([
        np.round(min(np.array(results[var2] - atmPressure)),4), 
        np.round(max(np.array(results[var2] - atmPressure)),4),
        np.round(np.mean(np.array(results[var2] - atmPressure)),4)
        ])
    
    minP = min(min(np.array(results[var1] - atmPressure)), min(np.array(results[var2] - atmPressure)))
    maxP = max(max(np.array(results[var1] - atmPressure)), max(np.array(results[var2] - atmPressure)))
    
    ax.set_ylim([minP -1, maxP+1])
    ax_11.set_ylim([minP-1, maxP+1])

def thoraxMouthPressureVolumeAxes(ax,results,atmPressure,sampPeriod,t):
    mouthVolumeThrough = np.array(np.cumsum(results['Q_VenLt']*sampPeriod))
    volumeThx = np.array(results['V_Vt'])+np.array(results['V_Hr'])+np.array(results['V_Ap'])+np.array(results['V_Cp'])+np.array(results['V_Vp'])+np.array(results['V_Hl'])+np.array(results['V_La'])+np.array(results['V_Lb'])
    
    ax.set_title('Pressure-Volume Curves')
    ax.plot(np.array(volumeThx)-min(np.array(volumeThx)), np.array(results['P_Thx']) - atmPressure, color='tab:blue', label = 'Thorax')
    ax.set_yticks([min(np.array(results['P_Thx']) - atmPressure), max(np.array(results['P_Thx']) - atmPressure), np.round(np.mean(np.array(results['P_Thx']) - atmPressure),4)])
    ax.set_ylabel('Thorax', color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_xlabel('Volume [mL]')


    ax_11 = ax.twinx()
    #ax_11.plot(np.array(results['P_La']) - atmPressure, np.array(results['V_La'])-min(np.array(results['V_La'])), color='tab:green', label = 'Alveoli (La)')
    ax_11.plot(np.array(results['V_La'])-min(np.array(results['V_La'])), np.array(results['P_La']) - atmPressure,  color='tab:green', label = 'Alveoli (La)')
    ax_11.plot(mouthVolumeThrough, np.array(results['P_Lt']) - atmPressure, color='tab:pink', label = 'Trachea (Lt)')
    ax_11.set_yticks([min(np.array(results['P_La']) - atmPressure), max(np.array(results['P_La']) - atmPressure)])
    ax_11.tick_params(axis='y', labelcolor='tab:green')
    ax_11.set_xlabel('Volume [mL]')

    ax_12 = ax.twinx()
    ax_12.set_yticks([min(np.array(results['P_Lt']) - atmPressure), max(np.array(results['P_Lt']) - atmPressure)])
    ax_12.tick_params(axis='y', labelcolor='tab:pink')
    ax_12.set_xlabel('Volume [mL]')

    minP = min(min(np.array(results['P_Lt'] - atmPressure)), min(np.array(results['P_La'] - atmPressure)))
    maxP = max(max(np.array(results['P_Lt'] - atmPressure)), max(np.array(results['P_La'] - atmPressure)))
    
    ax_11.set_ylim([minP-1, maxP+1])
    #ax_11.set_ylim([00, 5000])
    ax_12.set_ylim([minP-1, maxP+1])

    x =[
        max(np.array(volumeThx)-min(np.array(volumeThx))),
    ]
    y =[
        np.array(results['P_Thx'])[np.argmax(np.array(volumeThx))] - atmPressure,
    ]
    tick_labels = ['Thorax: ' + str(np.round(max(np.array(volumeThx)-min(np.array(volumeThx))),1)) + ' mL   ']

    texts = [ax.text(x, y, label) for x, y, label in zip(x, y, tick_labels)]
    adjust_text(texts, arrowprops=dict(arrowstyle='->', lw=1.5))

    x =[
        max(np.array(results['V_La'])-min(np.array(results['V_La']))),
        max(mouthVolumeThrough),
    ]
    y =[
        np.array(results['P_La'])[np.argmax(np.array(results['V_La']))] - atmPressure,
        np.array(results['P_Lt'])[np.argmax(mouthVolumeThrough)-1] - atmPressure,
    ]
    tick_labels = [
        'Alveoli: ' + str(np.round(max(np.array(results['V_La'])-min(np.array(results['V_La']))),1)) + ' mL',
        'Trachea: ' + str(np.round(max(mouthVolumeThrough),1)) + ' mL'
        ]

    texts = [ax_11.text(x, y, label) for x, y, label in zip(x, y, tick_labels)]
    adjust_text(texts, arrowprops=dict(arrowstyle='->', lw=1.5))

def heartPressureVolumeAxes(ax,results,title,atmPressure):
    formatter = ScalarFormatter(useMathText=False, useOffset=False)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_title(title)
    ax.plot(np.array(results['V_Hl']), np.array(results['P_Hl']) - atmPressure, color='tab:blue', label = 'Left Ventricle (Hl)')
    ax.plot(np.array(results['V_Hr']), np.array(results['P_Hr']) - atmPressure, color='tab:red', label = 'Right Ventricle (Hr)')
    ax.set_ylabel('Pressure [mmHg]')
    ax.set_xlabel('Volume [mL]')
    ax.legend()

def plotDoubleSigmoidAxes(idx, structures, results, ax_1):
    structure = structures['modelStructure']['controllers'][idx]
    idxToIgnore = 10
    atmPressure = 760.0
    targetValue = float(structure[2])
    minValueToControl = float(structure[3])
    #maxValueToControl = float(structure[4])
    slope = float(structure[5])
    slope1 = float(structure[6])

    step = 1
    separation = 0.4
    
    duration = float(structure[2])*2
    V0 = np.arange(0,duration,step)
    sigmoid = []
    sigmoid1 = []

    pressure = []
    
    
    inflectionPoint = targetValue
    dist = inflectionPoint * separation
    
    
    cMin = 30
    offset = minValueToControl
    amplitude = 0.0 - minValueToControl
    
    threshold = 0000
    
    for i in V0:
        c = amplitude / (1 + np.exp(-slope * (i - inflectionPoint-dist))) + offset
        c1 = amplitude / (1 + np.exp(slope1 * (i - inflectionPoint+dist))) + offset + cMin
        sigmoid.append(c)
        sigmoid1.append(c1)

    difference = np.abs(np.array(sigmoid) - np.array(sigmoid1))
    intersection = np.where(difference == np.min(difference))[0][0]

    mergedSigmoids = sigmoid1[0:int(intersection)] + sigmoid[int(intersection)-1:-1]

    for i in V0:
        pressure.append(i/mergedSigmoids[int(i)])

    
    
    p= -np.array(results['P_La'][idxToIgnore:] - atmPressure) + np.array(results['P_Thx'][idxToIgnore:]- atmPressure)
    vol = np.array(results[structure[0]][idxToIgnore:])
    com = np.array(results[structure[1]][idxToIgnore:])
    

    #ax_1.plot(V0,sigmoid, color='tab:red', label = 'Sigmoid', linewidth=1)
    #ax_1.plot(V0,sigmoid1, color='tab:blue', label = 'Sigmoid1', linewidth=1)
    ax_1.set_title('Compliance/Volume and Pressure/Volume Curves of the Alveoli')
    ax_1.plot(V0, mergedSigmoids, color='tab:green', label = 'Merged Sigmoids', linewidth=1.5)
    ax_1.plot(vol, com, color='tab:brown', label = structure[1], linewidth=2.5)
    ax_11 = ax_1.twinx()
    ax_11.plot(V0[0:5050],pressure[0:5050], color='tab:grey', label = 'Pressure', linewidth=1.5)
    ax_11.plot(vol, -p, color='tab:blue', label = '-(P_Thorax-P_Alveoli)', linewidth=2.5)
    
    
    ax_1.set_ylabel('Compliance [mL/mmHg]', color='tab:green')
    ax_1.tick_params(axis='y', labelcolor='tab:green')
    ax_1.set_xlabel('Volume [mL]')
    ax_11.set_xlabel('Volume [mL]')
    ax_11.set_ylabel('Pressure [mmhg]', color='tab:grey')
    ax_11.tick_params(axis='y', labelcolor='tab:grey')

    #ax_1.set_ylim(min(np.array(results[var1]) - atmPressure), max(np.array(results[var1]) - atmPressure))
    #ax_11.set_ylim(min(np.array(results[var2] - atmPressure)), max(np.array(results[var2] - atmPressure)))
   
    infPoint = int(inflectionPoint-dist)
    infPoint1 = int(inflectionPoint+dist)
    
    '''
    '''
    #print(int(min(vol)))
    ax_1.set_yticks([
        mergedSigmoids[int(np.where(V0 == 0)[0])],
        mergedSigmoids[int(np.where(V0 == int(min(vol)))[0])], 
        mergedSigmoids[int(np.where(V0 == infPoint)[0])],
        mergedSigmoids[int(np.where(V0 == int(max(vol)))[0])], 
        mergedSigmoids[int(np.where(V0 == int(intersection))[0])],
        mergedSigmoids[int(np.where(V0 == infPoint1)[0])],
        #mergedSigmoids[int(np.where(V0 == 4999)[0])],
        ])
    
    ax_11.set_yticks([
        pressure[int(np.where(V0 == 0)[0])],
        pressure[int(np.where(V0 == int(min(vol)))[0])], 
        pressure[int(np.where(V0 == infPoint)[0])],
        pressure[int(np.where(V0 == int(max(vol)))[0])], 
        pressure[int(np.where(V0 == int(intersection))[0])],
        pressure[int(np.where(V0 == infPoint1)[0])],
        #pressure[int(np.where(V0 == 4999)[0])],
        ])
    #ax_1.grid(True)
    ax_1.set_xticks([
        0,
        int(min(vol)), 
        #np.round(np.mean(vol),4),
        infPoint,
        int(max(vol)), 
        int(intersection),
        infPoint1,
        7000,
        ])
    ax_11.set_xticks([
        0,
        int(min(vol)), 
        #np.round(np.mean(vol),4),
        infPoint,
        int(max(vol)), 
        int(intersection),
        infPoint1,
        7000,
        ])

    x =[
        int(min(vol)), 
        infPoint,
        int(max(vol)), 
        int(intersection),
        infPoint1,
    ]
    y =[
        mergedSigmoids[int(np.where(V0 == int(min(vol)))[0])], 
        mergedSigmoids[int(np.where(V0 == infPoint)[0])],
        mergedSigmoids[int(np.where(V0 == int(max(vol)))[0])], 
        mergedSigmoids[int(np.where(V0 == int(intersection))[0])],
        mergedSigmoids[int(np.where(V0 == infPoint1)[0])],
    ]
    y1 =[
        pressure[int(np.where(V0 == int(min(vol)))[0])], 
        pressure[int(np.where(V0 == infPoint)[0])],
        pressure[int(np.where(V0 == int(max(vol)))[0])], 
        pressure[int(np.where(V0 == int(intersection))[0])],
        pressure[int(np.where(V0 == infPoint1)[0])],
    ]
    tick_labels = [
        'C1',
        'C2',
        'C3 ',
        'C4',
        'C5',
        ]
    tick_labels1 = [
        'P1',
        'P2',
        'P3',
        'P4',
        'P5',
        ]
    texts = [ax_1.text(x, y, label) for x, y, label in zip(x, y, tick_labels)]
    adjust_text(texts, V0, mergedSigmoids, force_text=0.5, arrowprops=dict(arrowstyle='->', lw=1.5))
    texts1 = [ax_11.text(x, y1, label) for x, y1, label in zip(x, y1, tick_labels1)]
    adjust_text(texts1, V0, pressure, force_text=0.5, arrowprops=dict(arrowstyle='->', lw=1.5))

def plotDoubleSigmoidAxesTree(idx, structures, results, ax_1):
    structure = structures['modelStructure']['controllers'][idx]
    idxToIgnore = 10
    atmPressure = 760.0
    targetValue = float(structure[2])
    minValueToControl = float(structure[3])
    #maxValueToControl = float(structure[4])
    slope = float(structure[5])
    slope1 = float(structure[6])

    step = 1
    separation = 0.4
    
    duration = float(structure[2])*2
    V0 = np.arange(0,duration,step)
    sigmoid = []
    sigmoid1 = []

    pressure = []
    
    
    inflectionPoint = targetValue
    dist = inflectionPoint * separation
    
    
    cMin = 30
    offset = minValueToControl
    amplitude = 0.0 - minValueToControl

    for i in V0:
        c = amplitude / (1 + np.exp(-slope * (i - inflectionPoint-dist))) + offset
        c1 = amplitude / (1 + np.exp(slope1 * (i - inflectionPoint+dist))) + offset + cMin
        sigmoid.append(c)
        sigmoid1.append(c1)

    difference = np.abs(np.array(sigmoid) - np.array(sigmoid1))
    intersection = np.where(difference == np.min(difference))[0][0]

    mergedSigmoids = sigmoid1[0:int(intersection)] + sigmoid[int(intersection)-1:-1]

    for i in V0:
        pressure.append(i/mergedSigmoids[int(i)])

    p= -np.array(results['P' + structure[0][1:]][idxToIgnore:] - atmPressure) + np.array(results['P_Thx'][idxToIgnore:]- atmPressure)
    vol = np.array(results[structure[0]][idxToIgnore:])
    com = np.array(results[structure[1]][idxToIgnore:])
    
    ax_1.set_title('Sigmoid' + structure[0][2:])
    ax_1.plot(V0, mergedSigmoids, color='tab:green', label = 'Merged Sigmoids', linewidth=1.5)
    ax_1.plot(vol, com, color='tab:brown', label = structure[1], linewidth=2.5)
    ax_11 = ax_1.twinx()

    lenV0 = int(inflectionPoint*1.4)
    ax_11.plot(V0[0:lenV0],pressure[0:lenV0], color='tab:grey', label = 'Pressure', linewidth=1.5)
    ax_11.plot(vol, -p, color='tab:blue', label = '-(P_Thorax-P_Alveoli)', linewidth=2.5)
    
    
    ax_1.tick_params(axis='y', labelcolor='tab:green')
    ax_11.tick_params(axis='y', labelcolor='tab:grey')
   
    infPoint = int(inflectionPoint-dist)
    infPoint1 = int(inflectionPoint+dist)
    
    ax_1.set_yticks([
        mergedSigmoids[int(np.where(V0 == 0)[0])],
        #mergedSigmoids[int(np.where(V0 == int(min(vol)))[0])], 
        mergedSigmoids[int(np.where(V0 == infPoint)[0])],
        #mergedSigmoids[int(np.where(V0 == int(max(vol)))[0])], 
        mergedSigmoids[int(np.where(V0 == int(intersection))[0])],
        mergedSigmoids[int(np.where(V0 == infPoint1)[0])],
        ])
    
    ax_11.set_yticks([
        pressure[int(np.where(V0 == 0)[0])],
        #pressure[int(np.where(V0 == int(min(vol)))[0])], 
        pressure[int(np.where(V0 == infPoint)[0])],
        #pressure[int(np.where(V0 == int(max(vol)))[0])], 
        pressure[int(np.where(V0 == int(intersection))[0])],
        pressure[int(np.where(V0 == infPoint1)[0])],
        ])
    ax_1.set_xticks([
        0,
        #int(min(vol)), 
        infPoint,
        #int(max(vol)), 
        int(intersection),
        infPoint1,
        ])
    ax_11.set_xticks([
        0,
        #int(min(vol)), 
        infPoint,
        #int(max(vol)), 
        int(intersection),
        infPoint1,
        ])

# █████   ██████ ████████ ██    ██  █████  ██          ██████  ██       ██████  ████████ ███████ 
#██   ██ ██         ██    ██    ██ ██   ██ ██          ██   ██ ██      ██    ██    ██    ██      
#███████ ██         ██    ██    ██ ███████ ██          ██████  ██      ██    ██    ██    ███████ 
#██   ██ ██         ██    ██    ██ ██   ██ ██          ██      ██      ██    ██    ██         ██ 
#██   ██  ██████    ██     ██████  ██   ██ ███████     ██      ███████  ██████     ██    ███████

def plotPressureVolumeHeart(results, totalTime, sampPeriod, atmPressure,modelObjects):
    t = np.arange(0, totalTime, sampPeriod)
    fig = mpl.figure(figsize=(15, 10))
    grid = gridspec.GridSpec(7, 2, figure=fig)
    
    ################################################################################################################################
    ax_1 = fig.add_subplot(grid[0:3, 0])
    heartPressureVolumeAxes(ax_1,results,'Pressure-Volume Curves',atmPressure)

    ################################################################################################################################
    ax_2 = fig.add_subplot(grid[3, 0])
    buildLeftRightAxisDependentScales(ax_2,results,'i_COl','i_COr','Stroke Volumes [mL]','Hl','Hr','tab:blue','tab:red',0.0,t)
    ax_2.set_xlabel('Time [s]')

    ax_3 = fig.add_subplot(grid[5, 0])
    buildPressureAndVolumeAxis(ax_3,results,'P_Hl','V_Hl','Left Ventricle (Hl)',atmPressure,t)
    #buildLeftRightAxisDependentPressureScales(ax_3,results,'P_Hl','P_As','Pressure Left Ventricle (Hl) & Systemic Artery (As)  [mmHg]','Hl','As','tab:blue','tab:green',atmPressure,t)
    ax_3.set_xlabel('Time [s]')

    ax_4 = fig.add_subplot(grid[6, 0])
    buildPressureAndVolumeAxis(ax_4,results,'P_Hr','V_Hr','Right Ventricle (Hl)',atmPressure,t)
    #buildLeftRightAxisDependentPressureScales(ax_4,results,'P_Hr','P_Ap','Pressure Left Ventricle (Hl) & Systemic Artery (As)  [mmHg]','Hr','Ap','tab:blue','tab:purple',atmPressure,t)
    ax_4.set_xlabel('Time [s]')

    #ax_5 = fig.add_subplot(grid[6, 0])
    #buildLeftRightAxisDependentScales(ax_5,results,'V_Hl','V_Hr','Volumes [mL]','Hl','Hr','tab:blue','tab:red',0.0,t)
    #ax_5.set_xlabel('Time [s]')

    ax_6 = fig.add_subplot(grid[4, 0])
    results['e_Hl'] = calculateElastance(modelObjects,results,'P_Hl')
    results['e_Hr'] = calculateElastance(modelObjects,results,'P_Hr')
    buildLeftRightAxisIndependentScales(ax_6,results,'e_Hl','e_Hr','Ventricle Compliances','Hl','Hr','tab:blue','tab:red',0.0,t)
    ax_6.set_xlabel('Time [s]')
    ################################################################################################################################

    ax_7 = fig.add_subplot(grid[0, 1])
    buildPressureAndVolumeAxis(ax_7,results,'P_Ap','V_Ap','Pulmonary Arteries (Ap)',atmPressure,t)

    ax_8 = fig.add_subplot(grid[1, 1])
    buildPressureAndVolumeAxis(ax_8,results,'P_Cp','V_Cp','Pulmonary Capillaries (Cp)',atmPressure,t)

    ax_9 = fig.add_subplot(grid[2, 1])
    buildPressureAndVolumeAxis(ax_9,results,'P_Vp','V_Vp','Pulmonary Veins (Vp)',atmPressure,t)

    ax_10 = fig.add_subplot(grid[3, 1])
    buildPressureAndVolumeAxis(ax_10,results,'P_As','V_As','Systemic Arteries (As)',atmPressure,t)

    ax_11 = fig.add_subplot(grid[4, 1])
    buildPressureAndVolumeAxis(ax_11,results,'P_Cs','V_Cs','Systemic Capillaries (Cs)',atmPressure,t)

    ax_12 = fig.add_subplot(grid[5, 1])
    buildPressureAndVolumeAxis(ax_12,results,'P_Vs','V_Vs','Systemic Veins (Vs)',atmPressure,t)

    ax_13 = fig.add_subplot(grid[6, 1])
    buildPressureAndVolumeAxis(ax_13,results,'P_Vt','V_Vt','Thoracic Veins (Vt)',atmPressure,t)

    mpl.tight_layout()
    mpl.show()

def plotPressureVolumeAlveoli(results, totalTime, sampPeriod, atmPressure, modelObjects,idx,structures):
    toIgone = 2
    t = np.arange(0, totalTime, sampPeriod)
    
    # Create a figure and a 2x2 grid of subplots
    fig = mpl.figure(figsize=(15, 9))
    grid = gridspec.GridSpec(5, 2, figure=fig)
    volumeThx = np.array(results['V_Vt'])+np.array(results['V_Hr'])+np.array(results['V_Ap'])+np.array(results['V_Cp'])+np.array(results['V_Vp'])+np.array(results['V_Hl'])+np.array(results['V_La'])+np.array(results['V_Lb'])
    

    ##################################################################################################################################
    ax_1 = fig.add_subplot(grid[:2, 0])
    thoraxMouthPressureVolumeAxes(ax_1,results,atmPressure,sampPeriod,t)

    ax_8 = fig.add_subplot(grid[2:-1, 0])
    plotDoubleSigmoidAxes(idx, structures, results, ax_8)
    
    ##################################################################################################################################
    ax_2 = fig.add_subplot(grid[-1, 0])
    ax_2.set_title('Alveoli Compliance [mL/mmHg]')
    ax_2.plot(np.array(t),np.array(results['C_La']), color='tab:olive', label = 'Alveoli compliance')
    ax_2.set_yticks([min(np.array(results['C_La'])), max(np.array(results['C_La'])), np.round(np.mean(np.array(results['C_La'])),4)])
    ax_2.set_xlabel('Time [s]')

    ##################################################################################################################################
    ax_3 = fig.add_subplot(grid[0, 1])
    ax_3.set_title('Lung Pressures [mmHg]')
    ax_3.plot(np.array(t[toIgone:]),np.array(results['P_Ven'][toIgone:]) - atmPressure, color='tab:blue', label = 'Ventilator')
    ax_3.plot(np.array(t[toIgone:]),np.array(results['P_Lt'][toIgone:]) - atmPressure, color='tab:pink', label = 'Trachea')
    ax_3.plot(np.array(t[toIgone:]),np.array(results['P_Lb'][toIgone:]) - atmPressure, color='tab:orange', label = 'Bronchi')
    ax_3.plot(np.array(t[toIgone:]),np.array(results['P_La'][toIgone:]) - atmPressure, color='tab:green', label = 'Alveoli') 
    ax_3.set_yticks([min(np.array(results['P_Ven']) - atmPressure), max(np.array(results['P_Ven']) - atmPressure), np.round(np.mean(np.array(results['P_Ven']) - atmPressure),4)])
    ax_3.set_ylabel('Ventilator', color='tab:blue')
    ax_3.tick_params(axis='y', labelcolor='tab:blue')

    ax_31 = ax_3.twinx()
    ax_31.set_yticks([min(np.array(results['P_La'][toIgone:]) - atmPressure), max(np.array(results['P_La'][toIgone:]) - atmPressure), np.round(np.mean(np.array(results['P_La'][toIgone:]) - atmPressure),4)])
    ax_31.set_ylabel('Alveoli', color='tab:green')
    ax_31.tick_params(axis='y', labelcolor='tab:green')

    minP = min(min(np.array(results['P_Ven'] - atmPressure)), min(np.array(results['P_La'] - atmPressure)))
    maxP = max(max(np.array(results['P_Ven'] - atmPressure)), max(np.array(results['P_La'] - atmPressure)))
    
    ax_3.set_ylim([minP-1, maxP+1])
    ax_31.set_ylim([minP-1, maxP+1])

    ##################################################################################################################################
    ax_4 = fig.add_subplot(grid[1, 1])
    results['V_Thx'] = volumeThx
    buildLeftRightAxisIndependentScales(ax_4,results,'V_La', 'V_Thx', 'Alveoli and Thorax Volumes [mL]','Alveoli','Thorax','tab:green','tab:blue',0.0,t)
    ax_4.set_xlabel('Time [s]')

    ax_5 = fig.add_subplot(grid[2, 1])
    buildLeftRightAxisDependentScales(ax_5,results,'V_Lt', 'V_Lb', 'Trachea and Bronchi Volumes [mL]','Lt','Lb','tab:pink','tab:orange',0.0,t)
    ax_5.set_xlabel('Time [s]')

    ##################################################################################################################################
    ax_6 = fig.add_subplot(grid[3, 1])
    ax_6.set_title('Lung Flows [mL/s]')
    ax_6.plot(np.array(t[toIgone:]),np.array(results['Q_VenLt'][toIgone:]), color='tab:pink', label = 'Trachea')
    ax_6.plot(np.array(t[toIgone:]),np.array(results['Q_LtLb'][toIgone:]), color='tab:orange', label = 'Bronchi')
    ax_6.plot(np.array(t[toIgone:]),np.array(results['Q_LbLa'][toIgone:]), color='tab:green', label = 'Alveoli') 
    ax_6.set_yticks([0.0,min(np.array(results['Q_VenLt'][toIgone:])), max(np.array(results['Q_VenLt'][toIgone:]))])
    ax_6.legend(loc='upper right')
    
    ##################################################################################################################################
    ax_7 = fig.add_subplot(grid[4, 1])
    results['uVol'] = calculateThoraxUnstressedVolume(modelObjects,results)
    buildLeftRightAxisIndependentScales(ax_7,results,'P_Thx', 'uVol', 'Thoracic Pressure and Unstressed Volume','Pressure [mmHg]','V0 [mL]','tab:blue','tab:cyan',atmPressure,t)
    ax_7.set_xlabel('Time [s]')


    mpl.tight_layout()
    mpl.show()

def plotPressureVolumeAlveoliTree(results, totalTime, sampPeriod, atmPressure, modelObjects,idx,structures):
    toIgone = 2
    t = np.arange(0, totalTime, sampPeriod)
    
    # Create a figure and a 2x2 grid of subplots
    fig = mpl.figure(figsize=(15, 9))
    grid = gridspec.GridSpec(5, 2, figure=fig)

    volumeAlveoli = np.zeros(int(totalTime/sampPeriod))
    volumeBronchi = np.zeros(int(totalTime/sampPeriod))
    for key,value in results.items():
        if key.startswith('V_La'):
            volumeAlveoli = volumeAlveoli + np.array(value)
        if key.startswith('V_Lb'):
            volumeBronchi = volumeBronchi + np.array(value)

    

    ##################################################################################################################################
    ax_1 = fig.add_subplot(grid[:2, 0])
    #thoraxMouthPressureVolumeAxes(ax_1,results,atmPressure,sampPeriod,t)
    ax_1.set_title('Alveoli Volumes [mL]')
    for key,value in results.items():
        if key.startswith('V_La'):
            ax_1.plot(np.array(t[toIgone:]),np.array(value[toIgone:]), label = key)
    ax_1.set_xlabel('Time [s]')


    ##################################################################################################################################
    ax_3 = fig.add_subplot(grid[2:-1, 0])
    ax_3.set_title('Lung Pressures [mmHg]')
    ax_3.plot(np.array(t[toIgone:]),np.array(results['P_Ven'][toIgone:]) - atmPressure, color='tab:blue', label = 'Ventilator')
    ax_3.plot(np.array(t[toIgone:]),np.array(results['P_Lt'][toIgone:]) - atmPressure, color='tab:pink', label = 'Trachea')
    for key,value in results.items():
        if key.startswith('P_La'):
            ax_3.plot(np.array(t[toIgone:]),np.array(value[toIgone:])- atmPressure, color='tab:green', label = key)
        elif key.startswith('P_Lb'):
            ax_3.plot(np.array(t[toIgone:]),np.array(value[toIgone:])- atmPressure, color='tab:orange', label = key)
    ax_3.set_yticks([min(np.array(results['P_Ven']) - atmPressure), max(np.array(results['P_Ven']) - atmPressure), np.round(np.mean(np.array(results['P_Ven']) - atmPressure),4)])
    ax_3.set_ylabel('Ventilator', color='tab:blue')
    ax_3.tick_params(axis='y', labelcolor='tab:blue')

    ax_31 = ax_3.twinx()
    ax_31.set_ylabel('Alveoli', color='tab:green')
    ax_31.tick_params(axis='y', labelcolor='tab:green')

    minAlv = 0
    maxAlv = 0
    for key,value in results.items():
        if key.startswith('P_La'):
            if min(np.array(value[toIgone:])- atmPressure) < minAlv:
                minAlv = min(np.array(value[toIgone:])- atmPressure)
            if max(np.array(value[toIgone:])- atmPressure) > maxAlv:
                maxAlv = max(np.array(value[toIgone:])- atmPressure)


    minP = min(min(np.array(results['P_Ven'] - atmPressure)), minAlv)
    maxP = max(max(np.array(results['P_Ven'] - atmPressure)), maxAlv)
    
    ax_3.set_ylim([minP-1, maxP+1])
    ax_31.set_ylim([minP-1, maxP+1])

    results['V_La'] = volumeAlveoli
    results['V_Lb'] = volumeBronchi

    volumeThx = np.array(results['V_Vt'])+np.array(results['V_Hr'])+np.array(results['V_Ap'])+np.array(results['V_Cp'])+np.array(results['V_Vp'])+np.array(results['V_Hl'])+np.array(results['V_La'])+np.array(results['V_Lb'])
    ax_4 = fig.add_subplot(grid[-1, 0])
    results['V_Thx'] = volumeThx
    buildLeftRightAxisIndependentScales(ax_4,results,'V_La', 'V_Thx', 'Alveoli and Thorax Volumes [mL]','Alveoli','Thorax','tab:green','tab:blue',0.0,t)
    ax_4.set_xlabel('Time [s]')
    ##################################################################################################################################
    '''
    ax_2 = fig.add_subplot(grid[-1, 0])
    ax_2.set_title('Alveoli Compliance [mL/mmHg]')
    ax_2.plot(np.array(t),np.array(results['C_La']), color='tab:olive', label = 'Alveoli compliance')
    ax_2.set_yticks([min(np.array(results['C_La'])), max(np.array(results['C_La'])), np.round(np.mean(np.array(results['C_La'])),4)])
    ax_2.set_xlabel('Time [s]')
    '''
    ##################################################################################################################################
    #ax_8 = fig.add_subplot(grid[:2, 1])
    #plotDoubleSigmoidAxesTree(idx, structures, results, ax_8)

    ax_5 = fig.add_subplot(grid[2, 1])
    buildLeftRightAxisDependentScales(ax_5,results,'V_Lt', 'V_Lb', 'Trachea and Bronchi Volumes [mL]','Lt','Lb','tab:pink','tab:orange',0.0,t)
    ax_5.set_xlabel('Time [s]')

    ##################################################################################################################################
    ax_6 = fig.add_subplot(grid[3, 1])
    ax_6.set_title('Lung Flows [mL/s]')
    ax_6.plot(np.array(t[toIgone:]),np.array(results['Q_VenLt'][toIgone:]), color='tab:pink', label = 'Trachea')
    for key,value in results.items():
        if key.startswith('Q_Lt'):
            ax_3.plot(np.array(t[toIgone:]),np.array(value[toIgone:])- atmPressure, label = key)
        elif key.startswith('Q_Lb'):
            ax_3.plot(np.array(t[toIgone:]),np.array(value[toIgone:])- atmPressure, label = key)
    ax_6.set_yticks([0.0,min(np.array(results['Q_VenLt'][toIgone:])), max(np.array(results['Q_VenLt'][toIgone:]))])
    ax_6.legend(loc='upper right')
    
    ##################################################################################################################################
    ax_7 = fig.add_subplot(grid[4, 1])
    results['uVol'] = calculateThoraxUnstressedVolume(modelObjects,results)
    buildLeftRightAxisIndependentScales(ax_7,results,'P_Thx', 'uVol', 'Thoracic Pressure and Unstressed Volume','Pressure [mmHg]','V0 [mL]','tab:blue','tab:cyan',atmPressure,t)
    ax_7.set_xlabel('Time [s]')


    mpl.tight_layout()
    mpl.show()

def plotSigmoidsTree(results, totalTime, sampPeriod, atmPressure, modelObjects,idx,structures):
    toIgone = 2
    t = np.arange(0, totalTime, sampPeriod)
    
    counter = 0
    for key,value in results.items():
        if key.startswith('V_La'):
            counter = counter + 1
    rowCols = int(np.sqrt(counter))

    fig = mpl.figure(figsize=(15, 15))
    grid = gridspec.GridSpec(rowCols, rowCols, figure=fig)
    counter = idx
    for i in range(rowCols):
        for j in range(rowCols):
            ax = fig.add_subplot(grid[i, j])
            if counter < len(structures['modelStructure']['controllers']):
                plotDoubleSigmoidAxesTree(counter, structures, results, ax)
            ax.set_xlabel('Volume [mL]')
            counter = counter + 1

    mpl.tight_layout()
    mpl.show()

def plotPartialPressures(results, totalTime, sampPeriod, atmPressure):
    t = np.arange(0, totalTime, sampPeriod)

    fig = mpl.figure(figsize=(10, 6))
    grid = gridspec.GridSpec(4, 2, figure=fig)

    ####################################################################################################################
    ax_1 = fig.add_subplot(grid[:2, 0])
    ax_1.set_title('Pressures [mmHg]')
    ax_1.plot(np.array(t),np.array(results['P_O2_Cp']), label = 'P_O2_Cp')
    ax_1.plot(np.array(t),np.array(results['P_O2_Cs']), label = 'P_O2_Cs')
    ax_1.plot(np.array(t),np.array(results['P_O2_Vp']), label = 'P_O2_Vp')
    ax_1.plot(np.array(t),np.array(results['P_O2_Vs']), label = 'P_O2_Vs')
    ax_1.plot(np.array(t),np.array(results['P_O2_Tis']), label = 'P_O2_Tis')
    ax_1.plot(np.array(t),np.array(results['P_O2_La']), label = 'P_O2_La')
    ax_1.legend()


    ax_2 = fig.add_subplot(grid[2:, 0])
    ax_2.plot(np.array(t),np.array(results['P_C2_Cp']), label = 'P_C2_Cp')
    ax_2.plot(np.array(t),np.array(results['P_C2_Cs']), label = 'P_C2_Cs')
    ax_2.plot(np.array(t),np.array(results['P_C2_Vp']), label = 'P_C2_Vp')
    ax_2.plot(np.array(t),np.array(results['P_C2_Vs']), label = 'P_C2_Vs')
    ax_2.plot(np.array(t),np.array(results['P_C2_Tis']), label = 'P_C2_Tis')
    ax_2.plot(np.array(t),np.array(results['P_C2_La']), label = 'P_C2_La')
    ax_2.legend()


    ####################################################################################################################
    ax_3 = fig.add_subplot(grid[0, 1])
    ax_3.set_title('Pressures [mmHg]')
    ax_3.plot(np.array(t),np.array(results['P_O2_Lt']), label = 'P_O2_Lt')
    ax_3.plot(np.array(t),np.array(results['P_O2_Lb']), label = 'P_O2_Lb')
    ax_3.plot(np.array(t),np.array(results['P_O2_La']), label = 'P_O2_La')
    
    ax_3.plot(np.array(t),np.array(results['P_O2_Ven']), label = 'P_O2_Ven')
    ax_3.legend()
    ####################################################################################################################
    ax_5 = fig.add_subplot(grid[1, 1])
    ax_5.plot(np.array(t),np.array(results['P_C2_Lt']), label = 'P_C2_Lt')
    ax_5.plot(np.array(t),np.array(results['P_C2_Lb']), label = 'P_C2_Lb')
    ax_5.plot(np.array(t),np.array(results['P_C2_La']), label = 'P_C2_La')
    
    ax_5.plot(np.array(t),np.array(results['P_C2_Ven']), label = 'P_C2_Ven')
    ax_5.legend()
    ####################################################################################################################
    ax_6 = fig.add_subplot(grid[2, 1])
    ax_6.plot(np.array(t),np.array(results['P_N2_Lt']), label = 'P_N2_Lt')
    ax_6.plot(np.array(t),np.array(results['P_N2_Lb']), label = 'P_N2_Lb')
    ax_6.plot(np.array(t),np.array(results['P_N2_La']), label = 'P_N2_La')
    
    ax_6.plot(np.array(t),np.array(results['P_N2_Ven']), label = 'P_N2_Ven')
    ax_6.legend()
    ####################################################################################################################
    ax_4 = fig.add_subplot(grid[3, 1])
    ax_4.plot(np.array(t),np.array(results['Q_C2_CpLa']), color='tab:blue', label = 'Q_C2_CpLa')
    ax_4.set_ylabel('mL/s', color='tab:blue')
    ax_4.tick_params(axis='y', labelcolor='tab:blue')

    ax_41 = ax_4.twinx()
    ax_41.plot(np.array(t),np.array(results['Q_O2_CpLa']), color='tab:red', label = 'Q_O2_CpLa')
    ax_41.set_ylabel('mL/s', color='tab:red')
    ax_41.tick_params(axis='y', labelcolor='tab:red')

    lines1, labels1 = ax_4.get_legend_handles_labels()
    lines2, labels2 = ax_41.get_legend_handles_labels()
    ax_41.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax_4.set_xlabel('Time [s]')

    mpl.tight_layout()
    mpl.show()

def plotCardioVolumes(results, totalTime, sampPeriod, atmPressure):
    t = np.arange(0, totalTime, sampPeriod)

    fig = mpl.figure(figsize=(15, 6))
    grid = gridspec.GridSpec(4, 2, figure=fig)

    ####################################################################################################################
    ax_1 = fig.add_subplot(grid[0, 0])
    buildPressureAndVolumeAxis(ax_1,results,'P_Ap','V_Ap','Pulmonary Arteries (Ap)',atmPressure,t)

    ax_2 = fig.add_subplot(grid[1, 0])
    buildPressureAndVolumeAxis(ax_2,results,'P_Cp','V_Cp','Pulmonary Capillaries (Cp)',atmPressure,t)

    ax_3 = fig.add_subplot(grid[2, 0])
    buildPressureAndVolumeAxis(ax_3,results,'P_Vp','V_Vp','Pulmonary Veins (Vp)',atmPressure,t)

    ax_4 = fig.add_subplot(grid[3, 0])
    buildPressureAndVolumeAxis(ax_4,results,'P_Ap','V_Ap','Pulmonary Arteries (Ap)',atmPressure,t)

    ax_5 = fig.add_subplot(grid[0, 1])
    buildPressureAndVolumeAxis(ax_5,results,'P_As','V_As','Systemic Arteries (As)',atmPressure,t)

    ax_6 = fig.add_subplot(grid[1, 1])
    buildPressureAndVolumeAxis(ax_6,results,'P_Cs','V_Cs','Systemic Capillaries (Cs)',atmPressure,t)

    ax_7 = fig.add_subplot(grid[2, 1])
    buildPressureAndVolumeAxis(ax_7,results,'P_Vs','V_Vs','Systemic Veins (Vs)',atmPressure,t)

    ax_8 = fig.add_subplot(grid[3, 1])
    buildPressureAndVolumeAxis(ax_8,results,'P_Vt','V_Vt','Thoracic Veins (Vt)',atmPressure,t)

    ax_4.set_xlabel('Time [s]')
    ax_8.set_xlabel('Time [s]')

    ####################################################################################################################
    '''
    ax_5 = fig.add_subplot(grid[3, 0])
    ax_5.set_title('Ventricles (Hl,Hr)')
    ax_5.plot(np.array(t),np.array(results['P_Hl']) - atmPressure, color='tab:blue', label = 'P_Hl')
    ax_5.plot(np.array(t),np.array(results['P_Hr']) - atmPressure, color='tab:cyan', label = 'P_Hr')
    ax_5.set_ylabel('', color='tab:blue')
    ax_5.tick_params(axis='y', labelcolor='tab:blue')

    ax_51 = ax_5.twinx()
    ax_51.plot(np.array(t),np.array(results['V_Hl']), color='tab:red', label = 'V_Hl')
    ax_51.plot(np.array(t),np.array(results['V_Hr']), color='tab:pink', label = 'V_Hr')
    ax_51.set_ylabel('', color='tab:red')
    ax_51.tick_params(axis='y', labelcolor='tab:red')


    ax_5.set_yticks([
        np.round(min(np.array(results['P_Hl'])) - atmPressure,4), 
        np.round(max(np.array(results['P_Hl'])) - atmPressure,4),
        np.round(np.mean(np.array(results['P_Hl']) - atmPressure),4)
        ])
    ax_51.set_yticks([
        np.round(min(np.array(results['V_Hl'])),4),
        np.round(max(np.array(results['V_Hl'])),4),
        np.round(np.mean(np.array(results['V_Hl'])),4)
        ])


    
    ax_5.set_xlabel('Time [s]')
    '''

    mpl.tight_layout()
    mpl.show()

def plotControllers(results, totalTime, sampPeriod, atmPressure):
    t = np.arange(0, totalTime, sampPeriod)

    fig = mpl.figure(figsize=(12, 6))
    grid = gridspec.GridSpec(4, 2, figure=fig)

    volumeSystemic = np.array(results['V_Vt']) + np.array(results['V_As']) + np.array(results['V_Cs']) + np.array(results['V_Vs'])
    volumePulmonary = np.array(results['V_Ap']) + np.array(results['V_Cp']) + np.array(results['V_Vp'])
    volumeHeart = np.array(results['V_Hl']) + np.array(results['V_Hr'])

    ####################################################################################################################
    ax = fig.add_subplot(grid[0, 0])
    buildLeftRightAxisIndependentScales(ax,results,'E_C_Hl','E_C_Hr','Elastances [mmHg/mL]','Hl','Hr','tab:blue','tab:red',0.0,t)

    ax = fig.add_subplot(grid[1, 0])
    buildLeftRightAxisIndependentScales(ax,results,'R_AsCs','R_ApCp','Resistances [mmHg*s/mL]','R_AsCs','R_ApCp','tab:blue','tab:red',0.0,t)
    
    ax = fig.add_subplot(grid[2, 0])
    buildLeftRightAxisDependentScales(ax,results,'i_COl','i_COr','Stroke Volumes [mL]','i_COl','i_COr','tab:blue','tab:red',0.0,t)

    ax = fig.add_subplot(grid[0, 1])
    buildLeftRightAxisIndependentScales(ax,results,'C_As','C_Ap','Arterial Capacities [mL/mmHg]','Systemic','Pulmonary','tab:blue','tab:red',0.0,t)

    ax = fig.add_subplot(grid[1, 1])
    buildLeftRightAxisIndependentScales(ax,results,'C_Cs','C_Cp','Capillary Capacities [mL/mmHg]','Systemic','Pulmonary','tab:blue','tab:red',0.0,t)

    ax = fig.add_subplot(grid[2, 1])
    buildLeftRightAxisIndependentScales(ax,results,'C_Vs','C_Vp','Venous Capacities [mL/mmHg]','Systemic','Pulmonary','tab:blue','tab:red',0.0,t)

    ax = fig.add_subplot(grid[3, 1])
    buildLeftRightAxisIndependentScales(ax,results,'C_Vt','a_Vs','Venous Capacities [mL/mmHg]','Thorax','Mean Venous mmHg','tab:blue','tab:red',0.0,t)

    ax = fig.add_subplot(grid[3, 0])
    buildLeftRightAxisIndependentScales(ax,results,'e_C_Hl','e_C_Hr','Ventricle Capacities [mL/mmHg]','Left','Right','tab:blue','tab:red',0.0,t)



    


    mpl.tight_layout()
    mpl.show()



def compareRunsCardioPV(results, results1, modelObjects, totalTime, sampPeriod, atmPressure, title1, title2):
    # comparison plots
    t = np.arange(0, totalTime, sampPeriod)
    fig = mpl.figure(figsize=(15, 10))
    grid = gridspec.GridSpec(6, 2, figure=fig)

    ################################################################################################################################
    ax_1 = fig.add_subplot(grid[:-3, 0])
    heartPressureVolumeAxes(ax_1,results,title1,atmPressure)

    ax_2 = fig.add_subplot(grid[3, 0])
    buildLeftRightAxisDependentScales(ax_2,results,'i_COl','i_COr','Stroke Volumes [mL]','Hl','Hr','tab:blue','tab:red',0.0,t)
    ax_2.set_xlabel('Time [s]')

    ax_3 = fig.add_subplot(grid[4, 0])
    results['e_Hl'] = calculateElastance(modelObjects,results,'P_Hl')
    results['e_Hr'] = calculateElastance(modelObjects,results,'P_Hr')
    buildLeftRightAxisIndependentScales(ax_3,results,'e_Hl','e_Hr','Ventricle Compliances [mL/mmHg]','Hl','Hr','tab:blue','tab:red',0.0,t)
    ax_3.set_xlabel('Time [s]')

    ax_4 = fig.add_subplot(grid[5, 0])
    buildLeftRightAxisIndependentPressureScales(ax_4,results,'P_As','P_Ap','Blood Pressures [mmHg]','As','Ap','tab:green','tab:purple',atmPressure,t)
    ax_4.set_xlabel('Time [s]')

    ################################################################################################################################
    ax_5 = fig.add_subplot(grid[:-3, 1])
    heartPressureVolumeAxes(ax_5,results1,title2,atmPressure)

    ax_6 = fig.add_subplot(grid[3, 1])
    buildLeftRightAxisDependentScales(ax_6,results1,'i_COl','i_COr','Stroke Volumes [mL]','Hl','Hr','tab:blue','tab:red',0.0,t)
    ax_6.set_xlabel('Time [s]')

    ax_7 = fig.add_subplot(grid[4, 1])
    results1['e_Hl'] = calculateElastance(modelObjects,results1,'P_Hl')
    results1['e_Hr'] = calculateElastance(modelObjects,results1,'P_Hr')
    buildLeftRightAxisIndependentScales(ax_7,results1,'e_Hl','e_Hr','Ventricle Compliances [mL/mmHg]','Hl','Hr','tab:blue','tab:red',0.0,t)
    ax_7.set_xlabel('Time [s]')

    ax_8 = fig.add_subplot(grid[5, 1])
    buildLeftRightAxisIndependentPressureScales(ax_8,results1,'P_As','P_Ap','Blood Pressures [mmHg]','As','Ap','tab:green','tab:purple',atmPressure,t)
    ax_8.set_xlabel('Time [s]')

    '''
    ax_9 = fig.add_subplot(grid[6, 0])
    buildLeftRightAxisIndependentScales(ax_9,results,'V_Vt','V_Vp','Systemic Venous Volumes [mL]','Vt','Vp','tab:green','tab:purple',0.0,t)
    ax_9.set_xlabel('Time [s]')
    
    ax_10 = fig.add_subplot(grid[6, 1])
    buildLeftRightAxisIndependentScales(ax_10,results1,'V_Vt','V_Vp','Systemic Venous Volumes [mL]','Vt','Vp','tab:green','tab:purple',0.0,t)
    ax_10.set_xlabel('Time [s]')
    '''
    ################################################################################################################################
    mpl.tight_layout()
    mpl.show()

def compareRunsLungPV(results, results1, modelObjects, totalTime, sampPeriod, atmPressure, title1, title2):
    # comparison plots
    t = np.arange(0, totalTime, sampPeriod)
    fig = mpl.figure(figsize=(15, 10))
    grid = gridspec.GridSpec(7, 2, figure=fig)

    ################################################################################################################################
    ax_1 = fig.add_subplot(grid[:-4, 0])
    thoraxMouthPressureVolumeAxes(ax_1,results,atmPressure,sampPeriod,t)

    ax_2 = fig.add_subplot(grid[3, 0])
    buildLeftRightAxisIndependentScales(ax_2,results,'C_La', 'V_La', 'Alveoli Complaince and Volume','Compliance [mL/mmHg]','Volume [mL]','tab:olive','tab:green',0.0,t)
    ax_2.set_xlabel('Time [s]')

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
    ################################################################################################################################
    mpl.tight_layout()
    mpl.show()



def plotLungCollapseTest(results, totalTime, sampPeriod, atmPressure, modelObjects,idx, structures):
    toIgone = 2
    t = np.arange(0, totalTime, sampPeriod)
    
    # Create a figure and a 2x2 grid of subplots
    fig = mpl.figure(figsize=(7, 9))
    grid = gridspec.GridSpec(6, 1, figure=fig)
    volumeThx = np.array(results['V_Vt'])+np.array(results['V_Hr'])+np.array(results['V_Ap'])+np.array(results['V_Cp'])+np.array(results['V_Vp'])+np.array(results['V_Hl'])+np.array(results['V_La'])+np.array(results['V_Lb'])
    
    ax_1 = fig.add_subplot(grid[0, 0])
    ax_1.set_title('Lung Pressures [mmHg]')
    ax_1.plot(np.array(t[toIgone:]),np.array(results['P_Ven'][toIgone:]) - atmPressure, color='tab:blue', label = 'Ventilator')
    ax_1.plot(np.array(t[toIgone:]),np.array(results['P_Lt'][toIgone:]) - atmPressure, color='tab:pink', label = 'Trachea')
    ax_1.plot(np.array(t[toIgone:]),np.array(results['P_Lb'][toIgone:]) - atmPressure, color='tab:orange', label = 'Bronchi')
    ax_1.plot(np.array(t[toIgone:]),np.array(results['P_La'][toIgone:]) - atmPressure, color='tab:green', label = 'Alveoli') 
    ax_1.set_yticks([min(np.array(results['P_Ven'][toIgone:]) - atmPressure), max(np.array(results['P_Ven'][toIgone:]) - atmPressure), np.round(np.mean(np.array(results['P_Ven'][toIgone:]) - atmPressure),4)])
    ax_1.set_ylabel('Ventilator', color='tab:blue')
    ax_1.tick_params(axis='y', labelcolor='tab:blue')

    ax_11 = ax_1.twinx()
    ax_11.set_yticks([min(np.array(results['P_La'][toIgone:]) - atmPressure), max(np.array(results['P_La'][toIgone:]) - atmPressure), np.round(np.mean(np.array(results['P_La'][toIgone:]) - atmPressure),4)])
    ax_11.set_ylabel('Alveoli', color='tab:green')
    ax_11.tick_params(axis='y', labelcolor='tab:green')
    minP = min(min(np.array(results['P_Ven'][toIgone:] - atmPressure)), min(np.array(results['P_La'][toIgone:] - atmPressure)))
    maxP = max(max(np.array(results['P_Ven'][toIgone:] - atmPressure)), max(np.array(results['P_La'][toIgone:] - atmPressure)))
    ax_1.set_ylim([minP-1, maxP+1])
    ax_11.set_ylim([minP-1, maxP+1])

    ax_2 = fig.add_subplot(grid[1, 0])
    ax_2.set_title('Lung Flows [mL/s]')
    ax_2.plot(np.array(t[toIgone:]),np.array(results['Q_VenLt'][toIgone:]), color='tab:pink', label = 'Trachea')
    ax_2.plot(np.array(t[toIgone:]),np.array(results['Q_LtLb'][toIgone:]), color='tab:orange', label = 'Bronchi')
    ax_2.plot(np.array(t[toIgone:]),np.array(results['Q_LbLa'][toIgone:]), color='tab:green', label = 'Alveoli') 
    ax_2.set_yticks([0.0,min(np.array(results['Q_VenLt'][toIgone:])), max(np.array(results['Q_VenLt'][toIgone:]))])
    ax_2.legend(loc='upper right')

    ax_3 = fig.add_subplot(grid[2, 0])
    results['V_Thx'] = volumeThx
    buildLeftRightAxisIndependentScales(ax_3,results,'V_La', 'V_Thx', 'Alveoli and Thorax Volumes [mL]','Alveoli','Thorax','tab:green','tab:blue',0.0,t)
    ax_3.set_xlabel('Time [s]')
    
    #ax_4 = fig.add_subplot(grid[3, 0])
    #ax_4.set_title('Alveoli Compliance [mL/mmHg]')
    #ax_4.plot(np.array(t),np.array(results['C_La']), color='tab:olive', label = 'Alveoli compliance')
    #ax_4.set_yticks([min(np.array(results['C_La'])), max(np.array(results['C_La'])), np.round(np.mean(np.array(results['C_La'])),4)])
    #ax_4.set_xlabel('Time [s]')

    ax_4 = fig.add_subplot(grid[3, 0])
    buildLeftRightAxisIndependentScales(ax_4,results,'C_La','E_C_Thx','','Alveoli Compliance [mL/mmHg]','Max Thorax V0 [mL]','tab:blue','tab:red',0.0,t)

    ax_6 = fig.add_subplot(grid[4:, 0])
    plotDoubleSigmoidAxes(idx, structures, results, ax_6)
    
    ##################################################################################################################################
    #ax_7 = fig.add_subplot(grid[4, 0])
    #results['uVol'] = calculateThoraxUnstressedVolume(modelObjects,results)
    #buildLeftRightAxisIndependentScales(ax_7,results,'P_Thx', 'uVol', 'Thoracic Pressure and Unstressed Volume','Pressure [mmHg]','V0 [mL]','tab:blue','tab:cyan',atmPressure,t)
    #ax_7.set_xlabel('Time [s]')


    mpl.tight_layout()
    mpl.show()

def plotPEEPVentilatorTest(results, totalTime, sampPeriod, atmPressure, modelObjects, idx, structures):
    toIgone = 2
    t = np.arange(0, totalTime, sampPeriod)
    
    # Create a figure and a 2x2 grid of subplots
    fig = mpl.figure(figsize=(15, 12))
    grid = gridspec.GridSpec(5, 2, figure=fig)
    volumeThx = np.array(results['V_Vt'])+np.array(results['V_Hr'])+np.array(results['V_Ap'])+np.array(results['V_Cp'])+np.array(results['V_Vp'])+np.array(results['V_Hl'])+np.array(results['V_La'])+np.array(results['V_Lb'])
    
    
    ax_1 = fig.add_subplot(grid[0, 0])
    ax_1.set_title('Lung Pressures [mmHg]')
    ax_1.plot(np.array(t[toIgone:]),np.array(results['P_Ven'][toIgone:]) - atmPressure, color='tab:blue', label = 'Ventilator')
    ax_1.plot(np.array(t[toIgone:]),np.array(results['P_Lt'][toIgone:]) - atmPressure, color='tab:pink', label = 'Trachea')
    ax_1.plot(np.array(t[toIgone:]),np.array(results['P_Lb'][toIgone:]) - atmPressure, color='tab:orange', label = 'Bronchi')
    ax_1.plot(np.array(t[toIgone:]),np.array(results['P_La'][toIgone:]) - atmPressure, color='tab:green', label = 'Alveoli') 
    ax_1.set_yticks([min(np.array(results['P_Ven'][toIgone:]) - atmPressure), max(np.array(results['P_Ven'][toIgone:]) - atmPressure), np.round(np.mean(np.array(results['P_Ven'][toIgone:]) - atmPressure),4)])
    ax_1.set_ylabel('Ventilator', color='tab:blue')
    ax_1.tick_params(axis='y', labelcolor='tab:blue')

    ax_11 = ax_1.twinx()
    ax_11.set_yticks([min(np.array(results['P_La'][toIgone:]) - atmPressure), max(np.array(results['P_La'][toIgone:]) - atmPressure), np.round(np.mean(np.array(results['P_La'][toIgone:]) - atmPressure),4)])
    ax_11.set_ylabel('Alveoli', color='tab:green')
    ax_11.tick_params(axis='y', labelcolor='tab:green')
    minP = min(min(np.array(results['P_Ven'][toIgone:] - atmPressure)), min(np.array(results['P_La'][toIgone:] - atmPressure)))
    maxP = max(max(np.array(results['P_Ven'][toIgone:] - atmPressure)), max(np.array(results['P_La'][toIgone:] - atmPressure)))
    ax_1.set_ylim([minP-1, maxP+1])
    ax_11.set_ylim([minP-1, maxP+1])
    ax_1.set_xlabel('Time [s]')

    

    ax_2 = fig.add_subplot(grid[1, 0])
    results['V_Thx'] = volumeThx
    buildLeftRightAxisIndependentScales(ax_2,results,'V_La', 'V_Thx', 'Alveoli and Thorax Volumes [mL]','Alveoli','Thorax','tab:green','tab:blue',0.0,t)
    ax_2.set_xlabel('Time [s]')
    
    ax_3 = fig.add_subplot(grid[2, 0])
    results['amp_La'] = np.array(results['i_Max']) - np.array(results['i_Min'])
    buildLeftRightAxisIndependentScales(ax_3,results,'C_La','amp_La','Lung Compliance vs Tidal Volume','Alveoli Compliance [mL/mmHg]','Tidal Volume [mL]','tab:blue','tab:red',0.0,t)
    ax_3.set_xlabel('Time [s]')
    
    ax_4 = fig.add_subplot(grid[3, 0])
    buildLeftRightAxisDependentScales(ax_4,results,'i_COr','i_COl','Stroke Volumes [mL]','Right Ventricle','Left Ventricle','tab:blue','tab:red',0.0,t)
    ax_4.set_xlabel('Time [s]')

    ax_5 = fig.add_subplot(grid[4, 0])
    buildLeftRightAxisDependentScales(ax_5,results,'V_Hr','V_Hl','Ventricle Volumes [mL]','Right Ventricle','Left Ventricle','tab:blue','tab:red',0.0,t)
    ax_5.set_xlabel('Time [s]')

    ax_6 = fig.add_subplot(grid[0:2, 1])
    plotDoubleSigmoidAxes(idx, structures, results, ax_6)

    ax_7 = fig.add_subplot(grid[2, 1])
    results['volumeSytemic'] = np.array(results['V_As'])+np.array(results['V_Cs'])+np.array(results['V_Vs'])
    results['volumePulm'] = np.array(results['V_Vt'])+np.array(results['V_Hr'])+np.array(results['V_Ap'])+np.array(results['V_Cp'])+np.array(results['V_Vp'])+np.array(results['V_Hl'])
    buildLeftRightAxisIndependentScales(ax_7,results,'volumePulm','volumeSytemic','Thoracic vs Systemic Blood Volumes [mL]','Thoracic Blood','Systemic Blood','tab:blue','tab:red',0.0,t)
    ax_7.set_xlabel('Time [s]')

    ax_8 = fig.add_subplot(grid[3, 1])
    buildLeftRightAxisDependentPressureScales(ax_8,results,'P_Vp','P_Vt','Venous Inflow Pressure [mmHg]','Plumonary Vein','Systemic Vein','tab:blue','tab:red',760.0,t)
    ax_8.set_xlabel('Time [s]')

    ax_9 = fig.add_subplot(grid[4, 1])
    buildLeftRightAxisIndependentPressureScales(ax_9,results,'V_Vp','V_Vt','Venous Inflow Volume [mL]','Plumonary Vein','Systemic Vein', 'tab:blue','tab:red',0.0,t)
    ax_9.set_xlabel('Time [s]')



    mpl.tight_layout()
    mpl.show()




def plotcardioFlows(results, totalTime, sampPeriod, atmPressure,modelObjects):
    t = np.arange(0, totalTime, sampPeriod)

    # Create a figure and a 2x2 grid of subplots
    fig = mpl.figure(figsize=(15, 9))
    grid = gridspec.GridSpec(4, 2, figure=fig)
    #avgFlowThxSyst = (np.array(results['Q_AsCs'])+np.array(results['Q_CsVs'])+np.array(results['Q_VsVt'])+np.array(results['Q_VtHr'])+np.array(results['Q_HlAs']))/5
    #avgFlowThxPulm = (np.array(results['Q_HrAp'])+np.array(results['Q_ApCp'])+np.array(results['Q_CpVp'])+np.array(results['Q_VpHl']))/4

    avgFlowThxSyst = (np.array(results['Q_AsCs'])+np.array(results['Q_CsVs'])+np.array(results['Q_VsVt']))/3
    avgFlowThxPulm = (np.array(results['Q_ApCp'])+np.array(results['Q_CpVp']))/2
    avgTotal = (avgFlowThxSyst + avgFlowThxPulm)/2

    ax_1 = fig.add_subplot(grid[0, 0])
    ax_1.set_title('Cardion Flows [mL/s]')
    ax_1.plot(np.array(t),np.array(results['Q_AsCs']), color='tab:pink', label = 'AsCs')
    ax_1.legend(loc='upper right')
    ax_1.legend()

    ax_2 = fig.add_subplot(grid[1, 0])
    ax_2.set_title('Cardion Flows [mL/s]')
    ax_2.plot(np.array(t),np.array(results['Q_CsVs']), color='tab:orange', label = 'CsVs')
    ax_2.legend(loc='upper right')
    ax_2.legend()

    ax_3 = fig.add_subplot(grid[2, 0])
    ax_3.set_title('Cardion Flows [mL/s]')
    ax_3.plot(np.array(t),np.array(results['Q_VsVt']), color='tab:green', label = 'VsVt') 
    ax_3.legend(loc='upper right')
    ax_3.legend()

    ax_4 = fig.add_subplot(grid[3, 0])
    ax_4.set_title('Lung Flows [mL/s]')
    ax_4.plot(np.array(t),np.array(avgFlowThxSyst), color='tab:blue', label = 'Avg Systemic')
    ax_4.plot(np.array(t),np.array(avgFlowThxPulm), color='tab:green', label = 'Avg Pulmonary')
    ax_4.plot(np.array(t),np.array(avgTotal), color='tab:red', label = 'Avg Total')
    ax_4.legend()
    
    
    ax_5 = fig.add_subplot(grid[0, 1])
    ax_5.set_title('Pulmonary Flows [mL/s]')
    ax_5.plot(np.array(t),np.array(results['Q_ApCp']), color='tab:pink', label = 'ApCp')
    ax_5.legend(loc='upper right')
    ax_5.legend()
    
    ax_6 = fig.add_subplot(grid[1, 1])
    ax_6.set_title('Pulmonary Flows [mL/s]')
    ax_6.plot(np.array(t),np.array(results['Q_CpVp']), color='tab:orange', label = 'CpVp')
    ax_6.legend(loc='upper right')
    ax_6.legend()

    ax_7 = fig.add_subplot(grid[2, 1])
    buildLeftRightAxisDependentScales(ax_7,results,'Q_VtHr','Q_HlAs','Flows [mL/s]','VtHr','HlAs','tab:blue','tab:red',0.0,t)
    ax_7.set_xlabel('Time [s]')

    ax_8 = fig.add_subplot(grid[3, 1])
    buildLeftRightAxisDependentScales(ax_8,results,'Q_VpHl','Q_VtHr','Flows [mL/s]','VpHl','VtHr','tab:blue','tab:red',0.0,t)
    ax_8.set_xlabel('Time [s]')

    mpl.tight_layout()
    mpl.show()




def plotSigmoid(idx, structures, results):
    structure = structures['modelStructure']['controllers'][idx]
    maxValueToControl = float(structure[4])
    minValueToControl = float(structure[3])
    targetValue = float(structure[2])
    proportionalConstant = float(structure[5])
    V0 = np.arange(0,float(structure[2])*2,0.1)
    sigmoid = []
    for i in V0:
        amplitude = maxValueToControl - minValueToControl
        offset = minValueToControl
        inflectionPoint = targetValue
        slope = proportionalConstant

        sigmoid.append(amplitude / (1 + np.exp(-slope * (i - inflectionPoint))) + offset)

    mpl.plot(V0,sigmoid, color='tab:red', label = 'Sigmoid')
    #mpl.plot(np.array(results[structure[0]]), np.array(results[structure[1]]), color='tab:blue', label = structure[1])
    mpl.title('Sigmoid for the compliance: ' + structure[1])
    mpl.ylabel('Compliance [mL/mmHg]')
    mpl.xlabel('Volume [mL]')
    mpl.legend()
    mpl.tight_layout()
    mpl.show()

def plotDoubleSigmoid(idx, structures, results):
    structure = structures['modelStructure']['controllers'][idx]
    idxToIgnore = 10
    atmPressure = 760.0
    targetValue = float(structure[2])
    minValueToControl = float(structure[3])
    #maxValueToControl = float(structure[4])
    slope = float(structure[5])
    slope1 = float(structure[6])

    step = 1
    separation = 0.3
    
    duration = float(structure[2])*2
    V0 = np.arange(0,duration,step)
    sigmoid = []
    sigmoid1 = []

    pressure = []
    
    
    inflectionPoint = targetValue
    dist = inflectionPoint * separation
    
    
    cMin = 48.99615327695676
    offset = minValueToControl
    amplitude = cMin - minValueToControl
    
    threshold = 0000
    
    for i in V0:
        c = amplitude / (1 + np.exp(-slope * (i - inflectionPoint-dist))) + offset
        c1 = amplitude / (1 + np.exp(slope1 * (i - inflectionPoint+dist))) + offset + cMin
        sigmoid.append(c)
        sigmoid1.append(c1)

    difference = np.abs(np.array(sigmoid) - np.array(sigmoid1))
    intersection = np.where(difference == np.min(difference))[0][0]

    mergedSigmoids = sigmoid1[0:int(intersection)] + sigmoid[int(intersection)-1:-1]

    for i in V0:
        pressure.append(i/mergedSigmoids[int(i)])

    
    
    p= -np.array(results['P_La'][idxToIgnore:]- atmPressure) + np.array(results['P_Thx'][idxToIgnore:]- atmPressure)
    vol = np.array(results[structure[0]][idxToIgnore:])
    com = np.array(results[structure[1]][idxToIgnore:])
    
    fig = mpl.figure(figsize=(6, 6))
    grid = gridspec.GridSpec(1, 1, figure=fig)
    ax_1 = fig.add_subplot(grid[0, 0])

    #ax_1.plot(V0,sigmoid, color='tab:red', label = 'Sigmoid', linewidth=1)
    #ax_1.plot(V0,sigmoid1, color='tab:blue', label = 'Sigmoid1', linewidth=1)
    ax_1.plot(V0, mergedSigmoids, color='tab:green', label = 'Merged Sigmoids', linewidth=1.5)
    ax_1.plot(vol, com, color='tab:brown', label = structure[1], linewidth=2.5)
    ax_11 = ax_1.twinx()
    ax_11.plot(V0[0:5000],pressure[0:5000], color='tab:grey', label = 'Pressure', linewidth=1.5)
    ax_11.plot(vol, -p, color='tab:blue', label = '-(P_Thorax-P_Alveoli)', linewidth=2.5)
    
    
    ax_1.set_ylabel('Compliance [mL/mmHg]', color='tab:green')
    ax_1.tick_params(axis='y', labelcolor='tab:green')
    ax_1.set_xlabel('Volume [mL]')
    ax_11.set_xlabel('Volume [mL]')
    ax_11.set_ylabel('Pressure [mmhg]', color='tab:grey')
    ax_11.tick_params(axis='y', labelcolor='tab:grey')

    #ax_1.set_ylim(min(np.array(results[var1]) - atmPressure), max(np.array(results[var1]) - atmPressure))
    #ax_11.set_ylim(min(np.array(results[var2] - atmPressure)), max(np.array(results[var2] - atmPressure)))
   
    infPoint = int(inflectionPoint-dist)
    infPoint1 = int(inflectionPoint+dist)
    
    ax_1.set_xticks([
        0,
        int(min(vol)), 
        #np.round(np.mean(vol),4),
        infPoint,
        int(max(vol)), 
        int(intersection),
        infPoint1,
        7000,
        ])


    
    ax_1.set_yticks([
        mergedSigmoids[int(np.where(V0 == 0)[0])],
        mergedSigmoids[int(np.where(V0 == int(min(vol)))[0])], 
        mergedSigmoids[int(np.where(V0 == infPoint)[0])],
        mergedSigmoids[int(np.where(V0 == int(max(vol)))[0])], 
        mergedSigmoids[int(np.where(V0 == int(intersection))[0])],
        mergedSigmoids[int(np.where(V0 == infPoint1)[0])],
        mergedSigmoids[int(np.where(V0 == 4999)[0])],
        ])
    
    ax_11.set_xticks([
        0,
        int(min(vol)), 
        #np.round(np.mean(vol),4),
        infPoint,
        int(max(vol)), 
        int(intersection),
        infPoint1,
        7000,
        ])
    ax_11.set_yticks([
        pressure[int(np.where(V0 == 0)[0])],
        pressure[int(np.where(V0 == int(min(vol)))[0])], 
        pressure[int(np.where(V0 == infPoint)[0])],
        pressure[int(np.where(V0 == int(max(vol)))[0])], 
        pressure[int(np.where(V0 == int(intersection))[0])],
        pressure[int(np.where(V0 == infPoint1)[0])],
        pressure[int(np.where(V0 == 4999)[0])],
        ])
    ax_1.grid(True)
    #ax_1.grid(True)
    
    #minP = min(min(np.array(results[var1] - atmPressure)), min(np.array(results[var2] - atmPressure)))
    #maxP = max(max(np.array(results[var1] - atmPressure)), max(np.array(results[var2] - atmPressure)))
    
    #ax_1.set_ylim([minP -1, maxP+1])
    #ax_11.set_ylim([minP-1, maxP+1])
    
    
    
    
    
    mpl.title('Compliance Volume Curve of the Alveoli, Intersection: ' + str(intersection))
    mpl.legend()
    mpl.tight_layout()
    mpl.show()

    print(sigmoid1[0:2000:100])

def plotDoubleSigmoidColorsRC(idx, structures, results):
    structure = structures['modelStructure']['controllers'][idx]
    
    targetValue = float(structure[2])
    minValueToControl = float(structure[3])
    #maxValueToControl = float(structure[4])
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
    colorLength = 500
    #mpl.plot(V0,sigmoid, color='tab:red', label = 'Sigmoid', linewidth=1)
    #mpl.plot(V0,sigmoid1, color='tab:blue', label = 'Sigmoid1', linewidth=1)
    mpl.plot(V0, mergedSigmoids, color='k', label = 'Merged Sigmoids', linewidth=3)
    for j in range(0,len(np.array(results[structure[1]])),colorLength):
        mpl.plot(np.array(results[structure[0]][j:j+colorLength]), np.array(results[structure[1]][j:j+colorLength]), label = 'Run: ' + str((j/colorLength)+1), linewidth=15-((j*2)/colorLength))
    #mpl.title(structure[1] + ' Min: ' + str(minValueToControl) + ' Target: ' + str(targetValue) + 'Intersection: ' + str(intersection))
    mpl.title('Compliance Volume Curve of the Alveoli, Intersection: ' + str(intersection))
    mpl.ylabel('Compliance [mL/mmHg]')
    mpl.xlabel('Volume [mL]')
    mpl.legend()
    mpl.tight_layout()
    mpl.show()





#███    ██  ██████  ████████     ██ ███    ██     ██    ██ ███████ ███████ 
#████   ██ ██    ██    ██        ██ ████   ██     ██    ██ ██      ██      
#██ ██  ██ ██    ██    ██        ██ ██ ██  ██     ██    ██ ███████ █████   
#██  ██ ██ ██    ██    ██        ██ ██  ██ ██     ██    ██      ██ ██      
#██   ████  ██████     ██        ██ ██   ████      ██████  ███████ ███████ 

def plotTriggers(cycle,results, totalTime, sampPeriod, atmPressure):
    
    t = np.arange(0, totalTime, sampPeriod)
    #mpl.rcParams['figure.dpi'] = model.plotParameters['dpi']
    fig, ax = mpl.subplots(5, 1, constrained_layout=True)
    ax[0].set_title(cycle + ' Trigger Plot', size=7)
    ax[0].plot(np.array(t),np.array(results['T']), label = 'T')
    ax[0].plot(np.array(t),np.array(results['trigger' + cycle]), label = 'trigger' + cycle)
    ax[0].plot(np.array(t),np.array(results['T0']), label = 'T0')

    ax[1].plot(np.array(t),np.array(results[cycle]), label = cycle)

    ax[1].plot(np.array(t),np.array(results['timer' + cycle]), label = 'timer' + cycle)

    ax[2].plot(np.array(t),np.array(results['P_Thx']) - atmPressure, label = 'P_Thx')


    if cycle == 'RC':
        ax[3].plot(np.array(t),np.array(results['P_Lt']) - atmPressure, label = 'P_Lt')
        #ax[4].plot(np.array(t),np.array(results['V_Lt']), label = 'V_Lt')
        ax[3].plot(np.array(t),np.array(results['P_La']) - atmPressure, label = 'P_La')
        ax[4].plot(np.array(t),np.array(results['V_La']), label = 'V_La')
    elif cycle == 'HC':
        ax[3].plot(np.array(t),np.array(results['P_Hl']) - atmPressure, label = 'P_Hl')
        ax[3].plot(np.array(t),np.array(results['P_Hr']) - atmPressure, label = 'P_Hr')
        ax[4].plot(np.array(t),np.array(results['V_Hl']), label = 'V_Hl')
        ax[4].plot(np.array(t),np.array(results['V_Hr']), label = 'V_Hr')

    ax[4].set_xlabel('Time [s]')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()
    ax[4].legend()
    mpl.show()


#████████ ███████ ███████ ████████     ██████  ██       ██████  ████████ ███████ 
#   ██    ██      ██         ██        ██   ██ ██      ██    ██    ██    ██      
#   ██    █████   ███████    ██        ██████  ██      ██    ██    ██    ███████ 
#   ██    ██           ██    ██        ██      ██      ██    ██    ██         ██ 
#   ██    ███████ ███████    ██        ██      ███████  ██████     ██    ███████ 


def plotGroup(var,results, totalTime, sampPeriod, atmPressure):
    for key,value in results.items():
        for key1 in var.keys():
            if key == key1:
                mpl.plot(np.array(results['T']),np.array(value), label = key)
    mpl.xlabel('Time (s)', fontsize = 7)
    #mpl.ylabel(varToPlot, fontsize = 7)
    mpl.legend()
    mpl.show()

def plotVar(var,results, totalTime, sampPeriod, atmPressure):
    mpl.plot(np.array(results['T']),np.array(results[var]), label = var)
    mpl.xlabel('Time (s)', fontsize = 7)
    #mpl.ylabel(varToPlot, fontsize = 7)
    mpl.legend()
    mpl.show()

def plotVars(var,results, totalTime, sampPeriod, atmPressure):
    for key in var:
        mpl.plot(np.array(results['T']),np.array(results[key]), label = key)
    mpl.xlabel('Time (s)', fontsize = 7)
    #mpl.ylabel(varToPlot, fontsize = 7)
    mpl.legend()
    mpl.show()

def plotPV(var,results, totalTime, sampPeriod, atmPressure):
    mpl.plot(np.array(results['P_' + var]),np.array(results['V_' + var]), label = var)
    mpl.xlabel('Pressure [mmHg]', fontsize = 7)
    mpl.ylabel('Volume [mL]', fontsize = 7)
    mpl.legend()
    mpl.show()


