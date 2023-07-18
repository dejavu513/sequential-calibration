# -*- coding: utf-8 -*-  'bright' for the small banners
"""
Created on Thu Mar 10 12:12:17 2022

@author: mtc86
"""
import numpy as np
import treeClass as tree

import matplotlib.pyplot as mpl
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import axes3d
from operator import add
from colorama import Fore, Back, Style
import json
import sys
import traceback

#import math as m

def normalizeArray(arr):
    maxArr = max(arr)
    minArr = min(arr)
    newArr = np.round((arr-minArr)/(maxArr-minArr),2)
    return newArr

# TODO parse the solutions and separete the lung data from the circulatory data    
def parseResults(modelObject):
    results = modelObject.solution
    
    volumes = results[0]
    pressures = results[1]
    currents = results[2]
    
    volNames = np.array(list(volumes.keys()))
    sizes = np.char.str_len(volNames)
    volMaxLen = max(sizes)
    
    preNames = np.array(list(pressures.keys()))
    sizes = np.char.str_len(preNames)
    preMaxLen = max(sizes)
    
    curNames = np.array(list(currents.keys()))
    sizes = np.char.str_len(curNames)
    curMaxLen = max(sizes)
    
    heartVols = []
    heartPressures = []
    heartCurrents = []
    
    lungVols = []
    lungPressures = []
    lungCurrents = []
    
    pulmCircVols = []
    pulmCircPressures = []
    pulmCircCurrents = []
    for idx,name in enumerate(volNames):
        pass
    
    return True

# Calculates the variation of the tolat volume of the system (estimates the breath volume)
def calculateVolumeVariaton(results):
    values = []
    for key, value in results.items():
        if 'Lu' in key:
            pass
        else:
            values.append(value)
    values = np.array(values)
    
    cumS = []
    for idx, value in enumerate(values.T):
        cumS.append(np.sum(value))
    return cumS

#  __  __           _      _   ____       _               
# |  \/  | ___   __| | ___| | / ___|  ___| |_ _   _ _ __  
# | |\/| |/ _ \ / _` |/ _ \ | \___ \ / _ \ __| | | | '_ \ 
# | |  | | (_) | (_| |  __/ |  ___) |  __/ |_| |_| | |_) |
# |_|  |_|\___/ \__,_|\___|_| |____/ \___|\__|\__,_| .__/ 
#                                                  |_| 
                                                 
def buildLungTree(params,tag):       
    ##################### Setup the model #########################################
    if params['gasExchange']:
        # Creates the first node of the tree
        trachea = tree.baseNode(
            params['start_coordinate'], params['length'], params['angle'], 
            params['radius'], 0, 0, params['inputName'], tag, params['asymmetry'], params['plane'],
            params['partialPressures'], params['partialPressuresY0']
            )
    else:
        trachea = tree.baseNode(
            params['start_coordinate'], params['length'], params['angle'], 
            params['radius'], 0, 0, params['inputName'], tag, params['asymmetry'], params['plane'],
            [], []
              )
    
    # Builds the alveolar tree
    if len(params['start_coordinate']) == 2:
        trachea.buildTree(params['maxLevels'], params['branchAngle'], params['lengthDecay'])
    else:
        trachea.build3DTree(params['maxLevels'], params['branchAngle'], params['lengthDecay'])

    # Gets the tree parameters as a list
    treeArray = trachea.getTreeArray()
    
    return treeArray

def buildCapilaryTree(params,tag,inputName,gasY0):
    
    ##################### Setup the model #########################################
    if params['gasExchange']:
        # Creates the first node of the tree
        artery = tree.baseNode(
            params['start_coordinate'], params['length'], params['angle'], 
            params['radius'], 0, 0, inputName, tag, params['asymmetry'], params['plane'],
            params['partialPressures'], gasY0
            )
    else:
        artery = tree.baseNode(
            params['start_coordinate'], params['length'], params['angle'], 
            params['radius'], 0, 0, inputName, tag, params['asymmetry'], params['plane'],
            [], []#params['partialPressures'], params['partialPressuresY0']
            )
    # Builds the alveolar tree
    artery.buildTree(params['maxLevels'], params['branchAngle'], params['lengthDecay'])

    # Gets the tree parameters as a list
    treeArray = artery.getTreeArray()
    
    return treeArray

 
#  _____ _           _                              _   _               _     
# |_   _(_)_ __ ___ (_)_ __   __ _   _ __ ___   ___| |_| |__   ___   __| |___ 
#   | | | | '_ ` _ \| | '_ \ / _` | | '_ ` _ \ / _ \ __| '_ \ / _ \ / _` / __|
#   | | | | | | | | | | | | | (_| | | | | | | |  __/ |_| | | | (_) | (_| \__ \
#   |_| |_|_| |_| |_|_|_| |_|\__, | |_| |_| |_|\___|\__|_| |_|\___/ \__,_|___/
#                            |___/                                            

def printTimes (times):
    print(Fore.GREEN + 'TreeBuildTime: ' + str(times['treeBuildTime'] - times['startTime']))
    print(Fore.GREEN + 'systemBuildTime: ' + str(times['systemBuildTime'] - times['treeBuildTime']))
    print(Fore.GREEN + 'systemSimulationTime: ' + str(times['systemSimulationTime'] - times['systemBuildTime']))
    print(Fore.GREEN + 'systemPlottingTime: ' + str(times['systemPlottingTime'] - times['systemSimulationTime']))
    print(Back.WHITE + Fore.GREEN + 'Total Time: ' + str(times['systemPlottingTime'] - times['startTime'])) 
    print(Style.RESET_ALL)


def printException(message):
    # Get current system exception
    ex_type, ex_value, ex_traceback = sys.exc_info()

    # Extract unformatter stack traces as tuples
    trace_back = traceback.extract_tb(ex_traceback)

    # Format stacktrace
    stack_trace = list()

    for trace in trace_back:
        stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))
    
    print(Fore.RED + message)
    print(Fore.RED + "Exception type : %s " % ex_type.__name__)
    print(Fore.RED + "Exception message : %s" %ex_value)
    print(Fore.RED + "Stack trace : %s" %stack_trace)
    print(Style.RESET_ALL)


#  _   _       _ _                                       _             
# | | | |_ __ (_) |_    ___ ___  _ ____   _____ _ __ ___(_) ___  _ __  
# | | | | '_ \| | __|  / __/ _ \| '_ \ \ / / _ \ '__/ __| |/ _ \| '_ \ 
# | |_| | | | | | |_  | (_| (_) | | | \ V /  __/ |  \__ \ | (_) | | | |
#  \___/|_| |_|_|\__|  \___\___/|_| |_|\_/ \___|_|  |___/_|\___/|_| |_|
                                                                      
def pa2mmHg (value):
    return value * 0.0075006156130264

def mmHg2Pa (value):
    return value * 133.3223900000007

def calculateResistorValue(length, radius, viscosity):
    result = (8*viscosity*length)/((radius**4)*np.pi)
    return result

def calculateCapacitorValue(length, radius, E, thickness):
    result = (3*length*(radius**3)*np.pi)/(2*E*thickness)
    print(result)
    return result

def calculateInductorValue(length, radius, density):
    result = (density*length)/(np.pi*(radius**2))
    return result

#  __  __           _      _      _                __  __      _   _               _     
# |  \/  | ___   __| | ___| |    / \  _   ___  __ |  \/  | ___| |_| |__   ___   __| |___ 
# | |\/| |/ _ \ / _` |/ _ \ |   / _ \| | | \ \/ / | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
# | |  | | (_) | (_| |  __/ |  / ___ \ |_| |>  <  | |  | |  __/ |_| | | | (_) | (_| \__ \
# |_|  |_|\___/ \__,_|\___|_| /_/   \_\__,_/_/\_\ |_|  |_|\___|\__|_| |_|\___/ \__,_|___/                                                                                   

# returns true if the component has a 'Lu' on the Id -> used on filter methods
def checkLungComponent(component):
    if 'Lu' in component['id']:
        return True
    else:
        return False

'''
███    ███  ██████  ██████  ███████ ██          ██████  ██       ██████  ████████ ███████ 
████  ████ ██    ██ ██   ██ ██      ██          ██   ██ ██      ██    ██    ██    ██      
██ ████ ██ ██    ██ ██   ██ █████   ██          ██████  ██      ██    ██    ██    ███████ 
██  ██  ██ ██    ██ ██   ██ ██      ██          ██      ██      ██    ██    ██         ██ 
██      ██  ██████  ██████  ███████ ███████     ██      ███████  ██████     ██    ███████ 
'''

# Plots the tree represented as a 'lung'                                                                   
def plotSimpleModelTree(compartments, options): 
   
    lung = list(filter(checkLungComponent,compartments))
    treeDesc = [d['rep'] for d in lung] 
        
    mpl.figure();
    for node in treeDesc:
        startXY = node['startXY']
        endXY = node['endXY']
        isLeaf = node['isLeaf']
        startEnd = np.array([startXY , endXY])
        mpl.plot(startEnd[:,0] , startEnd[:,1], color='blue')
        if isLeaf == True:
            mpl.scatter(startEnd[1,0] , startEnd[1,1] , color='red')
    if options['save']:
        mpl.savefig('Lung.png', dpi=1500)
    
    mpl.show()
    
# TODO Check this plot for errors
def plotSimple3DModelTree(compartments, options):
    lung = list(filter(checkLungComponent,compartments))
    treeDesc = [d['rep'] for d in lung] 
    
    sSize = 2.5
    lw = 0.5
      
    mpl.figure();
    ax = mpl.axes(projection='3d')
    for node in treeDesc:
        startXY = node['startXY']
        endXY = node['endXY']
        isLeaf = node['isLeaf']
        startEnd = np.array([startXY , endXY])
        ax.plot3D(startEnd[:,0] , startEnd[:,1], startEnd[:,2], linewidth=lw , color='blue')
        if isLeaf == True:
            ax.scatter3D(startEnd[1,0] , startEnd[1,1] , startEnd[1,2] , s=sSize , color='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')  
    ax.set_xlim(-0.15, 0.15)
    ax.set_ylim(-0.15, 0.15)
    ax.set_zlim(-0.3, 0)
    
    #mpl.savefig('Lung3D.png', dpi=1500)

    mpl.figure();
    for node in treeDesc:
        startXY = node['startXY']
        endXY = node['endXY']
        isLeaf = node['isLeaf']
        startEnd = np.array([startXY , endXY])
        ##############################
        #mpl.subplot(1,3,1, aspect='equal')
        mpl.plot(startEnd[:,1] , startEnd[:,0], linewidth=lw, color='blue')
        if isLeaf == True:
            mpl.scatter(startEnd[1,1] , startEnd[1,0] , s=sSize , color='red')
        mpl.title('XY (Top)')
        mpl.xlim([-0.15, 0.15])
        mpl.ylim([-0.15, 0.15])
        ##############################
    #mpl.savefig('LungTop.png', dpi=1500)
        
    mpl.figure();
    for node in treeDesc:
        startXY = node['startXY']
        endXY = node['endXY']
        isLeaf = node['isLeaf']
        startEnd = np.array([startXY , endXY])    
        
        #mpl.subplot(1,3,2, aspect='equal')
        mpl.plot(startEnd[:,1] , startEnd[:,2], linewidth=lw, color='blue')
        if isLeaf == True:
            mpl.scatter(startEnd[1,1] , startEnd[1,2] , s=sSize , color='red')
        mpl.title('YZ (Front)')
        mpl.xlim([-0.1, 0.1])
        mpl.ylim([-0.25,-0.05])
        ##############################
    #mpl.savefig('LungFront.png', dpi=1500)    
        
    mpl.figure();
    for node in treeDesc:
        startXY = node['startXY']
        endXY = node['endXY']
        isLeaf = node['isLeaf']
        startEnd = np.array([startXY , endXY])
        #mpl.subplot(1,3,3, aspect='equal')
        mpl.plot(startEnd[:,0] , startEnd[:,2], linewidth=lw, color='blue')
        if isLeaf == True:
            mpl.scatter(startEnd[1,0] , startEnd[1,2] , s=sSize , color='red')
        mpl.title('XZ (Side)')
        mpl.xlim([-0.15, 0.15])
        mpl.ylim([-0.3,0]) 
        
    if options['save']:
        mpl.savefig('Lung.png', dpi=1500)

# TODO fix plot -> ['rep'] no longer contains the value for R and C            
def plotModelTree(compartments, options):
    lung = list(filter(checkLungComponent,compartments))
    treeDesc = [d['rep'] for d in lung] 
    
    rB = np.zeros(0)
    cB = np.zeros(0)
    rA = np.zeros(0)
    cA = np.zeros(0)
        
    for i in np.arange(0,len(treeDesc),1)-1:
        if treeDesc[i]['isLeaf'] == True:
            rA = np.append(rA , treeDesc[i]['r'])
            cA = np.append(cA , treeDesc[i]['c'])  
        else:
            rB = np.append(rB , treeDesc[i]['r'])
            cB = np.append(cB , treeDesc[i]['c']) 

    colormapRB = mpl.get_cmap('winter') 
    cNormRB  = colors.LogNorm(vmin=min(rB), vmax=max(rB))
    scalarMapRB = cmx.ScalarMappable(norm=cNormRB, cmap=colormapRB)

    #colormapCB = mpl.get_cmap('Reds') 
    #cNormCB  = colors.Normalize(vmin=min(cB), vmax=max(cB))
    #scalarMapCB = cmx.ScalarMappable(norm=cNormCB, cmap=colormapCB)

    colormapRA = mpl.get_cmap('spring') 
    cNormRA  = colors.Normalize(vmin=min(rA), vmax=max(rA))
    scalarMapRA = cmx.ScalarMappable(norm=cNormRA, cmap=colormapRA)

    #colormapCA = mpl.get_cmap('Purples') 
    #cNormCA  = colors.Normalize(vmin=min(cA), vmax=max(cA))
    #scalarMapCA = cmx.ScalarMappable(norm=cNormCA, cmap=colormapCA)
    fig = mpl.figure()
    ax = fig.add_subplot(111)
    
    for  i in np.arange(0,len(treeDesc),1)-1:
        startEnd = np.array([treeDesc[i]['startXY'] , treeDesc[i]['endXY']])
        
        if treeDesc[i]['isLeaf'] == True:
            colorValR = scalarMapRA.to_rgba(treeDesc[i]['r'])
            #colorValC = scalarMapCA.to_rgba(treeDesc[i]['c'])
            ax.plot(startEnd[:,0] , startEnd[:,1] , linewidth=0.5 , color='black')
            ax.scatter(startEnd[1,0] , startEnd[1,1] , s=treeDesc[i]['c']*4000 , color=colorValR)
        else:
            colorValR = scalarMapRB.to_rgba(treeDesc[i]['r'])
            #colorValC = scalarMapCB.to_rgba(treeDesc[i]['c'])
            ax.scatter(startEnd[0,0] , startEnd[0,1] , s=treeDesc[i]['c']*4000 , color=colorValR)
            ax.plot(startEnd[:,0] , startEnd[:,1] , linewidth=0.5 , color='black')

    mpl.colorbar(scalarMapRB , location='left')
    #mpl.colorbar(scalarMapCB , location='bottom' , fraction=0.025)
    mpl.colorbar(scalarMapRA , location='right')
    #mpl.colorbar(scalarMapCA , location='bottom' , fraction=0.025)
    if options['save']:
        mpl.savefig('LungColor.png', dpi=1500)



'''
.#####...##..##..##......##...##...####...##..##...####...#####...##..##.
.##..##..##..##..##......###.###..##..##..###.##..##..##..##..##...####..
.#####...##..##..##......##.#.##..##..##..##.###..######..#####.....##...
.##......##..##..##......##...##..##..##..##..##..##..##..##..##....##...
.##.......####...######..##...##...####...##..##..##..##..##..##....##...
'''

# Sumarises all pulmunary system
def plotLungSinglePlot(modelObject, options):
    t = modelObject.simulation['t']
    result = modelObject.solution
    
    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 5,
        }
    
    grid = {
        'top': 1, 
        'bottom': 0, 
        'right': 1, 
        'left': 0,
        'hspace': 0, 
        'wspace': 0.07
        }

    lWidth = options['lWidth']
    legendSize = options['legendSize']
    idxsToIgnore = options['idxsToIgnore']
    
    keys = np.array(list(result[0].keys()))
    sizes = np.char.str_len(keys)
    maxLen = max(sizes)
   
    mpl.rcParams['figure.dpi'] = options['dpi']
    #fig, ax = mpl.subplots(2, 2, gridspec_kw = grid)
    fig, ax = mpl.subplots(2, 2, constrained_layout=True)
    
    [axi.xaxis.set_visible(False) for axi in ax.ravel()]
    [axi.tick_params(axis='both',labelsize=4, pad=1, direction='in',grid_linewidth = 1, labelrotation = 15) for axi in ax.ravel()]
    
    phaseChanges = np.where(np.roll(result[7]['resp'],1)!=result[7]['resp'])
    rectangles = np.zeros([int(len(phaseChanges[0])/2),2])
    counter = 0
    for idx,val in enumerate(phaseChanges[0]):
        if (idx % 2 == 0):
            rectangles[counter,0] = val
        else:
            rectangles[counter,1] = val - rectangles[counter,0]
            counter = counter + 1
    for rec in rectangles:
        start = int(rec[0])
        end = int(rec[0]+rec[1])
        ax[0,0].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[1,0].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[0,1].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[1,1].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
    
    meanVols = {}
    ampVols = {}
    meanPressure = {}
    ampPressure = {}
    maxPressure = {}
##### Volume Plot #############################################################
    tot = 0
    for key, value in result[0].items():
        if (len(key) == maxLen) and ('Lu' in key):
            meanVols[key] = str(int(np.round(np.mean(value[idxsToIgnore:-1]),1)))
            ampVols[key] = str(np.round(max(value[idxsToIgnore:-1])-min(value[idxsToIgnore:-1]),1))
            #ax[0,0].plot(t[3:-1], value[3:-1], label = key + ' -> ' + meanVols[key] + ' -> ' + ampVols[key], linewidth=lWidth)
            ax[0,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key+ ' -> ' + meanVols[key] + ' -> ' + ampVols[key], linewidth=lWidth)
            tot = tot + value[-1]
    

##### Pressure Plot ###########################################################
    for key, value in result[1].items():
        if (('Lu' in key) or ('u' == key)) and ('2' not in key):
            ampPressure[key] = str(np.round(max(value[idxsToIgnore:-1])-min(value[idxsToIgnore:-1])))
            maxPressure[key] = str(np.round(max(value[idxsToIgnore:-1])))
            meanPressure[key] = str(int(np.round(np.mean(value[idxsToIgnore:-1]))))
            #ax[0,1].plot(t[3:-1], value[3:-1], label = key + ' -> ' + meanPressure[key] + ' -> ' + ampPressure[key] + ' -> ' + maxPressure[key], linewidth=lWidth)
            ax[0,1].plot(t[idxsToIgnore:-1], (value[idxsToIgnore:-1] - modelObject.lungParams['atmosphericPressure']), label = key, linewidth=lWidth)
        
    #ax2.legend(prop={'size': legendSize})
    
##### Current Plot ############################################################
    zeros = np.zeros(len(t))
    for key, value in result[2].items():
        if ('Lu' in key) and (('2' not in key)):
            #ax[1,0].plot(t[3:-1], value[3:-1], label = key + ' -> ' + str(int(np.round(np.mean(value)))), linewidth=lWidth)
            ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth)
    ax[1,0].plot(t[idxsToIgnore:-1], zeros[idxsToIgnore:-1], color='b', linewidth=lWidth)
    
    
##### Trachea Bronc Plot ######################################################   
    for key, value in result[0].items():
        mean = str(int(np.round(np.mean(value[idxsToIgnore:-1]),1)))
        amp = str(np.round(max(value[idxsToIgnore:-1])-min(value[idxsToIgnore:-1]),1))
        if (len(key) < maxLen) and ('Lu' in key):
            ax[1,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key + ' -> ' + mean + ' -> ' + amp, linewidth=lWidth)
            tot = tot + value[-1]
    
    if (options['legend']):
        ax[0,0].legend(prop={'size': legendSize},facecolor='white', framealpha=1)
        ax[1,0].legend(prop={'size': legendSize},facecolor='white', framealpha=1)
        ax[0,1].legend(prop={'size': legendSize},facecolor='white', framealpha=1)
        ax[1,1].legend(prop={'size': legendSize},facecolor='white', framealpha=1)
        #fig.suptitle('Total Volume' + str(np.round(tot,1)))
    else:
        pass
    
    if options['printData']:
        print(Style.RESET_ALL)
        print('Mean volume (mL): ' + json.dumps(meanVols, indent=4))
        print('Amplitude of volume (mL): ' + json.dumps(ampVols, indent=4))
        #print('Mean Pressure (mmHg): ' + json.dumps(meanPressure, indent=4))
        #print('Pressure amplitude (mmHg): ' + json.dumps(ampPressure, indent=4))
        #print('Systolic Pressure (mmHg): ' + json.dumps(maxPressure, indent=4))
        print('Total Volume (mL) is: ' + str(np.round(tot,1)))

    ax[0,0].set_xlabel('Time (s)', fontsize = 5)
    ax[0,0].set_ylabel('Volume (mL)', fontsize = 5)
    ax[0,0].set_title('Volume on the Alveoli', size=7)

    ax[0,1].set_xlabel('Time (s)', fontsize = 5)
    ax[0,1].set_ylabel('Pressure (mmHg)', fontsize = 5)
    ax[0,1].set_title('Pulmonary Pressures', size=7) 

    ax[1,0].set_xlabel('Time (s)', fontsize = 5)
    ax[1,0].set_ylabel('Flow (mL/s)', fontsize = 5)
    ax[1,0].set_title('Pulmonary Flows', size=7) 

    ax[1,1].set_xlabel('Time (s)', fontsize = 5)
    ax[1,1].set_ylabel('Volume (mL)', fontsize = 5)
    ax[1,1].set_title('Volumes on the Bronchia Tree', size=7) 

    mpl.show()

# Filters out the circulatory components and generates 4 plots
# with volume on the alveoli/brinchia pressures and currents         
def plotLungResults(modelObject, options):
    t = modelObject.simulation['t']
    results = modelObject.solution
    
    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 5,
        }
    
    grid = {
        'top': 1, 
        'bottom': 0, 
        'right': 1, 
        'left': 0,
        'hspace': 0, 
        'wspace': 0.07
        }

    lWidth = options['lWidth']
    legendSize = options['legendSize'] 
    idxsToIgnore = options['idxsToIgnore']

    keys = np.array(list(results[0].keys()))
    sizes = np.char.str_len(keys)
    maxLen = max(sizes)
    
    
    # coolwarm
    colormapRA = mpl.get_cmap('hot') 
    cNormRA  = colors.Normalize(vmin=2, vmax=18)
    scalarMapRA = cmx.ScalarMappable(norm=cNormRA, cmap=colormapRA)
    
    #mpl.figure()
    mpl.rcParams['figure.dpi'] = options['dpi']
    fig2, ([ax1, ax2, ax3, ax4]) = mpl.subplots(4, 1, gridspec_kw = grid,)
    
    phaseChanges = np.where(np.roll(results[7]['resp'],1)!=results[7]['resp'])
    rectangles = np.zeros([int(len(phaseChanges[0])/2),2])
    counter = 0
    for idx,val in enumerate(phaseChanges[0]):
        if (idx % 2 == 0):
            rectangles[counter,0] = val
        else:
            rectangles[counter,1] = val - rectangles[counter,0]
            counter = counter + 1
    for rec in rectangles:
        start = int(rec[0])
        end = int(rec[0]+rec[1])
        ax1.axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax2.axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax3.axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax4.axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
    
    for key, value in results[0].items():
        if 'Lu' in key: 
            if len(key) == maxLen:               
                ax1.plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth)
            else:
                colorValR = scalarMapRA.to_rgba(len(key))                 
                ax2.plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color=colorValR)
    
    for key, value in results[1].items():
        if 'Lu' in key: 
            colorValR = scalarMapRA.to_rgba(len(key))                 
            ax3.plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1] - modelObject.lungParams['atmosphericPressure'], label = key, linewidth=lWidth, color=colorValR)
    
    for key, value in results[2].items():
        if ('Lu' in key) and ('2' not in key): 
            colorValR = scalarMapRA.to_rgba(len(key))                 
            ax4.plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color=colorValR)

# Plots the changes in total volume of the system (represents breadth volumes)                                                                              
def plotVolumeChanges(modelObject, options): 
    
    t = modelObject.simulation['t']
    result = modelObject.solution
    
    lWidth = options['lWidth']
    legendSize = options['legendSize']
    
    ########################################################################################
    
    mpl.figure()
    mpl.rcParams['figure.dpi'] = options['dpi']
    values = []
    for key, value in result[0].items():
        if 'Lu' in key:
            values.append(value)
    values = np.array(values)
    
    cumS = []
    for idx, value in enumerate(values.T):
        cumS.append(np.sum(value))
    mpl.plot(cumS-min(cumS), linewidth=lWidth)
    mpl.title('total Volume Changes: max: ' + str(round(max(cumS-min(cumS)))))


'''
..####....####...#####...#####...######...####...##..##...####....####....####...##..##..##.......####...#####..
.##..##..##..##..##..##..##..##....##....##..##..##..##..##..##..##......##..##..##..##..##......##..##..##..##.
.##......######..#####...##..##....##....##..##..##..##..######...####...##......##..##..##......######..#####..
.##..##..##..##..##..##..##..##....##....##..##...####...##..##......##..##..##..##..##..##......##..##..##..##.
..####...##..##..##..##..#####...######...####.....##....##..##...####....####....####...######..##..##..##..##.
'''
                                                    
# Plot that summarises all the circulatory system
def plotHeartResults(modelObject, options):
    t = modelObject.simulation['t']
    results = modelObject.solution
    
    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 5,
        }
    
    grid = {
        'top': 1, 
        'bottom': 0,
        'right': 1, 
        'left': 0,
        'hspace': 0, 
        'wspace': 0.07
        }

    lWidth = options['lWidth']
    legendSize = options['legendSize'] 
    idxsToIgnore = options['idxsToIgnore']

    keys = np.array(list(results[0].keys()))
    
    
    # coolwarm
    colormapRA = mpl.get_cmap('hot') 
    cNormRA  = colors.Normalize(vmin=2, vmax=18)
    scalarMapRA = cmx.ScalarMappable(norm=cNormRA, cmap=colormapRA)
    
    
    bloodTot = 0
    capilaryArray = np.zeros([len(t)])
    arterialArray = np.zeros([len(t)])
    venousArray = np.zeros([len(t)])
    
    # volumes of the tree
    arterialArray, capilaryArray, venousArray = plotTreeResults(modelObject, options, 0, grid, scalarMapRA, idxsToIgnore, lWidth, bloodTot)
    # Pressures of the tree
    arterialArrayPressure, capilaryArrayPressure, venousArrayPressure = plotTreeResults(modelObject, options, 1, grid, scalarMapRA, idxsToIgnore, lWidth, bloodTot)
    # Flows of the tree
    arterialArrayCurrent, capilaryArrayCurrent, venousArrayCurrent = plotTreeResults(modelObject, options, 2, grid, scalarMapRA, idxsToIgnore, lWidth, bloodTot)
    
    '''    mpl.figure()
    mpl.rcParams['figure.dpi'] = options['dpi']
    fig1, ax = mpl.subplots(4, 2, constrained_layout=True)
       
    #[axi.xaxis.set_visible(False) for axi in ax.ravel()]
    #[axi.rc('xtick', labelsize=8) for axi in ax.ravel()]
    [axi.tick_params(axis='both',labelsize=4, pad=1, direction='in',grid_linewidth = 1, labelrotation = 15) for axi in ax.ravel()]
    #[axi.rc('axes', labelsize=12) for axi in ax.ravel()]
    
    for key, value in results[0].items():
        if '2' not in key:
            if 'Lu' in key:                
                pass
            elif 'Ap' in key:
                pass
            elif 'Cp' in key:                
                pass
            elif 'Vp' in key: 
                pass
            elif 'As' in key:                
                mean = int(np.round(np.mean(value),1))
                amp = np.round(max(value)-min(value),1)
                ax[0,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key + ' -> ' + str(mean) + ' -> ' + str(amp), linewidth=lWidth)
                bloodTot = bloodTot + mean
            elif 'Cs' in key:                
                mean = int(np.round(np.mean(value),1))
                amp = np.round(max(value)-min(value),1)
                ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key + ' -> ' + str(mean) + ' -> ' + str(amp), linewidth=lWidth)
                bloodTot = bloodTot + mean
            elif 'Vs' in key:                
                mean = int(np.round(np.mean(value),1))
                amp = np.round(max(value)-min(value),1)
                ax[2,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key + ' -> ' + str(mean) + ' -> ' + str(amp), linewidth=lWidth)
                bloodTot = bloodTot + mean
            elif 'Hr' in key:                
                mean = int(np.round(np.mean(value),1))
                amp = np.round(max(value)-min(value),1)
                ax[3,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key + ' -> ' + str(mean) + ' -> ' + str(amp), linewidth=lWidth)
                bloodTot = bloodTot + mean
            elif 'Hl' in key:                
                mean = int(np.round(np.mean(value),1))
                amp = np.round(max(value)-min(value),1)
                ax[3,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key + ' -> ' + str(mean) + ' -> ' + str(amp), linewidth=lWidth)
                bloodTot = bloodTot + mean
            elif ('Ar' in key) or ('Al' in key): 
                mean = int(np.round(np.mean(value),1))
                bloodTot = bloodTot + mean   

    # AP -> Pulmonary arterial Volumes
    mean = int(np.round(np.mean(arterialArray),1))
    bloodTot = bloodTot + mean  
    amp = np.round(max(arterialArray)-min(arterialArray),1)
    ax[0,1].plot(t[idxsToIgnore:-1],arterialArray[idxsToIgnore:-1], label = 'Ap' + ' -> ' + str(mean) + ' -> ' + str(amp), linewidth=lWidth)
    
    # CP -> Pulmonary Capillary Volumes
    mean = int(np.round(np.mean(capilaryArray),1))
    bloodTot = bloodTot + mean 
    amp = np.round(max(capilaryArray)-min(capilaryArray),1)
    ax[1,1].plot(t[idxsToIgnore:-1],capilaryArray[idxsToIgnore:-1], label = 'Cp' + ' -> ' + str(mean) + ' -> ' + str(amp), linewidth=lWidth)
    
    # VP -> Pulmonary Venous Volumes
    mean = int(np.round(np.mean(venousArray),1))
    bloodTot = bloodTot + mean 
    amp = np.round(max(venousArray)-min(venousArray),1)
    ax[2,1].plot(t[idxsToIgnore:-1],venousArray[idxsToIgnore:-1], label = 'Vp' + ' -> ' + str(mean) + ' -> ' + str(amp), linewidth=lWidth)
    
    
    # Calculates the variation of the total volume on the system
    #cumS = calculateVolumeVariaton(results[resultType])
    
    if (options['legend']):
        [axi.legend(prop={'size': legendSize}) for axi in ax.ravel()]
            
    print(Fore.CYAN + 'The Blood Volume is: ' + str(np.round(bloodTot,1)) + ' Mililiters')
    print(Style.RESET_ALL)


    # axis labels
    if True:
        ax[0,0].set_xlabel('Time (s)', fontsize = 5)
        ax[0,0].set_ylabel('Volume (mL)', fontsize = 5)
        ax[0,0].set_title('Volume on Systemic Arteries', size=7)

        ax[0,1].set_xlabel('Time (s)', fontsize = 5)
        ax[0,1].set_ylabel('Volume (mL)', fontsize = 5)
        ax[0,1].set_title('Volume on Pulmonary Arteries', size=7)

        ax[1,0].set_xlabel('Time (s)', fontsize = 5)
        ax[1,0].set_ylabel('Volume (mL)', fontsize = 5)
        ax[1,0].set_title('Volume on Systemic Capillaries', size=7)

        ax[1,1].set_xlabel('Time (s)', fontsize = 5)
        ax[1,1].set_ylabel('Volume (mL)', fontsize = 5)
        ax[1,1].set_title('Volume on Pulmonary Capillaries', size=7)

        ax[2,0].set_xlabel('Time (s)', fontsize = 5)
        ax[2,0].set_ylabel('Volume (mL)', fontsize = 5)
        ax[2,0].set_title('Volume on Systemic Veins', size=7)

        ax[2,1].set_xlabel('Time (s)', fontsize = 5)
        ax[2,1].set_ylabel('Volume (mL)', fontsize = 5)
        ax[2,1].set_title('Volume on Pulmonary Veins', size=7)

        ax[3,0].set_xlabel('Time (s)', fontsize = 5)
        ax[3,0].set_ylabel('Volume (mL)', fontsize = 5)
        ax[3,0].set_title('Volume on Right Ventricle', size=7)

        ax[3,1].set_xlabel('Time (s)', fontsize = 5)
        ax[3,1].set_ylabel('Volume (mL)', fontsize = 5)
        ax[3,1].set_title('Volume on Left Ventricle', size=7)



    mpl.show()'''
    
    # __  __      _        _  _              _     ___ _     _   
    #|  \/  |__ _(_)_ _   | || |___ __ _ _ _| |_  | _ \ |___| |_ 
    #| |\/| / _` | | ' \  | __ / -_) _` | '_|  _| |  _/ / _ \  _|
    #|_|  |_\__,_|_|_||_| |_||_\___\__,_|_|  \__| |_| |_\___/\__|
                                                             
    
    mpl.figure()
    mpl.rcParams['figure.dpi'] = options['dpi']
    fig, ax = mpl.subplots(2, 2, constrained_layout=True)
    
    #[axi.xaxis.set_visible(False) for axi in ax.ravel()]
    [axi.tick_params(axis='both',labelsize=4, pad=1, direction='in',grid_linewidth = 1, labelrotation = 15) for axi in ax.ravel()]
    [axi.grid() for axi in ax.ravel()]
    
    meanVols = {}
    ampVols = {}
    meanPressure = {}
    ampPressure = {}
    maxPressure = {}
##### Volume Plot #############################################################

    for key, value in results[0].items():
        if '2' not in key:
            meanVols[key] = str(int(np.round(np.mean(value[idxsToIgnore:-1]),1)))
            ampVols[key] = str(np.round(max(value[idxsToIgnore:-1])-min(value[idxsToIgnore:-1]),1))
            if 'Hr' in key:
                ax[1,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='grey')
            elif 'Ar' in key:
                ax[1,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='gold')
            elif 'Hl' in key:
                ax[1,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='black')
            elif 'Al' in key:
                ax[1,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='yellow')
            elif 'Ap' in key:
                pass
            elif 'As' in key:
                ax[0,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color = 'darkred')
            elif 'Cp' in key:
                pass
            elif 'Cs' in key:
                ax[0,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color = 'darkgreen')
            elif 'Vp' in key:
                pass
            elif 'Vs' in key:
                ax[0,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color = 'darkblue')
            
    ax[0,0].plot(t[idxsToIgnore:-1],arterialArray[idxsToIgnore:-1], label = 'Ap', linewidth=lWidth, color = 'red')
    ax[0,0].plot(t[idxsToIgnore:-1],capilaryArray[idxsToIgnore:-1], label = 'Cp', linewidth=lWidth, color = 'lime')
    ax[0,0].plot(t[idxsToIgnore:-1],venousArray[idxsToIgnore:-1], label = 'Vp', linewidth=lWidth, color = 'blue')

    print(arterialArray)
        

    

##### Pressure Plot ###########################################################
    for key, value in results[1].items():
        if '2' not in key:
            ampPressure[key] = str(np.round(max(value[idxsToIgnore:-1])-min(value[idxsToIgnore:-1])))
            maxPressure[key] = str(np.round(max(value[idxsToIgnore:-1])))
            meanPressure[key] = str(int(np.round(np.mean(value[idxsToIgnore:-1]))))
            if 'Hr' in key:
                ax[0,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1] - modelObject.lungParams['atmosphericPressure'], label = key, linewidth=lWidth, color='grey')
            elif 'Hl' in key:
                ax[0,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1] - modelObject.lungParams['atmosphericPressure'], label = key, linewidth=lWidth, color='black')
            elif 'Al' in key:
                ax[0,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1] - modelObject.lungParams['atmosphericPressure'], label = key, linewidth=lWidth, color='yellow')
            elif 'Ar' in key:
                ax[0,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1] - modelObject.lungParams['atmosphericPressure'], label = key, linewidth=lWidth, color='gold')
            elif 'As' in key:
                ax[0,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1] - modelObject.lungParams['atmosphericPressure'], label = key, linewidth=lWidth, color='darkred')
            elif 'Cs' in key:
                ax[0,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1] - modelObject.lungParams['atmosphericPressure'], label = key, linewidth=lWidth, color='darkgreen')
            elif 'Vs' in key:
                ax[0,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1] - modelObject.lungParams['atmosphericPressure'], label = key, linewidth=lWidth, color='darkblue')
            elif key == 'V_Ap|0':
                ax[0,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1] - modelObject.lungParams['atmosphericPressure'], label = key, linewidth=lWidth, color='red')
        
    nrBranches = 2**(modelObject.treeParams['maxLevels'] + 1) - 1 - (2**modelObject.treeParams['maxLevels'])
    nrCapilaries = 2**(modelObject.treeParams['maxLevels'])
    #ax[0,1].plot(t[idxsToIgnore:-1],arterialArrayPressure[idxsToIgnore:-1]/nrBranches, label = 'Ap', linewidth=lWidth, color = 'red')
    #ax[0,1].plot(t[idxsToIgnore:-1],arterialArrayPressure[idxsToIgnore:-1]/nrBranches - modelObject.lungParams['atmosphericPressure'], label = 'Ap', linewidth=lWidth, color = 'salmon')
    ax[0,1].plot(t[idxsToIgnore:-1],capilaryArrayPressure[idxsToIgnore:-1]/nrCapilaries - modelObject.lungParams['atmosphericPressure'], label = 'Cp', linewidth=lWidth, color = 'lime')
    ax[0,1].plot(t[idxsToIgnore:-1],venousArrayPressure[idxsToIgnore:-1]/nrBranches - modelObject.lungParams['atmosphericPressure'], label = 'Vp', linewidth=lWidth, color = 'blue')

    
##### Current Plot ############################################################
    for key, value in results[2].items():
        if '2' not in key:
            if 'Hr' in key:
                ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='grey')
            elif 'Hl' in key:
                ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='black')
            elif 'Al' in key:
                ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='yellow')
            elif 'Ar' in key:
                ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='gold')
            elif 'As' in key:
                ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='darkred')
                print('Systemic Flow' + str(np.cumsum(value)[-1]))               
            elif 'Cs' in key:
                ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='darkgreen')
            elif 'Vs' in key:
                ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='darkblue')
            elif key == 'I_Ap|0|0':
                ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1]*2, label = 'Ap', linewidth=lWidth, color='red')
                print('Pulmonary Flow' + str(np.cumsum(value)[-1]))
            #elif key == 'I_Ap|0|1':
            #    ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='red')
            elif key == 'I_Vp|0':
                ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = 'Vp', linewidth=lWidth, color='blue')
    
    nrBranches = 2**(modelObject.treeParams['maxLevels'] + 1) - 1 - (2**modelObject.treeParams['maxLevels'])
    #ax[1,0].plot(t[idxsToIgnore:-1],arterialArrayCurrent[idxsToIgnore:-1]/nrBranches, label = 'Ap', linewidth=lWidth, color = 'red')
    ax[1,0].plot(t[idxsToIgnore:-1],capilaryArrayCurrent[idxsToIgnore:-1]/nrBranches, label = 'Cp', linewidth=lWidth, color = 'lime')
    #ax[1,0].plot(t[idxsToIgnore:-1],venousArrayCurrent[idxsToIgnore:-1]/nrBranches, label = 'Vp', linewidth=lWidth, color = 'blue')

    
       
    
    if (options['legend']):
        ax[0,0].legend(prop={'size': legendSize})
        ax[1,0].legend(prop={'size': legendSize})
        ax[0,1].legend(prop={'size': legendSize})
        ax[1,1].legend(prop={'size': legendSize})
    else:
        pass

    # axis Labels
    if True:
        ax[0,0].set_xlabel('Time (s)', fontsize = 5)
        ax[0,0].set_ylabel('Volume (mL)', fontsize = 5)
        ax[0,0].set_title('Volume on Blood vessels', size=7)

        ax[0,1].set_xlabel('Time (s)', fontsize = 5)
        ax[0,1].set_ylabel('Pressure (mmHg)', fontsize = 5)
        ax[0,1].set_title('Pressure on all Cardiovascular Compartments', size=7)

        ax[1,0].set_xlabel('Time (s)', fontsize = 5)
        ax[1,0].set_ylabel('Flow (mL/s)', fontsize = 5)
        ax[1,0].set_title('Flow on all Cardiovascular Compartments', size=7)

        ax[1,1].set_xlabel('Time (s)', fontsize = 5)
        ax[1,1].set_ylabel('Volume (mL)', fontsize = 5)
        ax[1,1].set_title('Volume on the Heart Compartments', size=7)

    mpl.show()
   
# used for when there are only results for the cardiovascular side of the model    
def plotHeartResultsSinglePlot(modelObject, options):
    t = modelObject.simulation['t']
    result = modelObject.solution
    
    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 5,
        }
    
    grid = {
        'top': 1, 
        'bottom': 0, 
        'right': 1, 
        'left': 0,
        'hspace': 0, 
        'wspace': 0.07
        }

    lWidth = options['lWidth']
    legendSize = options['legendSize']
    idxsToIgnore = options['idxsToIgnore']
    
    #mpl.figure()
    mpl.rcParams['figure.dpi'] = options['dpi']
    fig, ax = mpl.subplots(2, 2, constrained_layout=True)
    
    #[axi.xaxis.set_visible(False) for axi in ax.ravel()]
    [axi.tick_params(axis='both',labelsize=4, pad=1, direction='in',grid_linewidth = 1, labelrotation = 15) for axi in ax.ravel()]
    [axi.grid() for axi in ax.ravel()]
    
    meanVols = {}
    ampVols = {}
    meanPressure = {}
    ampPressure = {}
    maxPressure = {}
##### Volume Plot #############################################################
    tot = 0
    for key, value in result[0].items():
        meanVols[key] = str(int(np.round(np.mean(value[idxsToIgnore:-1]),1)))
        ampVols[key] = str(np.round(max(value[idxsToIgnore:-1])-min(value[idxsToIgnore:-1]),1))
        if ('Hr' in key) or ('Hl' in key):
            ax[1,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth)
        elif ('Ar' in key) or ('Al' in key):
            ax[1,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth)
        elif 'Ap' in key:
            ax[0,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color = 'red')
        elif 'As' in key:
            ax[0,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color = 'darkred')
        elif 'Cp' in key:
            ax[0,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color = 'lime')
        elif 'Cs' in key:
            ax[0,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color = 'darkgreen')
        elif 'Vp' in key:
            ax[0,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color = 'blue')
        elif 'Vs' in key:
            ax[0,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color = 'darkblue')
        
        tot = tot + value[-1]
    

##### Pressure Plot ###########################################################
    for key, value in result[1].items():
        ampPressure[key] = str(np.round(max(value[idxsToIgnore:-1])-min(value[idxsToIgnore:-1])))
        maxPressure[key] = str(np.round(max(value[idxsToIgnore:-1])))
        meanPressure[key] = str(int(np.round(np.mean(value[idxsToIgnore:-1]))))
        if 'Hr' in key:
            ax[0,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1] - modelObject.lungParams['atmosphericPressure'], label = key, linewidth=lWidth, color='grey')
        elif 'Hl' in key:
            ax[0,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1] - modelObject.lungParams['atmosphericPressure'], label = key, linewidth=lWidth, color='black')
        elif 'Al' in key:
            ax[0,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1] - modelObject.lungParams['atmosphericPressure'], label = key, linewidth=lWidth, color='yellow')
        elif 'Ar' in key:
            ax[0,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1] - modelObject.lungParams['atmosphericPressure'], label = key, linewidth=lWidth, color='gold')
        elif 'Ap' in key:
            ax[0,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1] - modelObject.lungParams['atmosphericPressure'], label = key, linewidth=lWidth, color='red')
        elif 'As' in key:
            ax[0,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1] - modelObject.lungParams['atmosphericPressure'], label = key, linewidth=lWidth, color='darkred')
        elif 'Cp' in key:
            ax[0,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1] - modelObject.lungParams['atmosphericPressure'], label = key, linewidth=lWidth, color='lime')
        elif 'Cs' in key:
            ax[0,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1] - modelObject.lungParams['atmosphericPressure'], label = key, linewidth=lWidth, color='darkgreen')
        elif 'Vp' in key:
            ax[0,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1] - modelObject.lungParams['atmosphericPressure'], label = key, linewidth=lWidth, color='blue')
        elif 'Vs' in key:
            ax[0,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1] - modelObject.lungParams['atmosphericPressure'], label = key, linewidth=lWidth, color='darkblue')

    
##### Current Plot ############################################################
    for key, value in result[2].items():
        if 'Hr' in key:
            ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='grey')
        elif 'Hl' in key:
            ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='black')
        elif 'Al' in key:
            ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='yellow')
        elif 'Ar' in key:
            ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='gold')
        elif 'Ap' in key:
            ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='red')
        elif 'As' in key:
            ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='darkred')
        elif 'Cp' in key:
            ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='lime')
        elif 'Cs' in key:
            ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='darkgreen')
        elif 'Vp' in key:
            ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='blue')
        elif 'Vs' in key:
            ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='darkblue')
       
    
    if (options['legend']):
        ax[0,0].legend(prop={'size': legendSize},facecolor='white', framealpha=1)
        ax[1,0].legend(prop={'size': legendSize},facecolor='white', framealpha=1)
        ax[0,1].legend(prop={'size': legendSize},facecolor='white', framealpha=1)
        ax[1,1].legend(prop={'size': legendSize},facecolor='white', framealpha=1)
    else:
        pass
    
    #print(Style.RESET_ALL)
    #print('Mean volume (mL): ' + json.dumps(meanVols, indent=4))
    #print('Amplitude of volume (mL): ' + json.dumps(ampVols, indent=4))
    #print('Mean Pressure (mmHg): ' + json.dumps(meanPressure, indent=4))
    #print('Pressure amplitude (mmHg): ' + json.dumps(ampPressure, indent=4))
    #print('Systolic Pressure (mmHg): ' + json.dumps(maxPressure, indent=4))
    #print('Total Volume (mL) is: ' + str(np.round(tot,1)))

        # axis Labels
    if True:
        ax[0,0].set_xlabel('Time (s)', fontsize = 5)
        ax[0,0].set_ylabel('Volume (mL)', fontsize = 5)
        ax[0,0].set_title('Volume on Blood vessels', size=7)

        ax[0,1].set_xlabel('Time (s)', fontsize = 5)
        ax[0,1].set_ylabel('Pressure (mmHg)', fontsize = 5)
        ax[0,1].set_title('Pressure on all Cardiovascular Compartments', size=7)

        ax[1,0].set_xlabel('Time (s)', fontsize = 5)
        ax[1,0].set_ylabel('Flow (mL/s)', fontsize = 5)
        ax[1,0].set_title('Flow on all Cardiovascular Compartments', size=7)

        ax[1,1].set_xlabel('Time (s)', fontsize = 5)
        ax[1,1].set_ylabel('Volume (mL)', fontsize = 5)
        ax[1,1].set_title('Volume on the Heart Compartments', size=7)

    mpl.show()

'''
..####....####....####...........######..##..##...####...##..##...####...##..##...####...######.
.##......##..##..##..............##.......####...##..##..##..##..##..##..###.##..##......##.....
.##.###..######...####...........####......##....##......######..######..##.###..##.###..####...
.##..##..##..##......##..........##.......####...##..##..##..##..##..##..##..##..##..##..##.....
..####...##..##...####...........######..##..##...####...##..##..##..##..##..##...####...######.
'''                                                      
# Plot that summarises the gases
def plotLungGasResultsSinglePlot(modelObject, options):
    t = modelObject.simulation['t']
    guidePressures={}
    for idx,species in enumerate(modelObject.params['partialPressures']):
        guidePressures[species + '_Atm'] = np.ones(len(t))* modelObject.lungParams['partialPressuresY0'][idx] 
        guidePressures[species + '_Blood'] = np.ones(len(t))* modelObject.cardioModelParams['venousPartialPressuresY0'][idx] 
        
    result = modelObject.solution
    
    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 5,
        }
    
    grid = {
        'top': 1, 
        'bottom': 0, 
        'right': 1, 
        'left': 0,
        'hspace': 0, 
        'wspace': 0.07
        }

    lWidth = options['lWidth']
    legendSize = options['legendSize']
    idxsToIgnore = options['idxsToIgnore']
    
    # coolwarm
    colormapRA = mpl.get_cmap('hot') 
    cNormRA  = colors.Normalize(vmin=0, vmax=modelObject.simulation['maxLevels'] + 1)
    scalarMapRA = cmx.ScalarMappable(norm=cNormRA, cmap=colormapRA)
    
    keys = np.array(list(result[3].keys()))
    sizes = np.char.str_len(keys)
    maxLen = max(sizes)
   
    mpl.rcParams['figure.dpi'] = options['dpi']
    fig, ax = mpl.subplots(3, 2, constrained_layout=True)
    
    #[axi.xaxis.set_visible(False) for axi in ax.ravel()]
    [axi.tick_params(axis='both',labelsize=4, pad=1, direction='in',grid_linewidth = 1, labelrotation = 15) for axi in ax.ravel()]
    
    phaseChanges = np.where(np.roll(result[7]['resp'],1)!=result[7]['resp'])
    rectangles = np.zeros([int(len(phaseChanges[0])/2),2])
    counter = 0
    for idx,val in enumerate(phaseChanges[0]):
        if (idx % 2 == 0):
            rectangles[counter,0] = val
        else:
            rectangles[counter,1] = val - rectangles[counter,0]
            counter = counter + 1
    for rec in rectangles:
        start = int(rec[0])
        end = int(rec[0]+rec[1])
        ax[0,0].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[1,0].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[2,0].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[0,1].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[1,1].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[2,1].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
    
    for key, value in result[3].items():
        comp = list(filter(lambda comp: key in comp['partialPressures'], modelObject.compartments))
        level = comp[0]['level']
        if ('N2' in key) and ('Lu' in key):
            if (len(key) == maxLen) and ('Lu' in key):
                ax[0,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth)
            else:
                colorValR = scalarMapRA.to_rgba(level)  
                ax[0,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color=colorValR)
        elif ('C2' in key) and('Lu' in key):
            if (len(key) == maxLen) and ('Lu' in key):
                ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth)
            else:
                colorValR = scalarMapRA.to_rgba(level) 
                ax[1,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color=colorValR)
        elif ('O2' in key) and('Lu' in key):
            if (len(key) == maxLen) and ('Lu' in key):
                ax[2,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth)
            else:
                colorValR = scalarMapRA.to_rgba(level) 
                ax[2,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color=colorValR)
    
    guidePressures = {}
    for idx,species in enumerate(modelObject.params['partialPressures']):
        guidePressures[species + '_Atm'] = np.ones(len(t))* modelObject.lungParams['partialPressuresY0'][idx] 
        guidePressures[species + '_Blood'] = np.ones(len(t))* modelObject.cardioModelParams['venousPartialPressuresY0'][idx]
        
        if ('N2' in species):
            ax[0,0].plot(t[idxsToIgnore:-1], guidePressures[species + '_Atm'][idxsToIgnore:-1], label = species + '_Atm', linewidth=lWidth, color='black')
            ax[0,0].plot(t[idxsToIgnore:-1], guidePressures[species + '_Blood'][idxsToIgnore:-1], label = species + '_Blood', linewidth=lWidth, color='grey')
            
        elif ('C2' in species):
            ax[1,0].plot(t[idxsToIgnore:-1], guidePressures[species + '_Atm'][idxsToIgnore:-1], label = species + '_Atm', linewidth=lWidth, color='black')
            ax[1,0].plot(t[idxsToIgnore:-1], guidePressures[species + '_Blood'][idxsToIgnore:-1], label = species + '_Blood', linewidth=lWidth, color='grey')
            
        elif ('O2' in species):
            ax[2,0].plot(t[idxsToIgnore:-1], guidePressures[species + '_Atm'][idxsToIgnore:-1], label = species + '_Atm', linewidth=lWidth, color='black')
            ax[2,0].plot(t[idxsToIgnore:-1], guidePressures[species + '_Blood'][idxsToIgnore:-1], label = species + '_Blood', linewidth=lWidth, color='grey')
            
    if (options['legend']):
        ax[0,0].legend(prop={'size': legendSize})
        ax[0,1].legend(prop={'size': legendSize})
        ax[1,0].legend(prop={'size': legendSize})
        ax[1,1].legend(prop={'size': legendSize})
        ax[2,0].legend(prop={'size': legendSize})
        ax[2,1].legend(prop={'size': legendSize})
        #fig.suptitle('Total Volume' + str(np.round(tot,1)))
        pass
    else:
        pass

    mpl.show()


# Plot that summarises the gases
def plotLungGasResultsSinglePlotV2(modelObject, options):
    t = modelObject.simulation['t']
    dt = t[1]-t[0]
    result = modelObject.solution
    
    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 5,
        }
    
    grid = {
        'top': 1, 
        'bottom': 0, 
        'right': 1, 
        'left': 0,
        'hspace': 0, 
        'wspace': 0.07
        }

    lWidth = options['lWidth']
    legendSize = options['legendSize']
    idxsToIgnore = options['idxsToIgnore']
    
    # coolwarm
    colormapRA = mpl.get_cmap('hot') 
    cNormRA  = colors.Normalize(vmin=0, vmax=modelObject.simulation['maxLevels'])
    scalarMapRA = cmx.ScalarMappable(norm=cNormRA, cmap=colormapRA)
    
    keys = np.array(list(result[3].keys()))
    sizes = np.char.str_len(keys)
    maxLen = max(sizes)
    minLen = min(sizes)
   
    mpl.rcParams['figure.dpi'] = options['dpi']
    fig, ax = mpl.subplots(3, 2, gridspec_kw = grid)
    
    [axi.xaxis.set_visible(False) for axi in ax.ravel()]
    [axi.tick_params(axis='both',labelsize=4, pad=1, direction='in',grid_linewidth = 1, labelrotation = 15) for axi in ax.ravel()]
    
    
    phaseChanges = np.where(np.roll(result[7]['resp'],1)!=result[7]['resp'])
    rectangles = np.zeros([int(len(phaseChanges[0])/2),2])
    counter = 0
    for idx,val in enumerate(phaseChanges[0]):
        if (idx % 2 == 0):
            rectangles[counter,0] = val
        else:
            rectangles[counter,1] = val - rectangles[counter,0]
            counter = counter + 1
    for rec in rectangles:
        start = int(rec[0])
        end = int(rec[0]+rec[1])
        ax[0,0].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[1,0].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[2,0].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[0,1].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[1,1].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[2,1].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
    
    for key, value in result[3].items():
        comp = list(filter(lambda comp: key in comp['partialPressures'], modelObject.compartments))
        level = comp[0]['level']
        '''
        if ('N2' in key):
            if (len(key) == maxLen) and ('Lu' in key):
                ax[0,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='green')
            elif (len(key) == minLen + 2) and ('Lu' in key):
                ax[2,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='green')
            elif ('Lu' in key):
                colorValR = scalarMapRA.to_rgba(level)  
                ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='green')
                #ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color=colorValR)
        '''       
        if ('C2' in key) :
            if (len(key) == maxLen) and ('Lu' in key):
                ax[0,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='red')
            elif (len(key) == minLen + 2) and ('Lu' in key):
                ax[2,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='red')
            elif ('Lu' in key):
                colorValR = scalarMapRA.to_rgba(level) 
                ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='red')
                #ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color=colorValR)
        elif ('O2' in key) :
            if (len(key) == maxLen) and ('Lu' in key):
                ax[0,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='blue')
            elif (len(key) == minLen + 2) and ('Lu' in key):
                ax[2,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='blue')
            elif ('Lu' in key):
                colorValR = scalarMapRA.to_rgba(level) 
                ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='blue')
                #ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color=colorValR)
    
    keys = np.array(list(result[2].keys()))
    sizes = np.char.str_len(keys)
    maxLen = max(sizes)
    minLen = min(sizes)
    for key, value in result[2].items():
        #comp = list(filter(lambda comp: key in comp['partialPressures'], modelObject.compartments))
        #level = comp[0]['level']
        if ('N2' in key):
            if ('Lu' in key):
                ax[0,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='green')
                
        elif ('C2' in key):
            if ('Lu' in key):
                ax[1,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='red')
        elif ('O2' in key):
            if ('Lu' in key):
                ax[2,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='blue')
    
    if (options['legend']):
        ax[0,0].legend(prop={'size': legendSize})
        ax[0,1].legend(prop={'size': legendSize})
        ax[1,0].legend(prop={'size': legendSize})
        ax[1,1].legend(prop={'size': legendSize})
        ax[2,0].legend(prop={'size': legendSize})
        ax[2,1].legend(prop={'size': legendSize})
        #fig.suptitle('Total Volume' + str(np.round(tot,1)))
        pass
    else:
        pass

    mpl.show()

# Fixed
def plotHeartGasResultsSinglePlot(modelObject, options):
    t = modelObject.simulation['t']
    result = modelObject.solution
    
    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 5,
        }
    
    grid = {
        'top': 1, 
        'bottom': 0, 
        'right': 1, 
        'left': 0,
        'hspace': 0, 
        'wspace': 0.07
        }

    lWidth = options['lWidth']
    legendSize = options['legendSize']
    idxsToIgnore = options['idxsToIgnore']
    
    # coolwarm
    colormapRA = mpl.get_cmap('hot') 
    cNormRA  = colors.Normalize(vmin=0, vmax=modelObject.simulation['maxLevels'])
    scalarMapRA = cmx.ScalarMappable(norm=cNormRA, cmap=colormapRA)
    
    keys = np.array(list(result[3].keys()))
    sizes = np.char.str_len(keys)
    maxLen = max(sizes)
   
    mpl.rcParams['figure.dpi'] = options['dpi']
    fig, ax = mpl.subplots(3, 3, constrained_layout=True)
    
    #[axi.xaxis.set_visible(False) for axi in ax.ravel()]
    [axi.tick_params(axis='both',labelsize=4, pad=1, direction='in',grid_linewidth = 1, labelrotation = 15) for axi in ax.ravel()]
    

    phaseChanges = np.where(np.roll(result[7]['resp'],1)!=result[7]['resp'])
    rectangles = np.zeros([int(len(phaseChanges[0])/2),2])
    counter = 0
    
    for idx,val in enumerate(phaseChanges[0]):
        if (idx % 2 == 0):
            rectangles[counter,0] = val
        else:
            rectangles[counter,1] = val - rectangles[counter,0]
            counter = counter + 1
    for rec in rectangles:
        start = int(rec[0])
        end = int(rec[0]+rec[1])
        ax[0,0].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[1,0].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[2,0].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[0,1].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[1,1].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[2,1].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[0,2].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[1,2].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[2,2].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
  
    
    for key, value in result[3].items():
        comp = list(filter(lambda comp: key in comp['partialPressures'], modelObject.compartments))
        level = comp[0]['level']
        if 'lu' not in key:
            if 'N2' in key:
                if ('Vp' in key) or ('As' in key):
                    ax[0,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth)
                elif ('Cp' in key) or ('Cs' in key):
                    colorValR = scalarMapRA.to_rgba(level)  
                    ax[0,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, )
                elif ('Ap' in key) or ('Vs' in key):
                    colorValR = scalarMapRA.to_rgba(level)  
                    ax[0,2].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, )
            elif 'C2' in key:
                if ('Vp' in key) or ('As' in key):
                    ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth)
                elif ('Cp' in key) or ('Cs' in key):
                    colorValR = scalarMapRA.to_rgba(level)  
                    ax[1,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth,)
                elif ('Ap' in key) or ('Vs' in key):
                    colorValR = scalarMapRA.to_rgba(level)  
                    ax[1,2].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, )
            elif 'O2' in key:
                if ('Vp' in key) or ('As' in key):
                    ax[2,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth)
                elif ('Cp' in key) or ('Cs' in key):
                    colorValR = scalarMapRA.to_rgba(level)  
                    ax[2,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, )
                elif ('Ap' in key) or ('Vs' in key):
                    colorValR = scalarMapRA.to_rgba(level)  
                    ax[2,2].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, )
    
    if (options['legend']):
        ax[0,0].legend(prop={'size': legendSize},facecolor='white', framealpha=1)
        ax[0,1].legend(prop={'size': legendSize},facecolor='white', framealpha=1)
        ax[0,2].legend(prop={'size': legendSize},facecolor='white', framealpha=1)
        ax[1,0].legend(prop={'size': legendSize},facecolor='white', framealpha=1)
        ax[1,1].legend(prop={'size': legendSize},facecolor='white', framealpha=1)
        ax[1,2].legend(prop={'size': legendSize},facecolor='white', framealpha=1)
        ax[2,0].legend(prop={'size': legendSize},facecolor='white', framealpha=1)
        ax[2,1].legend(prop={'size': legendSize},facecolor='white', framealpha=1)
        ax[2,2].legend(prop={'size': legendSize},facecolor='white', framealpha=1)
        #fig.suptitle('Total Volume' + str(np.round(tot,1)))
        pass
    else:
        pass

    ax[0,0].set_xlabel('Time (s)', fontsize = 5)
    ax[0,0].set_ylabel('Pressure (mmHg)', fontsize = 5)
    ax[0,0].set_title('PP of $N_2$ in Arterial blood', size=7)

    ax[0,1].set_xlabel('Time (s)', fontsize = 5)
    ax[0,1].set_ylabel('Pressure (mmHg)', fontsize = 5)
    ax[0,1].set_title('PP of $N_2$ in Capillaries', size=7)

    ax[0,2].set_xlabel('Time (s)', fontsize = 5)
    ax[0,2].set_ylabel('Pressure (mmHg)', fontsize = 5)
    ax[0,2].set_title('PP of $N_2$ in Venous Blood', size=7)


    ax[1,0].set_xlabel('Time (s)', fontsize = 5)
    ax[1,0].set_ylabel('Pressure (mmHg)', fontsize = 5)
    ax[1,0].set_title('PP of $CO_2$ in Arterial blood', size=7)

    ax[1,1].set_xlabel('Time (s)', fontsize = 5)
    ax[1,1].set_ylabel('Pressure (mmHg)', fontsize = 5)
    ax[1,1].set_title('PP of $CO_2$ in Capillaries', size=7)

    ax[1,2].set_xlabel('Time (s)', fontsize = 5)
    ax[1,2].set_ylabel('Pressure (mmHg)', fontsize = 5)
    ax[1,2].set_title('PP of $CO_2$ in Venous Blood', size=7)


    ax[2,0].set_xlabel('Time (s)', fontsize = 5)
    ax[2,0].set_ylabel('Pressure (mmHg)', fontsize = 5)
    ax[2,0].set_title('PP of $O_2$ in Arterial blood', size=7)

    ax[2,1].set_xlabel('Time (s)', fontsize = 5)
    ax[2,1].set_ylabel('Pressure (mmHg)', fontsize = 5)
    ax[2,1].set_title('PP of $O_2$ in Capillaries', size=7)

    ax[2,2].set_xlabel('Time (s)', fontsize = 5)
    ax[2,2].set_ylabel('Pressure (mmHg)', fontsize = 5)
    ax[2,2].set_title('PP of $O_2$ in Venous Blood', size=7)

    mpl.show()


# Plot that summarises the gases
def plotHeartGasResultsSinglePlotV2(modelObject, options):
    t = modelObject.simulation['t']
    dt = t[1]-t[0]
    result = modelObject.solution
    
    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 5,
        }
    
    grid = {
        'top': 1, 
        'bottom': 0, 
        'right': 1, 
        'left': 0,
        'hspace': 0, 
        'wspace': 0.07
        }

    lWidth = options['lWidth']
    legendSize = options['legendSize']
    idxsToIgnore = options['idxsToIgnore']
    
    # coolwarm
    colormapRA = mpl.get_cmap('hot') 
    cNormRA  = colors.Normalize(vmin=0, vmax=modelObject.simulation['maxLevels'])
    scalarMapRA = cmx.ScalarMappable(norm=cNormRA, cmap=colormapRA)
    
    keys = np.array(list(result[3].keys()))
    sizes = np.char.str_len(keys)
    maxLen = max(sizes)
    minLen = min(sizes)
   
    mpl.rcParams['figure.dpi'] = options['dpi']
    fig, ax = mpl.subplots(3, 2, gridspec_kw = grid)
    
    [axi.xaxis.set_visible(False) for axi in ax.ravel()]
    [axi.tick_params(axis='both',labelsize=4, pad=1, direction='in',grid_linewidth = 1, labelrotation = 15) for axi in ax.ravel()]
    
   
    phaseChanges = np.where(np.roll(result[7]['resp'],1)!=result[7]['resp'])
    rectangles = np.zeros([int(len(phaseChanges[0])/2),2])
    counter = 0
    for idx,val in enumerate(phaseChanges[0]):
        if (idx % 2 == 0):
            rectangles[counter,0] = val
        else:
            rectangles[counter,1] = val - rectangles[counter,0]
            counter = counter + 1
    for rec in rectangles:
        start = int(rec[0])
        end = int(rec[0]+rec[1])
        ax[0,0].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[1,0].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[2,0].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[0,1].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[1,1].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[2,1].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
    
    for key, value in result[3].items():
        comp = list(filter(lambda comp: key in comp['partialPressures'], modelObject.compartments))
        level = comp[0]['level']
        if 'lu' not in key:
            if 'N2' in key:
                if (len(key) == maxLen) and ('Lu' in key):
                    ax[0,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='green')
                elif (len(key) == minLen) and ('Lu' in key):
                    ax[2,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='green')
                else:
                    colorValR = scalarMapRA.to_rgba(level)  
                    ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='green')
                    #ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color=colorValR)
                
            elif 'C2' in key:
                if (len(key) == maxLen) and ('Lu' in key):
                    ax[0,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='red')
                elif (len(key) == minLen) and ('Lu' in key):
                    ax[2,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='red')
                else:
                    colorValR = scalarMapRA.to_rgba(level) 
                    ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='red')
                    #ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color=colorValR)
            elif 'O2' in key:
                if (len(key) == maxLen) and ('Lu' in key):
                    ax[0,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='blue')
                elif (len(key) == minLen) and ('Lu' in key):
                    ax[2,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='blue')
                else:
                    colorValR = scalarMapRA.to_rgba(level) 
                    ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='blue')
                    #ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color=colorValR)
    
    keys = np.array(list(result[2].keys()))
    sizes = np.char.str_len(keys)
    maxLen = max(sizes)
    minLen = min(sizes)
    for key, value in result[2].items():
        #comp = list(filter(lambda comp: key in comp['partialPressures'], modelObject.compartments))
        #level = comp[0]['level']
        if 'lu' not in key:
            if 'N2' in key:
                ax[0,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='green')  
            elif 'C2' in key:
                ax[1,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='red')
            elif 'O2' in key:
                ax[2,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='blue')
    
    if (options['legend']):
        #ax[0,0].legend(prop={'size': legendSize})
        #ax[0,1].legend(prop={'size': legendSize})
        #ax[1,0].legend(prop={'size': legendSize})
        #ax[1,1].legend(prop={'size': legendSize})
        #ax[2,0].legend(prop={'size': legendSize})
        #ax[2,1].legend(prop={'size': legendSize})
        #fig.suptitle('Total Volume' + str(np.round(tot,1)))
        pass
    else:
        pass

    mpl.show()

# Fixed  
def plotGasTransferResultsSinglePlot(modelObject, options):
    t = modelObject.simulation['t']
    dt = t[1]-t[0]
    result = modelObject.solution
    
    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 5,
        }
    
    grid = {
        'top': 1, 
        'bottom': 0, 
        'right': 1, 
        'left': 0,
        'hspace': 0, 
        'wspace': 0.07
        }

    lWidth = options['lWidth']
    legendSize = options['legendSize']
    idxsToIgnore = options['idxsToIgnore']
   
    mpl.rcParams['figure.dpi'] = options['dpi']
    fig, ax = mpl.subplots(3, 2, constrained_layout=True)
    
    [axi.xaxis.set_visible(False) for axi in ax.ravel()]
    [axi.tick_params(axis='both',labelsize=4, pad=1, direction='in',grid_linewidth = 1, labelrotation = 15) for axi in ax.ravel()]
    
    
    phaseChanges = np.where(np.roll(result[7]['resp'],1)!=result[7]['resp'])
    rectangles = np.zeros([int(len(phaseChanges[0])/2),2])
    counter = 0
    for idx,val in enumerate(phaseChanges[0]):
        if (idx % 2 == 0):
            rectangles[counter,0] = val
        else:
            rectangles[counter,1] = val - rectangles[counter,0]
            counter = counter + 1
    for rec in rectangles:
        start = int(rec[0])
        end = int(rec[0]+rec[1])
        ax[0,0].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[1,0].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[2,0].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[0,1].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[1,1].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[2,1].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
    
    
    for key, value in result[2].items():
        if ('Lu' in key) and ('N2' in key):
            ax[0,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='lime')  
        elif ('Lu' in key) and ('C2' in key):
            ax[1,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='pink')
        elif ('Lu' in key) and ('O2' in key):
            ax[2,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='cyan')

        if ('Lu' not in key) and ('N2' in key):
            ax[0,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='green')  
        elif ('Lu' not in key) and ('C2' in key):
            ax[1,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='red')
        elif ('Lu' not in key) and ('O2' in key):
            ax[2,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='blue')
            
    for key, value in result[3].items():
        if ('Cp' in key) and ('N2' in key):
            ax[0,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='darkgreen')  
        elif ('Cp' in key) and ('C2' in key):
            ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='darkred')
        elif ('Cp' in key) and ('O2' in key):
            ax[2,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='darkblue')
        
        if modelObject.simulation['maxLevels'] >= 2:
            if ('Lu' in key) and ('N2' in key) and (len(key)==(8 + 2*modelObject.simulation['maxLevels'])):
                ax[0,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='green')  
            elif ('Lu' in key) and ('C2' in key) and (len(key)==(8 + 2*modelObject.simulation['maxLevels'])):
                ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='red')
            elif ('Lu' in key) and ('O2' in key) and (len(key)==(8 + 2*modelObject.simulation['maxLevels'])):
                ax[2,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='blue')
        else:
            if ('Lu' in key) and ('N2' in key) and (len(key)==(10 + 2*modelObject.simulation['maxLevels'])):
                ax[0,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='green')  
            elif ('Lu' in key) and ('C2' in key) and (len(key)==(10 + 2*modelObject.simulation['maxLevels'])):
                ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='red')
            elif ('Lu' in key) and ('O2' in key) and (len(key)==(10 + 2*modelObject.simulation['maxLevels'])):
                ax[2,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth, color='blue')
    
    '''
    guidePressures = {}
    for idx,species in enumerate(modelObject.params['partialPressures']):
        guidePressures[species + '_Atm'] = np.ones(len(t))* modelObject.lungParams['partialPressuresY0'][idx] 
        guidePressures[species + '_Venous'] = np.ones(len(t))* modelObject.cardioModelParams['venousPartialPressuresY0'][idx]
        guidePressures[species + '_Arterial'] = np.ones(len(t))* modelObject.cardioModelParams['arterialPartialPressuresY0'][idx]
        guidePressures[species + '_Tissues'] = np.ones(len(t))* modelObject.cardioModelParams['tissuePartialPressuresY0'][idx]
        if ('N2' in species):
            ax[0,0].plot(t[idxsToIgnore:-1], guidePressures[species + '_Atm'][idxsToIgnore:-1], label = species + '_Atm', linewidth=lWidth, color='black')
            ax[0,0].plot(t[idxsToIgnore:-1], guidePressures[species + '_Venous'][idxsToIgnore:-1], label = species + '_Venous', linewidth=lWidth, color='cyan')
            ax[0,0].plot(t[idxsToIgnore:-1], guidePressures[species + '_Arterial'][idxsToIgnore:-1], label = species + '_Arterial', linewidth=lWidth, color='lightcoral')
            ax[0,0].plot(t[idxsToIgnore:-1], guidePressures[species + '_Tissues'][idxsToIgnore:-1], label = species + '_Tissues', linewidth=lWidth, color='darkmagenta')
        elif ('C2' in species):
            ax[1,0].plot(t[idxsToIgnore:-1], guidePressures[species + '_Atm'][idxsToIgnore:-1], label = species + '_Atm', linewidth=lWidth, color='black')
            ax[1,0].plot(t[idxsToIgnore:-1], guidePressures[species + '_Venous'][idxsToIgnore:-1], label = species + '_Venous', linewidth=lWidth, color='cyan')
            ax[1,0].plot(t[idxsToIgnore:-1], guidePressures[species + '_Arterial'][idxsToIgnore:-1], label = species + '_Arterial', linewidth=lWidth, color='lightcoral')
            ax[1,0].plot(t[idxsToIgnore:-1], guidePressures[species + '_Tissues'][idxsToIgnore:-1], label = species + '_Tissues', linewidth=lWidth, color='darkmagenta')
        elif ('O2' in species):
            ax[2,0].plot(t[idxsToIgnore:-1], guidePressures[species + '_Atm'][idxsToIgnore:-1], label = species + '_Atm', linewidth=lWidth, color='black')
            ax[2,0].plot(t[idxsToIgnore:-1], guidePressures[species + '_Venous'][idxsToIgnore:-1], label = species + '_Venous', linewidth=lWidth, color='cyan')
            ax[2,0].plot(t[idxsToIgnore:-1], guidePressures[species + '_Arterial'][idxsToIgnore:-1], label = species + '_Arterial', linewidth=lWidth, color='lightcoral')
            ax[2,0].plot(t[idxsToIgnore:-1], guidePressures[species + '_Tissues'][idxsToIgnore:-1], label = species + '_Tissues', linewidth=lWidth, color='darkmagenta')

    '''

    
    if (options['legend']):
        ax[0,0].legend(prop={'size': legendSize},facecolor='white', framealpha=1)
        #ax[0,1].legend(prop={'size': legendSize})
        ax[1,0].legend(prop={'size': legendSize},facecolor='white', framealpha=1)
        ax[1,1].legend(prop={'size': legendSize},facecolor='white', framealpha=1)
        ax[2,0].legend(prop={'size': legendSize},facecolor='white', framealpha=1)
        ax[2,1].legend(prop={'size': legendSize},facecolor='white', framealpha=1)
        #fig.suptitle('Total Volume' + str(np.round(tot,1)))
        pass
    else:
        pass    

    ax[0,0].set_xlabel('Time (s)', fontsize = 5)
    ax[0,0].set_ylabel('Pressure (mmHg)', fontsize = 5)
    ax[0,0].set_title('PP of $N_2$ Capillaries vs Alveoli', size=7)

    ax[0,1].set_xlabel('Time (s)', fontsize = 5)
    ax[0,1].set_ylabel('Flow (mL/s)', fontsize = 5)
    ax[0,1].set_title('Flow $N_2$', size=7) 

    ax[1,0].set_xlabel('Time (s)', fontsize = 5)
    ax[1,0].set_ylabel('Pressure (mmHg)', fontsize = 5)
    ax[1,0].set_title('PP of $CO_2$ Capillaries vs Alveoli', size=7) 

    ax[1,1].set_xlabel('Time (s)', fontsize = 5)
    ax[1,1].set_ylabel('Flow (mL/s)', fontsize = 5)
    ax[1,1].set_title('Flow $CO_2$', size=7) 

    ax[2,0].set_xlabel('Time (s)', fontsize = 5)
    ax[2,0].set_ylabel('Pressure (mmHg)', fontsize = 5)
    ax[2,0].set_title('PP of $O_2$ Capillaries vs Alveoli', size=7) 

    ax[2,1].set_xlabel('Time (s)', fontsize = 5)
    ax[2,1].set_ylabel('Flow (mL/s)', fontsize = 5)
    ax[2,1].set_title('Flow $O_2$', size=7) 

    mpl.show()

# Fixed
def plotLungGasResultsPartialPressuresPlot(modelObject, options):
    t = modelObject.simulation['t']
    result = modelObject.solution
    
    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 5,
        }
    
    grid = {
        'top': 1, 
        'bottom': 4, 
        'right': 1, 
        'left': 4,
        'hspace': 0.1, 
        'wspace': 0.07
        }

    lWidth = options['lWidth']
    legendSize = options['legendSize']
    idxsToIgnore = options['idxsToIgnore']
   
    mpl.rcParams['figure.dpi'] = options['dpi']
    fig, ax = mpl.subplots(2, 2, constrained_layout=True)

    #[axi.xaxis.set_visible(False) for axi in ax.ravel()]
    [axi.tick_params(axis='both',labelsize=4, pad=1, direction='in',grid_linewidth = 1, labelrotation = 15) for axi in ax.ravel()]
    
    phaseChanges = np.where(np.roll(result[7]['resp'],1)!=result[7]['resp'])
    rectangles = np.zeros([int(len(phaseChanges[0])/2),2])
    counter = 0
    for idx,val in enumerate(phaseChanges[0]):
        if (idx % 2 == 0):
            rectangles[counter,0] = val
        else:
            rectangles[counter,1] = val - rectangles[counter,0]
            counter = counter + 1
    for rec in rectangles:
        start = int(rec[0])
        end = int(rec[0]+rec[1])
        ax[0,0].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[1,0].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[0,1].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[1,1].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
    
##### N2 Plot #############################################################
    tot = 0
    for key, value in result[1].items():    
        if (('Lu' in key) or ('2_u' in key) ) and ('N2' in key):
        #if (('Lu' in key) or ('2_u' in key) or ('Arterial' in key) or ('Venous' in key)) and ('N2' in key):
            ax[0,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth)
            tot = tot + value[-1]
##### O2 Plot ###########################################################
    for key, value in result[1].items():
        if (('Lu' in key) or ('2_u' in key)) and ('O2' in key):
        #if (('Lu' in key) or ('2_u' in key) or ('Arterial' in key) or ('Venous' in key)) and ('O2' in key):
            ax[0,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth)
    
##### Current Plot ############################################################
    for key, value in result[2].items():
        if ('Lu' in key) and (('2' in key)):
            ax[1,0].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key, linewidth=lWidth)
    
##### C2 Plot ######################################################   
    for key, value in result[1].items():
        if (('Lu' in key) or ('2_u' in key)) and ('C2' in key):
        #if (('Lu' in key) or ('2_u' in key) or ('Arterial' in key) or ('Venous' in key)) and ('C2' in key):
            ax[1,1].plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key , linewidth=lWidth)
            tot = tot + value[-1]
    

    # result_list = [elements[i] for i in indices]  
    # plt.legend(lines[:2], ['first', 'second']);

    if (options['legend']):
        #ax[0,0].legend(prop={'size': legendSize,})
        ax[0,0].legend(prop={'size': legendSize},facecolor='white', framealpha=1)
        ax[1,0].legend(prop={'size': legendSize},facecolor='white', framealpha=1)
        ax[0,1].legend(prop={'size': legendSize},facecolor='white', framealpha=1)
        ax[1,1].legend(prop={'size': legendSize},facecolor='white', framealpha=1)
        #fig.suptitle('Total Volume' + str(np.round(tot,1)))
    else:
        pass

    ax[0,0].set_xlabel('Time (s)', fontsize = 5)
    ax[0,0].set_ylabel('Pressure (mmHg)', fontsize = 5)
    ax[0,0].set_title('Partial Pressure of $N_2$', size=7)

    ax[0,1].set_xlabel('Time (s)', fontsize = 5)
    ax[0,1].set_ylabel('Pressure (mmHg)', fontsize = 5)
    ax[0,1].set_title('Partial Pressure of $O_2$', size=7)

    ax[1,0].set_xlabel('Time (s)', fontsize = 5)
    ax[1,0].set_ylabel('Flow (mL/s)', fontsize = 5)
    ax[1,0].set_title('Gas flow through the membrane', size=7)

    ax[1,1].set_xlabel('Time (s)', fontsize = 5)
    ax[1,1].set_ylabel('Pressure (mmHg)', fontsize = 5)
    ax[1,1].set_title('Partial Pressure of $CO_2$', size=7)
    mpl.show()

# Horrible Plot
def plotTotalPartialPressures(modelObject, options):
    t = modelObject.simulation['t']
    result = modelObject.solution
    
    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 5,
        }
    
    grid = {
        'top': 1, 
        'bottom': 0, 
        'right': 1, 
        'left': 0,
        'hspace': 0, 
        'wspace': 0.07
        }

    lWidth = options['lWidth']
    legendSize = options['legendSize']
    idxsToIgnore = options['idxsToIgnore']
   
    mpl.rcParams['figure.dpi'] = options['dpi']
    fig, ax = mpl.subplots(2, 2, gridspec_kw = grid)
    
    [axi.xaxis.set_visible(False) for axi in ax.ravel()]
    [axi.tick_params(axis='both',labelsize=4, pad=1, direction='in',grid_linewidth = 1, labelrotation = 15) for axi in ax.ravel()]
    
    phaseChanges = np.where(np.roll(result[7]['resp'],1)!=result[7]['resp'])
    rectangles = np.zeros([int(len(phaseChanges[0])/2),2])
    counter = 0
    for idx,val in enumerate(phaseChanges[0]):
        if (idx % 2 == 0):
            rectangles[counter,0] = val
        else:
            rectangles[counter,1] = val - rectangles[counter,0]
            counter = counter + 1
    for rec in rectangles:
        start = int(rec[0])
        end = int(rec[0]+rec[1])
        ax[0,0].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[1,0].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[0,1].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)
        ax[1,1].axvspan(t[start],t[end], facecolor='0.7', alpha=0.5)

    for compartment in modelObject.compartments:
        #ch = list(filter(lambda comp: comp['id'] == child, self.compartments))
        filtered_dict = {k:v for (k,v) in result[3].items() if compartment['id'] == k[4:]}
        
        array = np.zeros(len(t))
        for key, value in filtered_dict.items():
            array = np.add(array,np.array(value))
        
        if 'Lu' in compartment['id']:
            ax[0,0].plot(t[idxsToIgnore:-1], array[idxsToIgnore:-1], label = compartment['id'], linewidth=lWidth)
        elif (compartment['id'] == 'Hl') or (compartment['id'] == 'Al') or (compartment['id'] == 'As') or (compartment['id'] == 'Vp'):
            ax[0,1].plot(t[idxsToIgnore:-1], array[idxsToIgnore:-1], label = compartment['id'], linewidth=lWidth)
        elif (compartment['id'] == 'Hr') or (compartment['id'] == 'Ar') or (compartment['id'] == 'Vs') or (compartment['id'] == 'Ap'):
            ax[1,1].plot(t[idxsToIgnore:-1], array[idxsToIgnore:-1], label = compartment['id'], linewidth=lWidth)
        else:
            ax[1,0].plot(t[idxsToIgnore:-1], array[idxsToIgnore:-1], label = compartment['id'], linewidth=lWidth)
    
    if (options['legend']):
        ax[0,0].legend(prop={'size': legendSize})
        ax[1,0].legend(prop={'size': legendSize})
        ax[0,1].legend(prop={'size': legendSize})
        ax[1,1].legend(prop={'size': legendSize})
        #fig.suptitle('Total Volume' + str(np.round(tot,1)))
    else:
        pass

    mpl.show()

'''
..####...##..##..##..##..........#####...##.......####...######..........##...##..######..######..##..##...####...#####....####..
.##..##..##..##...####...........##..##..##......##..##....##............###.###..##........##....##..##..##..##..##..##..##.....
.######..##..##....##............#####...##......##..##....##............##.#.##..####......##....######..##..##..##..##...####..
.##..##..##..##...####...........##......##......##..##....##............##...##..##........##....##..##..##..##..##..##......##.
.##..##...####...##..##..........##......######...####.....##............##...##..######....##....##..##...####...#####....####..                                                      
'''
# generates 3 plots -> arterial / capilaries / venous for a result type -> volume, pressure, current
# returns 3 arrays that contain the cumulative sum of all results of the tree
# one for each arterial / capilaries / venous
def plotTreeResults(modelObject, options, resultType, grid, colormap, idxsToIgnore, lWidth, bloodTot):
    
    t = modelObject.simulation['t']
    results = modelObject.solution
    
    
    mpl.rcParams['figure.dpi'] = options['dpi']
    fig2, ([ax1, ax2, ax3]) = mpl.subplots(3, 1, constrained_layout=True)
    #ax1.xaxis.set_visible(False)
    #ax2.xaxis.set_visible(False)
    #ax3.xaxis.set_visible(False)

    if resultType == 0 :
        #fig2.suptitle('Volume', fontsize=16)
        ax1.set_xlabel('Time (s)', fontsize = 5)
        ax1.set_ylabel('Volume (mL)', fontsize = 5)
        ax1.set_title('Volume on the Arteries', size=7)

        ax2.set_xlabel('Time (s)', fontsize = 5)
        ax2.set_ylabel('Volume (mL)', fontsize = 5)
        ax2.set_title('Volume on the Capillaries', size=7)

        ax3.set_xlabel('Time (s)', fontsize = 5)
        ax3.set_ylabel('Volume (mL)', fontsize = 5)
        ax3.set_title('Volume on the Veins', size=7)
    elif resultType == 1 :
        #fig2.suptitle('Pressure', fontsize=16)
        ax1.set_xlabel('Time (s)', fontsize = 5)
        ax1.set_ylabel('Pressure (mmHg)', fontsize = 5)
        ax1.set_title('Pressure on the Arteries', size=7)

        ax2.set_xlabel('Time (s)', fontsize = 5)
        ax2.set_ylabel('Pressure (mmHg)', fontsize = 5)
        ax2.set_title('Pressure on the Capillaries', size=7)

        ax3.set_xlabel('Time (s)', fontsize = 5)
        ax3.set_ylabel('Pressure (mmHg)', fontsize = 5)
        ax3.set_title('Pressure on the Veins', size=7)
    elif resultType == 2 :
        #fig2.suptitle('Current', fontsize=16)
        ax1.set_xlabel('Time (s)', fontsize = 5)
        ax1.set_ylabel('Flow (mL/s)', fontsize = 5)
        ax1.set_title('Flow on the Arteries', size=7)

        ax2.set_xlabel('Time (s)', fontsize = 5)
        ax2.set_ylabel('Flow (mL/s)', fontsize = 5)
        ax2.set_title('Flow on the Capillaries', size=7)

        ax3.set_xlabel('Time (s)', fontsize = 5)
        ax3.set_ylabel('Flow (mL/s)', fontsize = 5)
        ax3.set_title('Flow on the Veins', size=7)

    capilaryArray = np.zeros([len(t)])
    arterialArray = np.zeros([len(t)])
    venousArray = np.zeros([len(t)])
    
    if resultType == 1 :
        for key, value in results[resultType].items():
            if '2' not in key:
                if 'Ap' in key:
                    colorValR = colormap.to_rgba(len(key))                
                    mean = int(np.round(np.mean(value- modelObject.lungParams['atmosphericPressure']),1))
                    amp = np.round(max(value)-min(value),1)
                    ax1.plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1]- modelObject.lungParams['atmosphericPressure'], label = key + ' -> ' + str(mean) + ' -> ' + str(amp), linewidth=lWidth, color=colorValR)
                    arterialArray = np.array([x+y for x,y in zip(arterialArray, value)])
                elif 'Cp' in key:                
                    mean = int(np.round(np.mean(value- modelObject.lungParams['atmosphericPressure']),1))
                    amp = np.round(max(value)-min(value),1)
                    ax2.plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1]- modelObject.lungParams['atmosphericPressure'], label = key + ' -> ' + str(mean) + ' -> ' + str(amp), linewidth=lWidth)
                    capilaryArray = np.array([x+y for x,y in zip(capilaryArray, value)])
                elif 'Vp' in key: 
                    colorValR = colormap.to_rgba(len(key))                
                    mean = int(np.round(np.mean(value- modelObject.lungParams['atmosphericPressure']),1))
                    amp = np.round(max(value)-min(value),1)
                    ax3.plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1]- modelObject.lungParams['atmosphericPressure'], label = key + ' -> ' + str(mean) + ' -> ' + str(amp), linewidth=lWidth, color=colorValR)
                    venousArray = np.array([x+y for x,y in zip(venousArray, value)])
    else:
        for key, value in results[resultType].items():
            if '2' not in key:
                if 'Ap' in key:
                    colorValR = colormap.to_rgba(len(key))                
                    mean = int(np.round(np.mean(value),1))
                    amp = np.round(max(value)-min(value),1)
                    ax1.plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key + ' -> ' + str(mean) + ' -> ' + str(amp), linewidth=lWidth, color=colorValR)
                    arterialArray = np.array([x+y for x,y in zip(arterialArray, value)])
                    if resultType == 0 : bloodTot = bloodTot + mean
                elif 'Cp' in key:                
                    mean = int(np.round(np.mean(value),1))
                    amp = np.round(max(value)-min(value),1)
                    ax2.plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key + ' -> ' + str(mean) + ' -> ' + str(amp), linewidth=lWidth)
                    capilaryArray = np.array([x+y for x,y in zip(capilaryArray, value)])
                    if resultType == 0 : bloodTot = bloodTot + mean
                elif 'Vp' in key: 
                    colorValR = colormap.to_rgba(len(key))                
                    mean = int(np.round(np.mean(value),1))
                    amp = np.round(max(value)-min(value),1)
                    ax3.plot(t[idxsToIgnore:-1], value[idxsToIgnore:-1], label = key + ' -> ' + str(mean) + ' -> ' + str(amp), linewidth=lWidth, color=colorValR)
                    venousArray = np.array([x+y for x,y in zip(venousArray, value)])
                    if resultType == 0 : bloodTot = bloodTot + mean
    
    
    mpl.show()       
    if resultType == 1 :
        print('Mean Capilary pressure is: -> ' + str(np.mean(capilaryArray)))
    
    return arterialArray, capilaryArray, venousArray

'''
.#####...##..##..##.......####...######..........#####...##.......####...######...####..
.##..##..##..##..##......##......##..............##..##..##......##..##....##....##.....
.#####...##..##..##.......####...####............#####...##......##..##....##.....####..
.##......##..##..##..........##..##..............##......##......##..##....##........##.
.##.......####...######...####...######..........##......######...####.....##.....####..
'''                                       

def cutPulses(modelObject,resultType):
    results = modelObject.solution[resultType]
    t = modelObject.simulation['t']
    sampPeriod = modelObject.simulation['samplingPeriod']
    HC = modelObject.params['HC']
    
    pulseDuration = int(HC*(1/sampPeriod))
    totalPulses = int(len(t)/(pulseDuration))
    
    #pulses = np.zeros([totalPulses,pulseDuration,len(results)])
    pulses = {}
    
    for key, value in results.items():
        pulses[key] = list(value[i:i+pulseDuration] for i in range(0, len(value), pulseDuration))
       
    return pulses
    
def plotPulses(pulses,key,options):
    mpl.figure()
    mpl.rcParams['figure.dpi'] = options['dpi']
    for pulse in pulses[key]:
        mpl.plot(pulse)
    mpl.title = key
    
def plotElastanceCapacity(modelObject, options):
    t = modelObject.simulation['t']
    result = modelObject.solution
    grid = {
        'top': 1,
        'bottom': 0,
        'right': 1,
        'left': 0,
        'hspace': 0,
        'wspace': 0.07
        }
    Tsys = 0.3 * np.sqrt(1)  # Systole duration (seconds)
    lWidth = options['lWidth']

    keys = np.array(list(result[0].keys()))
    sizes = np.char.str_len(keys)
    mpl.rcParams['figure.dpi'] = options['dpi']
    fig, ax = mpl.subplots(2, 1)
    ax[0].plot(t, result[5], label = 'elastance', linewidth=2)
    #mpl.title('Elastance')
    ax[1].plot(t, result[6], label = 'capacity', linewidth=2)
    #mpl.title('Capacity')
    ax[0].xaxis.set_visible(False)
    ax[0].axvspan(0,Tsys, facecolor='0.7', alpha=0.5)
    ax[1].axvspan(0,Tsys, facecolor='0.7', alpha=0.5)
    ax[0].set_title('Elastance')
    ax[1].set_title('Capacitance')

    mpl.show()
