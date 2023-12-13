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
import tree
import math
import pandas as pd



def generate_latex_table(data_dict, column_names,header,name):
    # Begin the LaTeX table environment
    latex_table ="\\begin{table}\n"
    latex_table +="\\centering\n"
    latex_table += "\\begin{tabular}{|c|" + "|".join(["c"] * len(column_names)) + "|}\n"
    latex_table += "\\hline\n"

    # Add the column names as the table header
    latex_table += "\multicolumn{1}{|c|}{\multirow{2}{*}{\\textbf{"+ name +"}}} & \multicolumn{" + str(len(column_names)) + "}{|c|}{\\textbf{" + header + "}} \\\\ \\cline{2-" + str(len(column_names)+1) + "} \n"
    latex_table +=  "& " + " & ".join(map(str, column_names)) + " \\\\ \\hline \n"

    # Iterate through the dictionary and populate the table
    for key, values in data_dict.items():
        row = [key] + values
        latex_table += " & ".join(map(str, row)) + " \\\\\n"
        latex_table += "\\hline\n"

    # End the LaTeX table environment
    latex_table += "\\end{tabular}"
    latex_table += "\\caption{PLEASE COMPLETE ME!!}\n"
    latex_table += "\\end{table}"

    return latex_table

def generateCardioResultsTable(states,results,totalBloodVolume,atmPressure):
    
    header_vol = 'Volume \%'
    column_names_vol = ['Min','Avg', 'Max','Lit Values']
    tableVolume = {
        'Systemic Artery': [(min(results['V_As'])/totalBloodVolume)*100, (np.mean(results['V_As'])/totalBloodVolume)*100, (max(results['V_As'])/totalBloodVolume)*100,'13'],
        'Systemic Capillaries': [(min(results['V_Cs'])/totalBloodVolume)*100, (np.mean(results['V_Cs'])/totalBloodVolume)*100, (max(results['V_Cs'])/totalBloodVolume)*100,'7'],
        'Systemic Veins': [(min(results['V_Vs'])/totalBloodVolume)*100, (np.mean(results['V_Vs'])/totalBloodVolume)*100, (max(results['V_Vs'])/totalBloodVolume)*100,'64'],
        'Thoracic Veins': [(min(results['V_Vt'])/totalBloodVolume)*100, (np.mean(results['V_Vt'])/totalBloodVolume)*100, (max(results['V_Vt'])/totalBloodVolume)*100,'-'],
        'Right Ventricle': [(min(results['V_Hr'])/totalBloodVolume)*100, (np.mean(results['V_Hr'])/totalBloodVolume)*100, (max(results['V_Hr'])/totalBloodVolume)*100,'-'],
        'Pulmonary Artery': [(min(results['V_Ap'])/totalBloodVolume)*100, (np.mean(results['V_Ap'])/totalBloodVolume)*100, (max(results['V_Ap'])/totalBloodVolume)*100,'-'],
        'Pulmonary Capillaries': [(min(results['V_Cp'])/totalBloodVolume)*100, (np.mean(results['V_Cp'])/totalBloodVolume)*100, (max(results['V_Cp'])/totalBloodVolume)*100,'-'],
        'Pulmonary Veins': [(min(results['V_Vp'])/totalBloodVolume)*100, (np.mean(results['V_Vp'])/totalBloodVolume)*100, (max(results['V_Vp'])/totalBloodVolume)*100,'-'],
        'Left Ventricle': [(min(results['V_Hl'])/totalBloodVolume)*100, (np.mean(results['V_Hl'])/totalBloodVolume)*100, (max(results['V_Hl'])/totalBloodVolume)*100,'-'],
        'Pulmonary Vessels': [min(((results['V_Ap'] + results['V_Cp'] + results['V_Vp'])/totalBloodVolume)*100), np.mean(((results['V_Ap'] + results['V_Cp'] + results['V_Vp'])/totalBloodVolume)*100), max(((results['V_Ap'] + results['V_Cp'] + results['V_Vp'])/totalBloodVolume)*100),'9'],
        'Systemic Circulation': [min(((results['V_As'] + results['V_Cs'] + results['V_Vs'] + results['V_Vt'])/totalBloodVolume)*100), np.mean(((results['V_As'] + results['V_Cs'] + results['V_Vs'] + results['V_Vt'])/totalBloodVolume)*100),max(((results['V_As'] + results['V_Cs'] + results['V_Vs'] + results['V_Vt'])/totalBloodVolume)*100),'84'],
        'Heart': [min(((results['V_Hl'] + results['V_Hr'])/totalBloodVolume)*100),np.mean(((results['V_Hl'] + results['V_Hr'])/totalBloodVolume)*100), max(((results['V_Hl'] + results['V_Hr'])/totalBloodVolume)*100),'3'],
    }
    for key,val in tableVolume.items():
        tableVolume[key] = [round(values,1) if (isinstance(values, (int, float, complex))) else values for values in val]
    latex_table_vol = generate_latex_table(tableVolume, column_names_vol,header_vol,'Compartment')
    ########################################################################################################################################################################################
    header_pre = 'Pressure mmHg'
    column_names_pre = ['Min','Avg','Max','Literature Diastolic/Min','Literature Systolic/Max']
    tablePressure = {
        'Systemic Artery': [min(results['P_As']) - atmPressure, np.mean(results['P_As']) - atmPressure, max(results['P_As']) - atmPressure,'60-90','90-140'],
        'Systemic Capillaries': [min(results['P_Cs']) - atmPressure, np.mean(results['P_Cs']) - atmPressure, max(results['P_Cs']) - atmPressure,'-','-'],
        'Systemic Veins': [min(results['P_Vs']) - atmPressure, np.mean(results['P_Vs']) - atmPressure, max(results['P_Vs']) - atmPressure,'-','-'],
        'Thoracic Veins': [min(results['P_Vt']) - atmPressure, np.mean(results['P_Vt']) - atmPressure, max(results['P_Vt']) - atmPressure,'0-8','2-14'],
        'Right Ventricle': [min(results['P_Hr']) - atmPressure, np.mean(results['P_Hr']) - atmPressure, max(results['P_Hr']) - atmPressure,'0-8','15-28'],
        'Pulmonary Artery': [min(results['P_Ap']) - atmPressure, np.mean(results['P_Ap']) - atmPressure, max(results['P_Ap']) - atmPressure,'5-16','15-28'],
        'Pulmonary Capillaries': [min(results['P_Cp']) - atmPressure, np.mean(results['P_Cp']) - atmPressure, max(results['P_Cp']) - atmPressure,'1-12','9-23'],
        'Pulmonary Veins': [min(results['P_Vp']) - atmPressure, np.mean(results['P_Vp']) - atmPressure, max(results['P_Vp']) - atmPressure,'-','-'],
        'Left Ventricle': [min(results['P_Hl']) - atmPressure, np.mean(results['P_Hl']) - atmPressure,  max(results['P_Hl']) - atmPressure,'4-12','90-140'],
    }
    for key,val in tablePressure.items():
        tablePressure[key] = [round(values,1) if (isinstance(values, (int, float, complex))) else values for values in val]
    latex_table_pre = generate_latex_table(tablePressure, column_names_pre,header_pre,'Compartment')

    print('############################ Volume #####################################')
    print(latex_table_vol)
    print('########################### Pressure ####################################')
    print(latex_table_pre)
    print('#########################################################################')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('#########################################################################')
    print('########## -> Pressure <- ##########')
    for key,val in tablePressure.items():
        print(f"{key}: {val}")
        #print(key + ' : ' + str(val))
    print('#########################################################################')
    print('########## -> Volume <- ##########')
    for key,val in tableVolume.items():
        print(f"{key}: {val}")
        #print(key + ' : ' + str(val))
    print('#########################################################################')

    return latex_table_vol,latex_table_pre

def generateHeartResultsTable(states,results,atmPressure):
    
    cycleDuration = states['HC']
    nrCycles = int(len(results['HC'])/(states['HC']*100))

    heartTable = {
        'LV EF': [],
        'RV EF': [],
        'LV SV': [],
        'RV SV': [],
        'LV EDV': [],
        'RV EDV': [],
    }
    for i in range(nrCycles):
        cycleStart = int(i*cycleDuration*100)
        cycleEnd = int((i+1)*cycleDuration*100)
        cycleResultsHl = results['V_Hl'][cycleStart:cycleEnd]
        cycleResultsHr = results['V_Hr'][cycleStart:cycleEnd]

        lvSv = round(max(cycleResultsHl) - min(cycleResultsHl),2)
        rvSv = round(max(cycleResultsHr) - min(cycleResultsHr),2)
        lvEf = round(((max(cycleResultsHl) - min(cycleResultsHl))/max(cycleResultsHl))*100,2)
        rvEf = round(((max(cycleResultsHr) - min(cycleResultsHr))/max(cycleResultsHr))*100,2)
        lvEdv = round(max(cycleResultsHl),2)
        rvEdv = round(max(cycleResultsHr),2)

        heartTable['LV SV'].append(lvSv)
        heartTable['RV SV'].append(rvSv)
        heartTable['LV EF'].append(lvEf)
        heartTable['RV EF'].append(rvEf)
        heartTable['LV EDV'].append(lvEdv)
        heartTable['RV EDV'].append(rvEdv)
    

    heartTable_Gen = {
        'LV EF': [],
        'RV EF': [],
        'LV SV': [],
        'RV SV': [],
        'LV EDV': [],
        'RV EDV': [],
    }
    for key,values in heartTable.items():
        average = round(np.mean(values),1)
        

        if 'LV' in key:
            peakInspiration = round(100*(min(values)-average)/average,2)
            peakExpiration = round(100*(max(values)-average)/average,2)
        else:
            peakInspiration = round(100*(max(values)-average)/average,2)
            peakExpiration = round(100*(min(values)-average)/average,2)

        heartTable[key] = values + [average,peakInspiration,peakExpiration]
        heartTable_Gen[key] = [average,peakInspiration,peakExpiration]


    heartTable_Gen['LV EF'] = heartTable_Gen['LV EF'] + [65,65]
    heartTable_Gen['RV EF'] = heartTable_Gen['RV EF'] + [60,60]
    heartTable_Gen['LV SV'] = heartTable_Gen['LV SV'] + ['-4 +-7','+4 +-7']
    heartTable_Gen['RV SV'] = heartTable_Gen['RV SV'] + ['+10 +-5','-10 +-5']
    heartTable_Gen['LV EDV'] = heartTable_Gen['LV EDV'] + ['-3','+5 +-7']
    heartTable_Gen['RV EDV'] = heartTable_Gen['RV EDV'] + ['+9','-9']
    


    
    header = 'Volume \%'
    column_names = [str(i+1) for i in range(nrCycles)]
    column_names_Gen = ['Avg','Peak Ins','Peak Exp','Lit Peak Ins','Lit Peak Exp']

    latex_table_heart = generate_latex_table(heartTable, column_names,header,' ')
    latex_table_heart_Gen = generate_latex_table(heartTable_Gen, column_names_Gen,header,' ')
    
    print('#########################################################################')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('#########################################################################')
    print('############################ Heart Cycles ###############################')
    print(latex_table_heart_Gen)


    return latex_table_heart_Gen

def downsample_dictionary(original_dict, original_sampling_rate, new_sampling_rate):
    downsampled_dict = {}

    for key, original_array in original_dict.items():
        # Calculate the downsampling factor
        downsample_factor = int(original_sampling_rate / new_sampling_rate)

        # Downsample the array using NumPy's array slicing
        downsampled_array = original_array[::downsample_factor]

        # Store the downsampled array in the new dictionary
        downsampled_dict[key] = downsampled_array

    return downsampled_dict

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

def calculateVariablesForPlotting(results, totalTime, sampPeriod, atmPressure, modelObjects,idx,structures):
    windowSize = int(results['RC'][-1]/sampPeriod)
    avgPressure = np.zeros(int(totalTime/sampPeriod))
    counter = 0
    for key,value in results.items():
        if key.startswith('P_La') :
            avgPressure = np.array(avgPressure) + np.array(value)
            counter = counter + 1
    avgPressure = avgPressure / counter
    

    volumeAlveoli = np.zeros(int(totalTime/sampPeriod))
    volumeBronchi = np.zeros(int(totalTime/sampPeriod))
    volumeBloodLungs = np.zeros(int(totalTime/sampPeriod))
    volumeBloodSystemic = np.zeros(int(totalTime/sampPeriod))
    volumeBloodHeart = np.zeros(int(totalTime/sampPeriod))
    volumeBloodThorax = np.zeros(int(totalTime/sampPeriod))
    volumeBloodPleura = np.zeros(int(totalTime/sampPeriod))

    for key,value in results.items():
        if key.startswith('V_La'):
            volumeAlveoli = volumeAlveoli + np.array(value)
        if key.startswith('V_Lb'):
            volumeBronchi = volumeBronchi + np.array(value)
        if ('V_Hl' in key) or ('V_Hr' in key):
            volumeBloodHeart = volumeBloodHeart + np.array(value)
        if ('V_Ap' in key) or ('V_Cp' in key) or ('V_Vp' in key):
            volumeBloodLungs = volumeBloodLungs + np.array(value)

        if ('V_As' in key) or ('V_Cs' in key) or ('V_Vs' in key):
            volumeBloodSystemic = volumeBloodSystemic + np.array(value)
        if ('V_Hl' in key) or ('V_Hr' in key) or ('V_Ap' in key) or ('V_Cp' in key) or ('V_Vp' in key) or ('V_Vt' in key):
            volumeBloodThorax = volumeBloodThorax + np.array(value)
        if ('V_Cp' in key) or ('V_Vp' in key):
            volumeBloodPleura = volumeBloodPleura + np.array(value)
        
    totalLungAirVolume = volumeAlveoli + volumeBronchi
    volumeThx = volumeBloodThorax + totalLungAirVolume 
    volumePlr = volumeBloodPleura + volumeAlveoli

    results['V_Alveoli'] = volumeAlveoli
    results['V_Bronchi'] = volumeBronchi
    results['V_Thx'] = volumeThx
    results['totalPleuraVolume'] = volumePlr
    results['V_BloodLungs'] = volumeBloodLungs
    results['V_BloodSystemic'] = volumeBloodSystemic
    results['V_BloodHeart'] = volumeBloodHeart
    results['V_BloodThorax'] = volumeBloodThorax
    results['P_AvgLa'] = avgPressure

    tidalVolume = np.zeros(int(totalTime/sampPeriod))
    for i in np.arange(0,len(totalLungAirVolume ),windowSize):
        maxVal = max(totalLungAirVolume[i:i+windowSize])
        minVal = min(totalLungAirVolume[i:i+windowSize])
        tidalVolume[i:i+windowSize] = tidalVolume[i:i+windowSize] + np.round((maxVal - minVal),0)
    results['tidalVolume'] = tidalVolume
    results['totalLungAirVolume']= totalLungAirVolume
    results['tidalVolume'] = tidalVolume

    results['uVol'] = plots.calculateThoraxUnstressedVolume(modelObjects,results)

    return results



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
        'totalBloodVolume' : 5000.0, # Total Blood Volume not in use yet
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
        'E': 2,
        'slopeFraction': 0.2,
    }

    # Names for the nodes to use in the model
    compartmentNames = ['As', 'Cs', 'Vs', 'Vt', 'Hr', 'Ap', 'Cp', 'Vp', 'Hl', 'Atm', 'Thx', 'Plr', 'Tis', 'Ven', 'Per']

    ############################################### Connectivity Matrix  ####################################################
    #                               As Cs Vs Vt Hr Ap Cp Vp Hl Atm Thx Plr Tis Ven Per col/row
    connectivityMatrix = np.array([ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  # As
                                    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  # Cs
                                    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  # Vs
                                    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  # Vt
                                    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  0,  0,  0,  # Hr
                                    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,  0,  0,  0,  # Ap
                                    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,  0,  0,  0,  # Cp
                                    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,  0,  0,  0,  # Vp
                                    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  # Hl
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  # Atm
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  # Thx
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  # Plr
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  # Tis
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  # Ven
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  # Per
                                        ]).reshape(15, 15)


    ############################################### Resistor Parameters  ####################################################
    # Reisitors parameters (Names and number of resistors should be calculated by the connectivityMatrix) 
    resistorNames = ['AsCs', 'CsVs', 'VsVt', 'VtHr', 'HrAp', 'ApCp', 'CpVp', 'VpHl', 'HlAs']

    #                                     R       L       y0  type
    resistorParamsMatrix = np.array([   0.901, 0.00000, 0.000, 1,  # AsCs #!! controlled
                                        0.080, 0.00000, 0.000, 1,  # CsVs
                                        0.040, 0.00000, 0.000, 1,  # VsVt
                                        0.005, 0.00001, 0.000, 0,  # VtHr
                                        0.006, 0.00010, 0.000, 0,  # HrAp
                                        0.0575, 0.00000, 0.000, 1,  # ApCp #!! controlled
                                        0.02, 0.00000, 0.000, 1,  # CpVp #!! -0.005
                                        0.005, 0.00001, 0.000, 0,  # VpHl #!! -0.005
                                        0.006, 0.00010, 0.000, 0,  # HlAs
                                        ]).reshape(9, 4)


    ############################################### Capacitor Parameters  ####################################################
    #                                      C       V0     Emax    Emin   Trise    y0   type
    capacitorsParamsMatrix = np.array([ 1.6500, 500.00, 0.0000, 0.0000, 0.0000, 572.0, 2,  # As
                                        25.351, 0.0000, 0.0000, 0.0000, 0.0000, 343.0, 1,  # Cs
                                        438.51, 1500.0, 0.0000, 0.0000, 0.0000, 3144.0, 1,  # Vs #!!
                                        28.007, 000.00, 0.0000, 0.0000, 0.0000, 234.0, 1,  # Vt
                                        0.0000, 0.0000, 0.6172, 0.1000, 0.0000, 100.0, 0,  # Hr 
                                        3.1833, 090.00, 0.0000, 0.0000, 0.0000, 191.0, 2,  # Ap
                                        10.526, 10.0000, 0.0000, 0.0000, 0.0000, 96.0, 2,  # Cp
                                        23.889, 20.00, 0.0000, 0.0000, 0.0000, 195.0, 2,  # Vp
                                        0.0000, 0.0000, 2.7777, 0.1443, 0.0000, 125.0, 0,  # Hl 
                                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 760.00, 5,  # Atm
                                        300.00, 3500.0, 3800.0, 3100.0, 0.0000, 755.59, 3,  # Thx
                                        50.000, 2500.0, 4300.0, 3600.0, 0.0000, 757.30, 4,  # Plr
                                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 760.00, 5,  # Tis
                                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 760.00, 6,  # Ven
                                        70.000, 000.00, 0.0000, 0.0000, 0.0000, 0.0000, 3,  # Per

                                        ]).reshape(15, 7)

    ############################################### Pressure Bias Map  ####################################################    
    #                                       As Cs Vs Vt Hr Ap Cp Vp Hl Atm Thx Plr Tis Ven Per
    connectivityPresBiasMatrix = np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # As
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Cs
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Vs
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Vt
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Hr
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Ap
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Cp
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Vp
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Hl
                                            1, 1, 1, 0, 0, 0, 0, 0, 0, 1,  1,  0,  1,  1,  0,  # Atm
                                            0, 0, 0, 1, 1, 1, 0, 1, 1, 0,  0,  1,  0,  0,  1,  # Thx
                                            0, 0, 0, 0, 0, 0, 1, 0, 0, 0,  0,  0,  0,  0,  0,  # Plr
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Tis
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Ven
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Per
                                        ]).reshape(15, 15)


    ############################################### Region Volume Map  ####################################################
    #                                       As Cs Vs Vt Hr Ap Cp Vp Hl Atm Thx Plr Tis Ven Per
    connectivityRegionVolMatrix = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # As
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Cs
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Vs
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Vt
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Hr
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Ap
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Cp
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Vp
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Hl
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Atm
                                            0, 0, 0, 1, 1, 1, 1, 1, 1, 0,  0,  0,  0,  0,  0,  # Thx
                                            0, 0, 0, 0, 0, 0, 1, 1, 0, 0,  0,  0,  0,  0,  0,  # Plr
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Tis
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Ven
                                            0, 0, 0, 0, 1, 0, 0, 0, 1, 0,  0,  0,  0,  0,  0,  # Per
                                        ]).reshape(15, 15)


    ############################################### Membrane Resisros Map Map  ############################################
    #                                           As Cs Vs Vt Hr Ap Cp Vp Hl Atm Thx Plr Tis Ven Per
    connectivityMemResistorsMatrix = np.array([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0, # As
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Cs
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Vs
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Vt
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Hr
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Ap
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Cp
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Vp
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Hl
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Atm
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Thx
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Plr
                                                0, 1, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Tis
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Ven
                                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Per
                                        ]).reshape(15, 15)

    ############################################### Cycles and Durations  #################################################

    cycles = ['HC', 'RC']
    #cyclesDuration = [0.9561, 2.8685]
    cyclesDuration = [0.8, 5.0]
    #                                    As Cs Vs Vt Hr Ap Cp Vp Hl Atm Thx Plr Tis Ven Per
    cyclesDistributionMatrix = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0,  0,  0,  0,  0, 0,  # HC
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  1,  0,  0,  1, 0,  # RC
                                    ]).reshape(2, 15)
    
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
    #                                   As Cs Vs Vt Hr Ap Cp Vp Hl Atm Thx Plr Tis Ven
    gasDistributionMatrix = np.array([  0, 0, 0, 0, 0, 0, 0, 0, 0, 1,  0,  0,  0,  1,  1,  # Atmosphere
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Alveoli
                                        1, 0, 0, 0, 0, 0, 1, 1, 1, 0,  0,  0,  0,  0,  0,  # Arterial
                                        0, 1, 1, 1, 1, 1, 0, 0, 0, 0,  0,  0,  0,  0,  0,  # Venous
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  1,  0,  0,  # Tissues
                                    ]).reshape(5, 15)
    
    ############################################### Controllers  ##########################################################
    #                      period  varName    varInt  Type  y0
    integrators = np.array(['HC', 'i_Hl',    'Q_HlAs', 0, 69.810,  # i_Hl
                            'HC', 'i_COl',   'i_Hl',   1, 70.920,  # i_COl
                            'HC', 'i_Hr',    'Q_HrAp', 0, 68.530,  # i_Hr
                            'HC', 'i_COr',   'i_Hr',   1, 67.980,  # i_COr
                            'HC', 'i_MaxAp', 'P_Ap',   2, 784.18,  # i_RC
                            'HC', 'i_Max_Ap','i_MaxAp',1, 785.07,  # i_TV
                            'HC', 'i_MinAp', 'P_Ap',   3, 774.84,  # i_RC
                            'HC', 'i_Min_Ap','i_MinAp',1, 774.98,  # i_TV
                            'HC', 'i_MaxAs', 'P_As',   2, 861.96,  # i_RC
                            'HC', 'i_Max_As','i_MaxAs',1, 862.53,  # i_TV
                            'HC', 'i_MinAs', 'P_As',   3, 833.82,  # i_RC
                            'HC', 'i_Min_As','i_MinAs',1, 833.94,  # i_TV
                            #'RC', 'i_MaxLa', 'V_La',   2, 3000.0,  # i_RC
                            #'RC', 'i_Max',   'i_MaxLa',1, 3000.0,  # i_TV
                            #'RC', 'i_MinLa', 'V_La',   3, 2501.4,   # i_RC
                            #'RC', 'i_Min',   'i_MinLa',1, 2501.4,  # i_TV
                    ]).reshape(12, 5)
    
    #                 AverageVar  ofVar    AtmPress Period type
    averages = np.array(['a_As', 'P_As',   'P_Atm',  3.0,   0,  # As #87.11
                         'a_Ap', 'P_Ap',   'P_Atm',  3.0,   0,  # Ap #17.76
                         'a_Vs', 'P_Vs',   'P_Atm',  3.0,   0,  # Ap #17.76
                         'a_COl', 'i_COl', 71.61,     1.0,    1,  # COl
                         'a_COr', 'i_COr', 69.54,     1.0,    1,  # COr
                         #'a_Min', 'i_Min', 2502.3,   1.0,    1,  # TV
                         #'a_Max', 'i_Max', 2995.5,   1.0,    1,  # TV
                    ]).reshape(5, 5)
    
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

def initModelWithTree(ventilator,control = True, calibration = False):
    modelStructure = modelStructureAndParameters(gasExchange=False,control=True)

    totalBloodVolume = 5000.0
    target_Hl = 105.00
    target_Hr = 105.00
    e_C_Hr_Min = 30.0
    e_C_Hr_Max = 4.0
    e_C_Hl_Min = 15.0
    e_C_Hl_Max = 2.0
    if control:
        if calibration:
            #                                         TargetVar ToCtrllVar TargetVal MinVal MaxVal     Kp    Kd  type
            modelStructure['controllers'] = np.array([
                                                    'V_Hr',   'e_C_Hr',  target_Hr,   e_C_Hr_Min, e_C_Hr_Max,  0.080, 3.000,  2,  # e_C_Hr
                                                    'V_Hl',   'e_C_Hl',  target_Hl,   e_C_Hl_Min, e_C_Hl_Max,  0.080, 3.000,  2,  # e_C_Hl
                                                    
                                                    'a_As',   'E_C_Hl',   90.84,   0.010,  200.00,  0.02, 3.0,  0,  # E_Hl
                                                    'a_Ap',   'E_C_Hr',   17.2,    0.010,  200.00,  0.02, 3.0,  0,  # E_Hr
                                                    'a_COl',  'R_AsCs',   68.4,    0.200,  4.0000,  0.01, 3.0,  1,  # As
                                                    'a_COr',  'R_ApCp',   68.4,    0.010,  0.6000,  0.01, 3.0,  1,  # Ap

                                                    #'',  'R_AsCs',   68.4,    0.000,  0.0000,  0.00, 0.0,  0,  # As
                                                    #'',  'R_ApCp',   68.4,    0.000,  0.0000,  0.00, 0.0,  0,  # Ap

                                                    'V_As',  'C_As',   0.13 * totalBloodVolume,    0.010,  10.0,  0.02, 3.0,  0,
                                                    'V_Cs',  'C_Cs',   0.07 * totalBloodVolume,    0.010,  50.0,  0.02, 3.0,  0,
                                                    'a_Vs',  'C_Vs',   10.0                      ,    150.0,  1400.0,  0.02, 3.0,  1,
                                                    'V_Vt',  'C_Vt',   0.03 * totalBloodVolume,    0.010,  50.0,  0.02, 3.0,  0,
                                                    'V_Ap',  'C_Ap',   0.03 * totalBloodVolume,    0.010,  10.0,  0.02, 3.0,  0,
                                                    'V_Cp',  'C_Cp',   0.03 * totalBloodVolume,    0.010,  15.0,  0.02, 3.0,  0,
                                                    'V_Vp',  'C_Vp',   0.06 * totalBloodVolume,    0.010,  50.0,  0.02, 3.0,  0,
                                                    
                                                    #'',  'C_Plr',   2.0,    0.000,  0.0000,  0.00, 0.0,  5,  # 
                                                     
                            ]).reshape(14, 8)
        else:
            modelStructure['controllers'] = np.array([
                                                    'V_Hr',   'e_C_Hr',  target_Hr,   e_C_Hr_Min, e_C_Hr_Max,  0.080, 3.000,  2,  # e_C_Hr
                                                    'V_Hl',   'e_C_Hl',  target_Hl,   e_C_Hl_Min, e_C_Hl_Max,  0.080, 3.000,  2,  # e_C_Hl
                                                    #'',  'C_Plr',   2.0,    0.000,  0.0000,  0.00, 0.0,  5,  #
                                                    'V_Cp',  'R_CpVp',   30,    24.000,  0.02000,  0.08, 0.0,  6,  # Ap
                                                    'V_Vp',  'R_VpHl',   30,    48.000,  0.00500,  0.08, 0.0,  6,  # Ap
                                                    #'V_Hl',  'R_HlAs',   30,    12.000,  0.00600,  0.08, 0.0,  6,  # Ap
                            ]).reshape(4, 8)
    else:
        modelStructure['controllers'] = np.array([]).reshape(0, 8)

    # tree params
    params = {
        ############# Resistance
        'R_Lt': 0.002,
        'R_Lb': 0.008,
        'R_La': 0.002,
        ############# Compliance
        'C_Lt': 0.5,
        'C_Lb': 2.0,
        'C_La': 200.0, ####
        ############# Starting Volumes
        'V0_Lt': 75.0,
        'V0_Lb': 75.0,
        'V0_La': 1800.0, ###
        ############# idxs of the thorax and ventilator
        'idxThx': 10,
        'idxVen': 13,
        'idxPlr': 11,
        ############# doubleSigmoid Parameters for big Alveoli
        'infPoint': 40.0,
        'maxCompliance': 0.90 * 300,
        'separation': 380.0,
        'k': 0.10,
        'cMin': 0.5,
        ############# doubleSigmoid Parameters for small Alveoli
        'infPoint_a': 20.0,
        'maxCompliance_a': 0.10 * 300,
        'separation_a': 50.0,
        'k_a': 0.3,
        'cMin_a': 0.5,
        ############# Parameters for sigmoid Variation between alveoli
        'infPoint_a_PerBigAlv':0.45,
        'infPoint_a_PerSmallAlv':0.35,
        'maxC_a_DC':2.0,
        'maxC_a_PerBigAlv':0.25,
        'maxC_a_PerSmallAlv':0.3,
        ############# sigmoid Parameters for the alveoli Resistors
        'minRes_a': 0.03,
        'maxRes_a': 0.03,
        'k_Res_a': -0.6,
        'dist_Res_a': 6.0,

        'minRes': 0.003,
        'maxRes': 0.003,
        'k_Res': -0.6,
        'dist_Res': 6.0,
    }
    
    if ventilator:
        modelStructure['ventilatorParams']['amplitude'] = 10.0 #8.8
    else:
        modelStructure['ventilatorParams']['amplitude'] = 0.0
    
    tree.create_connectivity_matrix(3,params,modelStructure)
    
    states, modelObjects, structures = modelGenerator.modelInit(modelStructure)

    states['V_As'] = 0.13 * totalBloodVolume
    states['V_Cs'] = 0.07 * totalBloodVolume
    states['V_Vs'] = 0.61 * totalBloodVolume
    states['V_Vt'] = 0.03 * totalBloodVolume
    states['V_Hr'] = 0.02 * totalBloodVolume
    states['V_Ap'] = 0.03 * totalBloodVolume
    states['V_Cp'] = 0.03 * totalBloodVolume
    states['V_Vp'] = 0.06 * totalBloodVolume
    states['V_Hl'] = 0.02 * totalBloodVolume

    states['C_As'] = 1.6560343757664413
    states['C_Cs'] = 20.784643925720665
    states['C_Vs'] = 307.0587399860084
    states['C_Vt'] = 17.96579280238321
    states['C_Ap'] = 3.176084428833837
    states['C_Cp'] = 10.147698416310314
    states['C_Vp'] = 22.899911461386782

    states['R_AsCs'] = 0.8655370106577612
    states['R_CsVs'] = 0.08
    states['R_VsVt'] = 0.04
    states['R_VtHr'] = 0.005
    states['R_HrAp'] = 0.006
    states['R_ApCp'] = 0.05113659340151016
    states['R_CpVp'] = 0.02
    states['R_VpHl'] = 0.005
    states['R_HlAs'] = 0.006

    states['E_C_Hr'] = 0.5756813703281292
    states['e_C_Hr'] = 1/ 15
    states['E_C_Hl'] = 2.8126995563827997
    states['e_C_Hl'] = 1/ 7.5

    return states, modelObjects, modelStructure, structures




#██████  ██       ██████  ████████ ███████ 
#██   ██ ██      ██    ██    ██    ██      
#██████  ██      ██    ██    ██    ███████ 
#██      ██      ██    ██    ██         ██ 
#██      ███████  ██████     ██    ███████ 

def axSigmoid(ax,results,pressure,volume,title,atmPressure,t, idx, structures):
    structure = structures['modelStructure']['controllers'][idx]
    maxValueToControl = float(structure[4])
    minValueToControl = float(structure[3])
    targetValue = float(structure[2])
    proportionalConstant = float(structure[5])
    V0 = np.arange(0,300,0.1)
    sigmoid = []
    for i in V0:
        amplitude = maxValueToControl - minValueToControl
        offset = minValueToControl
        inflectionPoint = targetValue
        slope = proportionalConstant

        sigmoid.append(amplitude / (1 + np.exp(-slope * (i - inflectionPoint))) + offset)

    ax.plot(V0,sigmoid, color='tab:red', label = 'Sigmoid')
    ax.plot(np.array(results[structure[0]]), np.array(results[structure[1]]), color='tab:blue', label = structure[1])

    ax.set_title('Sigmoid for the compliance: ' + structure[1])
    ax.set_ylabel('[mL/mmHg]', color='tab:red')
    ax.tick_params(axis='y', labelcolor='tab:red')
    ax.set_xlabel('Volume [mL]')
    ax.legend()

    var2 = 1/np.array(results[structure[1]])
    
    ax_11 = ax.twinx()
    ax_11.set_ylabel('[mL/mmHg]', color='tab:blue')
    ax_11.tick_params(axis='y', labelcolor='tab:blue')
    
    ax.set_ylim(min(np.array(sigmoid)), max(np.array(sigmoid)))
    ax_11.set_ylim(min(np.array(var2)), max(np.array(var2)))
    ax.set_yticks([
        min(np.array(sigmoid)), 
        max(np.array(sigmoid)), 
        np.round(np.mean(np.array(sigmoid)),4)
        ])
    ax_11.set_yticks([
        min(np.array(var2)), 
        max(np.array(var2)),
        np.round(np.mean(np.array(var2)),4)
        ])
    
    minP = min(min(np.array(sigmoid)), min(np.array(var2)))
    maxP = max(max(np.array(sigmoid)), max(np.array(var2)))
    
    ax.set_ylim([minP -1, maxP+1])
    ax_11.set_ylim([minP-1, maxP+1])

def plotSigmoid(results, totalTime, sampPeriod, atmPressure, modelObjects,idx,structures):
    t = np.arange(0, totalTime, sampPeriod)
    fig = mpl.figure(figsize=(5, 5))
    grid = gridspec.GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(grid[0, 0])
    axSigmoid(ax,results,'','','',atmPressure,t, idx, structures)
    ax.set_xlabel('Time [s]')
    mpl.tight_layout()
    mpl.show()


def plotDoubleSigmoidAxesTree(idx, structures, results, ax_1):
    atmPressure = 760.0
    structure = structures['modelStructure']['controllers'][idx]
    
    
    inflectionPoint = float(structure[2])
    maxCompliance = float(structure[3])
    dist = float(structure[4])
    slope = float(structure[5])
    cMin = float(structure[6])

    step = 1
    duration = float(inflectionPoint) + dist + 4
    V0 = np.arange(0,duration,step)
    sigmoid_close = []
    sigmoid_open = []
    pressure = []
    
    
    offset = (maxCompliance) + cMin
    amplitude = cMin - maxCompliance

    for i in V0:
        c_open = (amplitude) / (1 + np.exp(slope * (i - inflectionPoint))) + offset
        c_close = (amplitude) / (1 + np.exp(-slope * ((i) - (inflectionPoint+dist)))) + offset
        sigmoid_open.append(c_open)
        sigmoid_close.append(c_close)

    
    difference = np.abs(np.array(sigmoid_close) - np.array(sigmoid_open))
    intersection = np.where(difference == np.min(difference))[0][0]

    mergedSigmoids = sigmoid_open[0:int(intersection)] + sigmoid_close[int(intersection)-1:-1]

    '''
    for i in V0:
        c = amplitude / (1 + np.exp(-slope * (i - inflectionPoint-(dist)))) + offset
        c1 = (amplitude-cMin) / (1 + np.exp(slope1 * (i - inflectionPoint+dist))) + offset + cMin
        sigmoid.append(c)
        sigmoid1.append(c1)

    difference = np.abs(np.array(sigmoid) - np.array(sigmoid1))
    intersection = np.where(difference == np.min(difference))[0][0]

    mergedSigmoids = sigmoid1[0:int(intersection)] + sigmoid[int(intersection)-1:-1]
    '''

    for i in V0:
        pressure.append(i/mergedSigmoids[int(i)])

    p= -np.array(results['P' + structure[1][1:]] - atmPressure) + np.array(results['P_Plr'] - atmPressure)
    vol = np.array(results['V' + structure[1][1:]])
    com = np.array(results[structure[1]])
    
    if True:
        minVol = np.round(min(vol),0)
        ax_1.set_title(structure[1] + ' -> ' + minVol.astype(str), fontsize=8)
        ax_1.plot(V0, mergedSigmoids, color='tab:green', label = 'Merged Sigmoids', linewidth=1.5)
        ax_1.plot(vol, com, color='tab:brown', label = structure[1], linewidth=2.5)
        ax_11 = ax_1.twinx()

        lenV0 = int(inflectionPoint+dist+0.2*dist)
        ax_11.plot(V0[0:lenV0],pressure[0:lenV0], color='tab:grey', label = 'Pressure', linewidth=1.5)
        ax_11.plot(vol, -p, color='tab:blue', label = '-(P_Thorax-P_Alveoli)', linewidth=2.5)
        
        
        ax_1.tick_params(axis='y', labelcolor='tab:green', labelsize=8)
        ax_11.tick_params(axis='y', labelcolor='tab:grey', labelsize=8)

        ax_1.tick_params(axis='x', labelsize=7, labelrotation=90)
        ax_11.tick_params(axis='x', labelsize=7, labelrotation=90)
    
        infPoint = int(inflectionPoint)
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
            pressure[int(np.where(V0 == infPoint)[0])],
            pressure[int(np.where(V0 == infPoint1)[0])],
            max(pressure[0:int(infPoint)]),
            ])
        ax_1.set_xticks([
            0,
            #int(min(vol)), 
            intersection,
            infPoint,
            #int(max(vol)), 
            infPoint1,
            pressure.index(max(pressure[0:int(infPoint)])),
            ])
        ax_11.set_xticks([
        0,
        #int(min(vol)), 
        intersection,
        infPoint,
        #int(max(vol)), 
        infPoint1,
        pressure.index(max(pressure[0:int(infPoint)])),
        ])

def plotSigmoidsTree(results, totalTime, sampPeriod, atmPressure, modelObjects,idx,structures):
    
    controllers = structures['modelStructure']['controllers']

    unique_lengths = {}
    counter = 0
    for key,value in results.items():
        if key.startswith('C_La'):
            counter = counter + 1
            len_key = len(key)
            if len_key in unique_lengths:
                unique_lengths[len_key] += 1
            else:
                unique_lengths[len_key] = 1

    if counter == 10:
        row = 5
        col = 2
    elif counter == 40:
        row = 8
        col = 5

    fig = mpl.figure(figsize=(12, 16))
    grid = gridspec.GridSpec(row, col, figure=fig)
    
    counterRow = 0
    for i in range(len(controllers)):
        if controllers[i][1].startswith('C_La') and len(controllers[i][1]) == 8:
            name = controllers[i][1]
            ax = fig.add_subplot(grid[counterRow, 0])
            plotDoubleSigmoidAxesTree(i, structures, results, ax)
            
            counterCol = 1
            for j in range(len(controllers)):
                if name in controllers[j][1] and len(controllers[j][1]) != 8:
                    ax = fig.add_subplot(grid[counterRow, counterCol])
                    plotDoubleSigmoidAxesTree(j, structures, results, ax)
                    counterCol = counterCol + 1
            
            counterRow = counterRow + 1





    mpl.tight_layout()
    mpl.show()



def plotTreePEEP(results, totalTime, sampPeriod, atmPressure, modelObjects,idx,structures):
    toIgone = 0
    windowSize = 500 # Window size for the moving average
    t = np.arange(0, totalTime, sampPeriod)
    results = calculateVariablesForPlotting(results, totalTime, sampPeriod, atmPressure, modelObjects,idx,structures)

    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    fig = mpl.figure(figsize=(15, 12))
    grid = gridspec.GridSpec(8, 2, figure=fig)
    ##################################################################################################################################
    ax = fig.add_subplot(grid[0, :])
    ax.set_title('Big Alveoli Volumes [mL]')
    for key,value in results.items():
        if key.startswith('V_La') and len(key) == 8:
            ax.plot(np.array(t[toIgone:]),np.array(value[toIgone:]), label = key)
    ax.set_xlabel('Time [s]')
    
    ##################################################################################################################################
    ax = fig.add_subplot(grid[1, :])
    ax.set_title('Small Alveoli Volumes [mL]')
    for key,value in results.items():
        if key.startswith('V_La') and len(key) != 8:
            ax.plot(np.array(t[toIgone:]),np.array(value[toIgone:]), label = key)
    ax.set_xlabel('Time [s]')
            
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ax = fig.add_subplot(grid[2, 0])
    plots.buildLeftRightAxisIndependentScales(ax,results,'V_BloodThorax', 'V_BloodSystemic', 'Thoracic vs Systemic Blood Volumes [mL]','Thorax','Systemic','tab:blue','tab:red',0.0,t)
    ax.set_xlabel('Time [s]')

    ##################################################################################################################################
    ax = fig.add_subplot(grid[3, 0])
    plots.buildLeftRightAxisDependentScales(ax,results,'i_COl','i_COr','Stroke Volumes [mL]','Hl','Hr','tab:blue','tab:red',0.0,t)
    ax.set_xlabel('Time [s]')
    
    ##################################################################################################################################
    ax = fig.add_subplot(grid[4, 0])
    plots.buildLeftRightAxisIndependentScales(ax,results,'V_Hl', 'V_Hr', 'Ventricle Volumes [mL]','Left','Right','tab:blue','tab:red',0.0,t)
    ax.set_xlabel('Time [s]')
    
    ##################################################################################################################################
    ax = fig.add_subplot(grid[5, 0])
    plots.buildLeftRightAxisIndependentScales(ax,results,'P_Vp', 'V_Vp', 'Pulmonary Venous','Pressure','Volume','tab:blue','tab:red',atmPressure,t)
    ax.set_xlabel('Time [s]')
    
    ##################################################################################################################################
    ##################################################################################################################################
    ax = fig.add_subplot(grid[2, 1])
    plots.buildLeftRightAxisIndependentScales(ax,results,'P_Ven', 'uVol', ' Vantilator Pressure [mmHg] and Pleura Unstressed Volume [mL]','Ventilator','V0_Plr','tab:red','tab:blue',atmPressure,t)
    ax.set_xlabel('Time [s]')

    ##################################################################################################################################
    ax = fig.add_subplot(grid[3, 1])
    plots.buildLeftRightAxisIndependentScales(ax,results,'V_Alveoli', 'tidalVolume', 'Total Alveoli and Tidal Volumes [mL]','Alveoli','Tidal Volume','tab:green','tab:blue',0.0,t)
    ax.set_xlabel('Time [s]')

    ##################################################################################################################################
    ax = fig.add_subplot(grid[4, 1])
    plots.buildLeftRightAxisIndependentScales(ax,results,'P_Thx', 'V_Thx', 'Thoracic Pressure and Volume','Pressure [mmHg]','Volume [mL]','tab:red','tab:cyan',atmPressure,t)
    ax.set_xlabel('Time [s]')

    ##################################################################################################################################
    ax = fig.add_subplot(grid[5, 1])
    plots.buildLeftRightAxisIndependentScales(ax,results,'P_Plr', 'totalPleuraVolume', 'Pleural Pressure and Volume','Pressure [mmHg]','Volume [mL]','tab:blue','tab:red',atmPressure,t)
    ax.set_xlabel('Time [s]')

    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    
    maxArrayVol = np.zeros(int(totalTime/sampPeriod))
    minArrayVol = np.zeros(int(totalTime/sampPeriod))
    maxArrayPre = np.zeros(int(totalTime/sampPeriod))
    minArrayPre = np.zeros(int(totalTime/sampPeriod))

    minVolume = min(results['totalLungAirVolume'])
    for i in np.arange(0,len(results['totalLungAirVolume'] ),windowSize):
        breathVol = results['totalLungAirVolume'][i:i+windowSize]
        breathPress = results['P_Ven'][i:i+windowSize]

        ratio = breathVol*breathPress
        #ratio = breathPress/breathVol
        
        maxVal = max(ratio)
        minVal = min(ratio)
        maxVal_index = np.argmax(breathPress)
        minVal_index = np.argmin(breathPress[20:])+20
        
        maxArrayVol[i:i+windowSize] = maxArrayVol[i:i+windowSize] + breathVol[maxVal_index]
        minArrayVol[i:i+windowSize] = minArrayVol[i:i+windowSize] + breathVol[minVal_index]
        
        maxArrayPre[i:i+windowSize] = maxArrayPre[i:i+windowSize] + breathPress[maxVal_index]
        minArrayPre[i:i+windowSize] = minArrayPre[i:i+windowSize] + breathPress[minVal_index]

    ax = fig.add_subplot(grid[6:, :])
    ax.set_title('Volume [mL]/Pressure [mmHg] Curves')
    mifOfArray = int(len(results['P_Ven'])/2)
    ax.plot( results['P_Ven'][0:mifOfArray] - atmPressure, results['totalLungAirVolume'][0:mifOfArray] - minVolume, color='tab:green', label = 'Loops',linewidth=0.5)
    #ax.plot( results['P_Ven'][20000:] - atmPressure, totalLungAirVolume[20000:] - minVolume, color='tab:purple', label = key)
    
    ax.plot( maxArrayPre[0:int((totalTime/sampPeriod)/2)] - atmPressure, maxArrayVol[0:int((totalTime/sampPeriod)/2)] - minVolume, color='tab:blue', label = 'Inspiration',linewidth=3.5)
    ax.plot( minArrayPre[0:int((totalTime/sampPeriod)/2)] - atmPressure, minArrayVol[0:int((totalTime/sampPeriod)/2)] - minVolume, color='tab:red', label = 'Expiration',linewidth=3.5)

    #ax.plot( maxArrayPre[int((totalTime/sampPeriod)/2):] - atmPressure, maxArrayVol[int((totalTime/sampPeriod)/2):] - minVolume, color='tab:cyan', label = key)
    #ax.plot( minArrayPre[int((totalTime/sampPeriod)/2):] - atmPressure, minArrayVol[int((totalTime/sampPeriod)/2):] - minVolume, color='tab:pink', label = key)

    ax.set_xlabel('Pressure [mmHg]')
    ax.set_ylabel('Volume [mL]')
    ax.legend()
    
    

    mpl.tight_layout()
    mpl.show()

def plotTreeDeepBreath(results, totalTime, sampPeriod, atmPressure, modelObjects,idx,structures):
    toIgone = 00
    windowSize = 500
    t = np.arange(0, totalTime, sampPeriod)

    fig = mpl.figure(figsize=(15, 9))
    grid = gridspec.GridSpec(5 , 2, figure=fig)


    avgPressure = np.zeros(int(totalTime/sampPeriod))
    counter = 0
    for key,value in results.items():
        if key.startswith('P_La') :
            avgPressure = np.array(avgPressure) + np.array(value)
            counter = counter + 1
    avgPressure = avgPressure / counter
    

    volumeAlveoli = np.zeros(int(totalTime/sampPeriod))
    volumeBronchi = np.zeros(int(totalTime/sampPeriod))
    volumeBloodLungs = np.zeros(int(totalTime/sampPeriod))
    volumeBloodSystemic = np.zeros(int(totalTime/sampPeriod))
    volumeBloodHeart = np.zeros(int(totalTime/sampPeriod))
    volumeBloodThorax = np.zeros(int(totalTime/sampPeriod))

    for key,value in results.items():
        if key.startswith('V_La'):
            volumeAlveoli = volumeAlveoli + np.array(value)
        if key.startswith('V_Lb'):
            volumeBronchi = volumeBronchi + np.array(value)
        if ('V_Hl' in key) or ('V_Hr' in key):
            volumeBloodHeart = volumeBloodHeart + np.array(value)
        if ('V_Ap' in key) or ('V_Cp' in key) or ('V_Vp' in key):
            volumeBloodLungs = volumeBloodLungs + np.array(value)

        if ('V_As' in key) or ('V_Cs' in key) or ('V_Vs' in key):
            volumeBloodSystemic = volumeBloodSystemic + np.array(value)
        if ('V_Hl' in key) or ('V_Hr' in key) or ('V_Ap' in key) or ('V_Cp' in key) or ('V_Vp' in key) or ('V_Vt' in key):
            volumeBloodThorax = volumeBloodThorax + np.array(value)
        
    volumeThx = volumeBloodThorax + volumeAlveoli + volumeBronchi

    results['V_Alveoli'] = volumeAlveoli
    results['V_Bronchi'] = volumeBronchi
    results['V_Thx'] = volumeThx
    results['V_BloodLungs'] = volumeBloodLungs
    results['V_BloodSystemic'] = volumeBloodSystemic
    results['V_BloodHeart'] = volumeBloodHeart
    results['V_BloodThorax'] = volumeBloodThorax

    totalLungAirVolume = volumeAlveoli + volumeBronchi
    tidalVolume = np.zeros(int(totalTime/sampPeriod))
    
    maxVal = max(totalLungAirVolume[0:500])
    minVal = min(totalLungAirVolume[0:500])
    tidalVolume[0:500] = tidalVolume[0:500] + (maxVal - minVal)
    maxVal = max(totalLungAirVolume[500:1500])
    minVal = min(totalLungAirVolume[500:1500])
    tidalVolume[500:1500] = tidalVolume[500:1500] + (maxVal - minVal)
    maxVal = max(totalLungAirVolume[1500:])
    minVal = min(totalLungAirVolume[1500:])
    tidalVolume[1500:] = tidalVolume[1500:] + (maxVal - minVal)
    results['tidalVolume'] = tidalVolume

    results['uVol'] = plots.calculateThoraxUnstressedVolume(modelObjects,results)


    ##################################################################################################################################
    ax = fig.add_subplot(grid[0, :])
    ax.set_title('Big Alveoli Volumes [mL]')
    for key,value in results.items():
        if key.startswith('V_La') and len(key) == 8:
            ax.plot(np.array(t[toIgone:]),np.array(value[toIgone:]), label = key)
    ax.set_xlabel('Time [s]')
    
    ##################################################################################################################################
    ax = fig.add_subplot(grid[1, :])
    ax.set_title('Small Alveoli Volumes [mL]')
    for key,value in results.items():
        if key.startswith('V_La') and len(key) != 8:
            ax.plot(np.array(t[toIgone:]),np.array(value[toIgone:]), label = key)
    ax.set_xlabel('Time [s]')
            
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ax = fig.add_subplot(grid[2, 0])
    plots.buildLeftRightAxisIndependentScales(ax,results,'V_BloodThorax', 'V_BloodSystemic', 'Thoracic vs Systemic Blood Volumes [mL]','Thorax','Systemic','tab:blue','tab:red',0.0,t)
    ax.set_xlabel('Time [s]')

    ##################################################################################################################################
    ax = fig.add_subplot(grid[3, 0])
    plots.buildLeftRightAxisDependentScales(ax,results,'i_COl','i_COr','Stroke Volumes [mL]','Hl','Hr','tab:blue','tab:red',0.0,t)
    ax.set_xlabel('Time [s]')
    
    ##################################################################################################################################
    ax = fig.add_subplot(grid[4, 0])
    plots.buildLeftRightAxisIndependentScales(ax,results,'V_BloodHeart', 'V_BloodLungs', 'Heart vs Lung Blood Volumes [mL]','Heart','Lungs','tab:blue','tab:red',0.0,t)
    ax.set_xlabel('Time [s]')

    ##################################################################################################################################
    ##################################################################################################################################
    ax = fig.add_subplot(grid[2, 1])
    plots.buildLeftRightAxisIndependentScales(ax,results,'P_Ven', 'uVol', ' Vantilator Pressure [mmHg] and Thoracic Unstressed Volume [mL]','Ventilator','V_Thx0','tab:red','tab:blue',atmPressure,t)
    ax.set_xlabel('Time [s]')

    ##################################################################################################################################
    ax = fig.add_subplot(grid[3, 1])
    plots.buildLeftRightAxisIndependentScales(ax,results,'V_Alveoli', 'tidalVolume', 'Total Alveoli and Tidal Volumes [mL]','Alveoli','Tidal Volume','tab:green','tab:blue',0.0,t)
    ax.set_xlabel('Time [s]')

    ##################################################################################################################################
    ax = fig.add_subplot(grid[4, 1])
    plots.buildLeftRightAxisIndependentScales(ax,results,'P_Thx', 'V_Thx', 'Thoracic Pressure and Volume','Pressure [mmHg]','Volume [mL]','tab:blue','tab:cyan',atmPressure,t)
    ax.set_xlabel('Time [s]')

    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    
    
    

    mpl.tight_layout()
    mpl.show()

def plotLungVentComparison(results, totalTime, sampPeriod, atmPressure, modelObjects,idx,structures):
    toIgone = 500
    windowSize = toIgone
    t = np.arange(0, totalTime, sampPeriod)

    results = calculateVariablesForPlotting(results, totalTime, sampPeriod, atmPressure, modelObjects,idx,structures)
    
    fig = mpl.figure(figsize=(15, 15))
    grid = gridspec.GridSpec(11, 2, figure=fig)

    resSelf = {}
    resVen = {}
    for key,value in results.items():
        resSelf[key] = value[0:int(len(t)/2)]
        resVen[key] = value[int(len(t)/2):]

    t1 = t[0:int(len(t)/2)]  
    t11 = int(len(t)/2)   
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ax = fig.add_subplot(grid[0, 0])
    plots.buildLeftRightAxisIndependentScales(ax, resSelf,'P_Ven', 'uVol', ' Ventilator Pressure [mmHg] and Thoracic Unstressed Volume [mL]','Ventilator','V_Thx0','tab:red','tab:blue',atmPressure,t1)
    ax.set_xlabel('Time [s]')
    ax = fig.add_subplot(grid[0, 1])
    plots.buildLeftRightAxisIndependentScales(ax, resVen,'P_Ven', 'uVol', ' Ventilator Pressure [mmHg] and Thoracic Unstressed Volume [mL]','Ventilator','V_Thx0','tab:red','tab:blue',atmPressure,t1)
    ax.set_xlabel('Time [s]')
    
    ##################################################################################################################################
    minVolume = min(results['totalLungAirVolume'][0:t11])
    ax = fig.add_subplot(grid[1:3, 0])
    ax.set_title('Volume [mL]/Pressure [mmHg] Curves')
    ax.plot( results['P_Lt'][0:t11] - atmPressure, results['totalLungAirVolume'][0:t11] - minVolume, color='tab:pink', label = 'Trachea',linewidth=1.5)
    ax.plot( results['P_AvgLa'][0:t11] - atmPressure, results['totalLungAirVolume'][0:t11] - minVolume, color='tab:green', label = 'Alveoli',linewidth=1.5)
    ax.set_xlabel('Pressure [mmHg]')
    ax.set_ylabel('Volume [mL]')
    ax.legend()

    minVolume = min(results['totalLungAirVolume'][t11:])
    ax = fig.add_subplot(grid[1:3, 1])
    ax.set_title('Volume [mL]/Pressure [mmHg] Curves')
    ax.plot( results['P_Lt'][t11:] - atmPressure, results['totalLungAirVolume'][t11:] - minVolume, color='tab:pink', label = 'Trachea',linewidth=1.5)
    ax.plot( results['P_AvgLa'][t11:] - atmPressure, results['totalLungAirVolume'][t11:] - minVolume, color='tab:green', label = 'Alveoli',linewidth=1.5)
    ax.set_xlabel('Pressure [mmHg]')
    ax.set_ylabel('Volume [mL]')
    ax.legend()

    ##################################################################################################################################
    ax = fig.add_subplot(grid[3, 0])
    plots.buildLeftRightAxisIndependentScales(ax, resSelf,'V_Alveoli', 'tidalVolume', 'Total Alveoli and Tidal Volumes [mL]','Alveoli','Tidal Volume','tab:green','tab:blue',0.0,t1)
    ax.set_xlabel('Time [s]')

    ax = fig.add_subplot(grid[3, 1])
    plots.buildLeftRightAxisIndependentScales(ax, resVen,'V_Alveoli', 'tidalVolume', 'Total Alveoli and Tidal Volumes [mL]','Alveoli','Tidal Volume','tab:green','tab:blue',0.0,t1)
    ax.set_xlabel('Time [s]')

    ##################################################################################################################################
    ax = fig.add_subplot(grid[4, 0])
    plots.buildLeftRightAxisIndependentScales(ax, resSelf,'P_Thx', 'V_Thx', 'Thoracic Pressure and Volume','mmHg','mL','tab:blue','tab:red',atmPressure,t1)
    ax.set_xlabel('Time [s]')

    ax = fig.add_subplot(grid[4, 1])
    plots.buildLeftRightAxisIndependentScales(ax, resVen,'P_Thx', 'V_Thx', 'Thoracic Pressure and Volume','mmHg','mL','tab:blue','tab:red',atmPressure,t1)
    ax.set_xlabel('Time [s]')

    ax = fig.add_subplot(grid[5, 0])
    plots.buildLeftRightAxisIndependentScales(ax, resSelf,'P_Plr', 'totalPleuraVolume', 'Pleura Pressure and Volume','mmHg','mL','tab:blue','tab:red',atmPressure,t1)
    ax.set_xlabel('Time [s]')

    ax = fig.add_subplot(grid[5, 1])
    plots.buildLeftRightAxisIndependentScales(ax, resVen,'P_Plr', 'totalPleuraVolume', 'Pleura Pressure and Volume','mmHg','mL','tab:blue','tab:red',atmPressure,t1)
    ax.set_xlabel('Time [s]')

    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ax = fig.add_subplot(grid[6:8, 0])
    plots.heartPressureVolumeAxes(ax, resSelf,'Pressure-Volume Curves',atmPressure)

    ax = fig.add_subplot(grid[6:8, 1])
    plots.heartPressureVolumeAxes(ax, resVen,'Pressure-Volume Curves',atmPressure)

    ##################################################################################################################################
    ax = fig.add_subplot(grid[8, 0])
    plots.buildLeftRightAxisDependentScales(ax, resSelf,'i_COl','i_COr','Stroke Volumes [mL]','Hl','Hr','tab:blue','tab:red',0.0,t1)
    ax.set_xlabel('Time [s]')

    ax = fig.add_subplot(grid[8, 1])
    plots.buildLeftRightAxisDependentScales(ax, resVen,'i_COl','i_COr','Stroke Volumes [mL]','Hl','Hr','tab:blue','tab:red',0.0,t1)
    ax.set_xlabel('Time [s]')
    
    ##################################################################################################################################
    ax = fig.add_subplot(grid[9, 0])
    plots.buildPressureAndVolumeAxis(ax, resSelf,'P_As','V_As','Systemic Arteries (As)',atmPressure,t1)
    ax.set_xlabel('Time [s]')

    ax = fig.add_subplot(grid[9, 1])
    plots.buildPressureAndVolumeAxis(ax, resVen,'P_As','V_As','Systemic Arteries (As)',atmPressure,t1)
    ax.set_xlabel('Time [s]')

    ##################################################################################################################################
    ax = fig.add_subplot(grid[10, 0])
    plots.buildPressureAndVolumeAxis(ax, resSelf,'P_Vp','V_Vp','Pulmonary Veins (Vp)',atmPressure,t1)
    ax.set_xlabel('Time [s]')

    ax = fig.add_subplot(grid[10, 1])
    plots.buildPressureAndVolumeAxis(ax, resVen, 'P_Vp','V_Vp','Pulmonary Veins (Vp)',atmPressure,t1)
    ax.set_xlabel('Time [s]')
    
    ##################################################################################################################################
    

    ##################################################################################################################################
    ##################################################################################################################################
    mpl.tight_layout()
    mpl.show()

    perHeartbeat = generateHeartResultsTable(states,resSelf,simulationParams['atmPressure'])
    perHeartbeat = generateHeartResultsTable(states,resVen,simulationParams['atmPressure'])

def plotVentricleComplaince(results, totalTime, sampPeriod, atmPressure, modelObjects,idx,structures):
    toIgone = 500
    windowSize = toIgone
    t = np.arange(0, totalTime, sampPeriod)
    
    fig = mpl.figure(figsize=(15, 12))
    grid = gridspec.GridSpec(6, 2, figure=fig)

    resNoBias = {}
    resBias = {}
    for key,value in results.items():
        resNoBias[key] = value[0:int(len(t)/2)]
        resBias[key] = value[int(len(t)/2):]

    t1 = t[0:int(len(t)/2)]  
    t11 = int(len(t)/2)   
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ax = fig.add_subplot(grid[0:2, 0])
    plots.heartPressureVolumeAxes(ax, resNoBias,'Ventricles with constant diastolic elastance',atmPressure)
    
    ax = fig.add_subplot(grid[0:2, 1])
    plots.heartPressureVolumeAxes(ax, resBias,'Ventricles with variable diastolic elastance',atmPressure)
    
    ##################################################################################################################################
    ax = fig.add_subplot(grid[2, 0])
    plots.buildLeftRightAxisDependentScales(ax, resNoBias,'i_COl','i_COr','Stroke Volumes [mL]','Hl','Hr','tab:blue','tab:red',0.0,t1)
    ax.set_xlabel('Time [s]')

    ax = fig.add_subplot(grid[2, 1])
    plots.buildLeftRightAxisDependentScales(ax, resBias,'i_COl','i_COr','Stroke Volumes [mL]','Hl','Hr','tab:blue','tab:red',0.0,t1)
    ax.set_xlabel('Time [s]')

    ##################################################################################################################################
    
    resNoBias['e_Hl'] = calculateElastance(modelObjects,resNoBias,'P_Hl')
    resNoBias['e_Hr'] = calculateElastance(modelObjects,resNoBias,'P_Hr')
    ax = fig.add_subplot(grid[3, 0])
    plots.buildLeftRightAxisIndependentScales(ax,resNoBias,'e_Hl','e_Hr','Ventricle Compliances [mL/mmHg]','Hl','Hr','tab:blue','tab:red',0.0,t1)
    ax.set_xlabel('Time [s]')

    resBias['e_Hl'] = calculateElastance(modelObjects,resBias,'P_Hl')
    resBias['e_Hr'] = calculateElastance(modelObjects,resBias,'P_Hr')
    ax = fig.add_subplot(grid[3, 1])
    plots.buildLeftRightAxisIndependentScales(ax,resBias,'e_Hl','e_Hr','Ventricle Compliances [mL/mmHg]','Hl','Hr','tab:blue','tab:red',0.0,t1)
    ax.set_xlabel('Time [s]')

    ##################################################################################################################################
    ax = fig.add_subplot(grid[4, 0])
    plots.buildPressureAndVolumeAxis(ax, resNoBias,'P_As','V_As','Systemic Arteries (As)',atmPressure,t1)
    ax.set_xlabel('Time [s]')

    ax = fig.add_subplot(grid[4, 1])
    plots.buildPressureAndVolumeAxis(ax, resBias,'P_As','V_As','Systemic Arteries (As)',atmPressure,t1)
    ax.set_xlabel('Time [s]')

    ##################################################################################################################################
    ax = fig.add_subplot(grid[5, 0])
    plots.buildPressureAndVolumeAxis(ax, resNoBias,'P_Vp','V_Vp','Pulmonary Veins (Vp)',atmPressure,t1)
    ax.set_xlabel('Time [s]')

    ax = fig.add_subplot(grid[5, 1])
    plots.buildPressureAndVolumeAxis(ax, resBias, 'P_Vp','V_Vp','Pulmonary Veins (Vp)',atmPressure,t1)
    ax.set_xlabel('Time [s]')

    mpl.tight_layout()
    mpl.show()
    ##################################################################################################################################
    fig = mpl.figure(figsize=(12, 5))
    grid = gridspec.GridSpec(1, 2, figure=fig)
    
    ax = fig.add_subplot(grid[0, 0])
    axSigmoid(ax,resBias,'','','',atmPressure,t1, 0, structures)

    ax = fig.add_subplot(grid[0, 1])
    axSigmoid(ax,resBias,'','','',atmPressure,t1, 1, structures)

    ##################################################################################################################################
    ##################################################################################################################################
    mpl.tight_layout()
    mpl.show()

    perHeartbeat = generateHeartResultsTable(states,resNoBias,simulationParams['atmPressure'])
    perHeartbeat = generateHeartResultsTable(states,resBias,simulationParams['atmPressure'])

def plotPressureVolumeHeart(results, totalTime, sampPeriod, atmPressure,modelObjects):
    t = np.arange(0, totalTime, sampPeriod)
    fig = mpl.figure(figsize=(15, 10))
    grid = gridspec.GridSpec(7, 2, figure=fig)
    
    ################################################################################################################################
    ax_1 = fig.add_subplot(grid[0:3, 0])
    plots.heartPressureVolumeAxes(ax_1,results,'Pressure-Volume Curves',atmPressure)

    ################################################################################################################################
    ax_2 = fig.add_subplot(grid[3, 0])
    plots.buildLeftRightAxisDependentScales(ax_2,results,'i_COl','i_COr','Stroke Volumes [mL]','Hl','Hr','tab:blue','tab:red',0.0,t)
    ax_2.set_xlabel('Time [s]')

    ax_3 = fig.add_subplot(grid[5, 0])
    plots.buildPressureAndVolumeAxis(ax_3,results,'P_Hl','V_Hl','Left Ventricle (Hl)',atmPressure,t)
    #buildLeftRightAxisDependentPressureScales(ax_3,results,'P_Hl','P_As','Pressure Left Ventricle (Hl) & Systemic Artery (As)  [mmHg]','Hl','As','tab:blue','tab:green',atmPressure,t)
    ax_3.set_xlabel('Time [s]')

    ax_4 = fig.add_subplot(grid[6, 0])
    plots.buildPressureAndVolumeAxis(ax_4,results,'P_Hr','V_Hr','Right Ventricle (Hl)',atmPressure,t)
    #buildLeftRightAxisDependentPressureScales(ax_4,results,'P_Hr','P_Ap','Pressure Left Ventricle (Hl) & Systemic Artery (As)  [mmHg]','Hr','Ap','tab:blue','tab:purple',atmPressure,t)
    ax_4.set_xlabel('Time [s]')

    #ax_5 = fig.add_subplot(grid[6, 0])
    #buildLeftRightAxisDependentScales(ax_5,results,'V_Hl','V_Hr','Volumes [mL]','Hl','Hr','tab:blue','tab:red',0.0,t)
    #ax_5.set_xlabel('Time [s]')

    ax_6 = fig.add_subplot(grid[4, 0])
    results['e_Hl'] = plots.calculateElastance(modelObjects,results,'P_Hl')
    results['e_Hr'] = plots.calculateElastance(modelObjects,results,'P_Hr')
    plots.buildLeftRightAxisIndependentScales(ax_6,results,'e_Hl','e_Hr','Ventricle Compliances','Hl','Hr','tab:blue','tab:red',0.0,t)
    ax_6.set_xlabel('Time [s]')
    ################################################################################################################################

    ax_7 = fig.add_subplot(grid[0, 1])
    plots.buildPressureAndVolumeAxis(ax_7,results,'P_Ap','V_Ap','Pulmonary Arteries (Ap)',atmPressure,t)

    ax_8 = fig.add_subplot(grid[1, 1])
    plots.buildPressureAndVolumeAxis(ax_8,results,'P_Cp','V_Cp','Pulmonary Capillaries (Cp)',atmPressure,t)

    ax_9 = fig.add_subplot(grid[2, 1])
    plots.buildPressureAndVolumeAxis(ax_9,results,'P_Vp','V_Vp','Pulmonary Veins (Vp)',atmPressure,t)

    ax_10 = fig.add_subplot(grid[3, 1])
    plots.buildPressureAndVolumeAxis(ax_10,results,'P_As','V_As','Systemic Arteries (As)',atmPressure,t)

    ax_11 = fig.add_subplot(grid[4, 1])
    plots.buildPressureAndVolumeAxis(ax_11,results,'P_Cs','V_Cs','Systemic Capillaries (Cs)',atmPressure,t)

    ax_12 = fig.add_subplot(grid[5, 1])
    plots.buildPressureAndVolumeAxis(ax_12,results,'P_Vs','V_Vs','Systemic Veins (Vs)',atmPressure,t)

    ax_13 = fig.add_subplot(grid[6, 1])
    plots.buildPressureAndVolumeAxis(ax_13,results,'P_Vt','V_Vt','Thoracic Veins (Vt)',atmPressure,t)

    mpl.tight_layout()
    mpl.show()

def plotLungResults(results, totalTime, sampPeriod, atmPressure, modelObjects,idx,structures):
    toIgone = 0
    windowSize = 500 # Window size for the moving average
    t = np.arange(0, totalTime, sampPeriod)
    
    ##### AVERAGE PRESSURE #####
    avgPressure = np.zeros(int(totalTime/sampPeriod))
    counter = 0
    for key,value in results.items():
        if key.startswith('P_La') :
            avgPressure = np.array(avgPressure) + np.array(value)
            counter = counter + 1
    avgPressure = avgPressure / counter
    

    ##### Init arrays with final dimensions #####
    volumeAlveoli = np.zeros(int(totalTime/sampPeriod))
    volumeBronchi = np.zeros(int(totalTime/sampPeriod))
    volumeBloodLungs = np.zeros(int(totalTime/sampPeriod))
    volumeBloodSystemic = np.zeros(int(totalTime/sampPeriod))
    volumeBloodHeart = np.zeros(int(totalTime/sampPeriod))
    volumeBloodThorax = np.zeros(int(totalTime/sampPeriod))
    volumePleura = np.zeros(int(totalTime/sampPeriod))

    ##### Populate the arrays with data #####
    for key,value in results.items():
        if key.startswith('V_La'):
            volumeAlveoli = volumeAlveoli + np.array(value)
        if key.startswith('V_Lb'):
            volumeBronchi = volumeBronchi + np.array(value)
        if ('V_Hl' in key) or ('V_Hr' in key):
            volumeBloodHeart = volumeBloodHeart + np.array(value)
        if ('V_Ap' in key) or ('V_Cp' in key) or ('V_Vp' in key):
            volumeBloodLungs = volumeBloodLungs + np.array(value)

        if ('V_As' in key) or ('V_Cs' in key) or ('V_Vs' in key):
            volumeBloodSystemic = volumeBloodSystemic + np.array(value)
        if ('V_Hl' in key) or ('V_Hr' in key) or ('V_Ap' in key) or ('V_Cp' in key) or ('V_Vp' in key) or ('V_Vt' in key):
            volumeBloodThorax = volumeBloodThorax + np.array(value)
        if ('V_Cp' in key) or ('V_Vp' in key):
            volumePleura = volumePleura + np.array(value)
        
    volumeThx = volumeBloodThorax + volumeAlveoli + volumeBronchi
    volumePleura = volumePleura + volumeAlveoli + volumeBronchi
    totalLungAirVolume = volumeAlveoli + volumeBronchi
    
    ##### Calculate the moving average and replace in results array #####
    #volumeBloodThorax[windowSize-1:] = np.convolve(volumeBloodThorax, np.ones(windowSize) / windowSize, mode='valid') # Moving average
    #volumeBloodLungs[windowSize-1:] = np.convolve(volumeBloodLungs, np.ones(windowSize) / windowSize, mode='valid')
    #volumeBloodSystemic[windowSize-1:] = np.convolve(volumeBloodSystemic, np.ones(windowSize) / windowSize, mode='valid')
    #volumeBloodHeart[windowSize-1:] = np.convolve(volumeBloodHeart, np.ones(windowSize) / windowSize, mode='valid')
    #results['i_COl'][windowSize-1:] = np.convolve(results['i_COl'], np.ones(windowSize) / windowSize, mode='valid')
    #results['i_COr'][windowSize-1:] = np.convolve(results['i_COr'], np.ones(windowSize) / windowSize, mode='valid')

    results['V_Alveoli'] = volumeAlveoli
    results['V_Bronchi'] = volumeBronchi
    results['V_Thx'] = volumeThx
    results['V_Plr'] = volumePleura
    results['V_BloodLungs'] = volumeBloodLungs
    results['V_BloodSystemic'] = volumeBloodSystemic
    results['V_BloodHeart'] = volumeBloodHeart
    results['V_BloodThorax'] = volumeBloodThorax

    ##### Calculate the tidal volume #####
    tidalVolume = np.zeros(int(totalTime/sampPeriod))
    for i in np.arange(0,len(totalLungAirVolume ),windowSize):
        maxVal = max(totalLungAirVolume[i:i+windowSize])
        minVal = min(totalLungAirVolume[i:i+windowSize])
        tidalVolume[i:i+windowSize] = tidalVolume[i:i+windowSize] + (maxVal - minVal)
    results['tidalVolume'] = tidalVolume

    results['uVol'] = plots.calculateThoraxUnstressedVolume(modelObjects,results)

    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    fig = mpl.figure(figsize=(15, 12))
    grid = gridspec.GridSpec(8, 2, figure=fig)
    ##################################################################################################################################
    ax = fig.add_subplot(grid[0, :])
    ax.set_title('Big Alveoli Volumes [mL]')
    for key,value in results.items():
        if key.startswith('V_La') and len(key) == 8:
            ax.plot(np.array(t[toIgone:]),np.array(value[toIgone:]), label = key)
    ax.set_xlabel('Time [s]')
    
    ##################################################################################################################################
    ax = fig.add_subplot(grid[1, :])
    ax.set_title('Small Alveoli Volumes [mL]')
    for key,value in results.items():
        if key.startswith('V_La') and len(key) != 8:
            ax.plot(np.array(t[toIgone:]),np.array(value[toIgone:]), label = key)
    ax.set_xlabel('Time [s]')
            
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ax = fig.add_subplot(grid[2, 0])
    plots.buildLeftRightAxisIndependentScales(ax,results,'V_BloodThorax', 'V_BloodSystemic', 'Thoracic vs Systemic Blood Volumes [mL]','Thorax','Systemic','tab:blue','tab:red',0.0,t)
    ax.set_xlabel('Time [s]')

    ##################################################################################################################################
    ax = fig.add_subplot(grid[3, 0])
    plots.buildLeftRightAxisDependentScales(ax,results,'i_COl','i_COr','Stroke Volumes [mL]','Hl','Hr','tab:blue','tab:red',0.0,t)
    ax.set_xlabel('Time [s]')
    
    ##################################################################################################################################
    ax = fig.add_subplot(grid[4, 0])
    plots.buildLeftRightAxisIndependentScales(ax,results,'V_BloodHeart', 'V_BloodLungs', 'Heart vs Lung Blood Volumes [mL]','Heart','Lungs','tab:blue','tab:red',0.0,t)
    ax.set_xlabel('Time [s]')
    
    ##################################################################################################################################
    ax = fig.add_subplot(grid[5, 0])
    plots.buildLeftRightAxisIndependentScales(ax,results,'V_Thx', 'V_Plr', 'Thoracic Vs Pleura Volume','Thorax','Pleura','tab:blue','tab:red',atmPressure,t)
    ax.set_xlabel('Time [s]')
    
    ##################################################################################################################################
    ##################################################################################################################################
    ax = fig.add_subplot(grid[2, 1])
    plots.buildLeftRightAxisIndependentScales(ax,results,'P_Ven', 'uVol', ' Vantilator Pressure [mmHg] and Pleura Unstressed Volume [mL]','Ventilator','V0_Plr','tab:red','tab:blue',atmPressure,t)
    ax.set_xlabel('Time [s]')

    ##################################################################################################################################
    ax = fig.add_subplot(grid[3, 1])
    plots.buildLeftRightAxisIndependentScales(ax,results,'V_Alveoli', 'tidalVolume', 'Total Alveoli and Tidal Volumes [mL]','Alveoli','Tidal Volume','tab:green','tab:blue',0.0,t)
    ax.set_xlabel('Time [s]')

    ##################################################################################################################################
    ax = fig.add_subplot(grid[4, 1])
    plots.buildLeftRightAxisIndependentScales(ax,results,'P_Thx', 'V_Thx', 'Thoracic Pressure and Volume','Pressure [mmHg]','Volume [mL]','tab:blue','tab:cyan',atmPressure,t)
    ax.set_xlabel('Time [s]')

    ##################################################################################################################################
    ax = fig.add_subplot(grid[5, 1])
    plots.buildLeftRightAxisIndependentScales(ax,results,'P_Plr', 'V_Plr', 'Pleural Pressure and Volume','Pressure [mmHg]','Volume [mL]','tab:blue','tab:red',atmPressure,t)
    ax.set_xlabel('Time [s]')

    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    
    maxArrayVol = np.zeros(int(totalTime/sampPeriod))
    minArrayVol = np.zeros(int(totalTime/sampPeriod))
    maxArrayPre = np.zeros(int(totalTime/sampPeriod))
    minArrayPre = np.zeros(int(totalTime/sampPeriod))

    minVolume = min(totalLungAirVolume)
    for i in np.arange(0,len(totalLungAirVolume ),windowSize):
        breathVol = totalLungAirVolume[i:i+windowSize]
        breathPress = results['P_Ven'][i:i+windowSize]

        ratio = breathVol*breathPress
        #ratio = breathPress/breathVol
        
        maxVal = max(ratio)
        minVal = min(ratio)
        maxVal_index = np.argmax(breathPress)
        minVal_index = np.argmin(breathPress[20:])+20
        
        maxArrayVol[i:i+windowSize] = maxArrayVol[i:i+windowSize] + breathVol[maxVal_index]
        minArrayVol[i:i+windowSize] = minArrayVol[i:i+windowSize] + breathVol[minVal_index]
        
        maxArrayPre[i:i+windowSize] = maxArrayPre[i:i+windowSize] + breathPress[maxVal_index]
        minArrayPre[i:i+windowSize] = minArrayPre[i:i+windowSize] + breathPress[minVal_index]

    results['tidalVolume'] = tidalVolume

    
    ax = fig.add_subplot(grid[6:, :])
    ax.set_title('Volume [mL]/Pressure [mmHg] Curves')
    ax.plot( results['P_Ven'] - atmPressure, totalLungAirVolume - minVolume, color='tab:green', label = 'Loops',linewidth=0.5)
    ax.plot( maxArrayPre[0:int((totalTime/sampPeriod)/2)] - atmPressure, maxArrayVol[0:int((totalTime/sampPeriod)/2)] - minVolume, color='tab:blue', label = 'Inspiration',linewidth=3.5)
    ax.plot( minArrayPre[0:int((totalTime/sampPeriod)/2)] - atmPressure, minArrayVol[0:int((totalTime/sampPeriod)/2)] - minVolume, color='tab:red', label = 'Expiration',linewidth=3.5)
    
    ax.plot( avgPressure - atmPressure, totalLungAirVolume - minVolume, color='tab:purple', label = key)

    ax.set_xlabel('Pressure [mmHg]')
    ax.set_ylabel('Volume [mL]')
    ax.legend()
    
    

    mpl.tight_layout()
    mpl.show()



#████████ ███████ ███████ ████████ ███████ 
#   ██    ██      ██         ██    ██      
#   ██    █████   ███████    ██    ███████ 
#   ██    ██           ██    ██         ██ 
#   ██    ███████ ███████    ██    ███████ 

# Support function to run the simulation
def runSimulation(states, cpModel, runTime, totalRunsToIgnore, totalRuns, pressureRes, runsRes, structures):
    trun = time.time()
    # Create the model object with the parameters and the equations to be solved by JAX
    nanValues = False
    for value in states.values():
        if isinstance(value, (float, int)) and math.isnan(value):
            nanValues = True
            print('NAN Values Present')
            break
            

    if not nanValues:
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

#########################################################################################
#########################################################################################

# Type 0
def runCalibration(simulationParams):
    with jax.default_device(jax.devices("cpu")[0]):
        jax.config.update("jax_enable_x64", True)

        totalTime = simulationParams['runTime']*(simulationParams['base_runsToSave'])
        simulationParams['totalTime'] = totalTime

        states, modelObjects, modelStructure, structures = initModelWithTree(ventilator=simulationParams['ventilator'], control = simulationParams['control'], calibration = simulationParams['calibration'])

        if simulationParams['ventilator']:
            states['E_C_Thx'] = 3000.0 # Paralyse the thorax
            states['e_C_Thx'] = 3000.0 # Paralyse the thorax
        
        # Run 1
        cpModel = models.CardioPulmonaryModel(states, modelObjects)

        if simulationParams['ventilator']:
            states['PEEP'] = simulationParams['PEEP']
        
        pressureRes, runsRes, trun, states = runSimulation(
                states, 
                cpModel, 
                runTime=simulationParams['runTime'], 
                totalRunsToIgnore=simulationParams['base_runsToIgnore'], 
                totalRuns=simulationParams['base_runsToSave'], 
                pressureRes={}, 
                runsRes={}, 
                structures=structures
            )
        results = pressureRes | runsRes

    return states, modelObjects, modelStructure, results, structures

# Type 1
def runBaselineSimulation(simulationParams):
    with jax.default_device(jax.devices("cpu")[0]):
        jax.config.update("jax_enable_x64", True)

        totalTime = simulationParams['runTime']*(simulationParams['base_runsToSave'])
        simulationParams['totalTime'] = totalTime

        states, modelObjects, modelStructure, structures = initModelWithTree(ventilator=simulationParams['ventilator'], control = simulationParams['control'], calibration = simulationParams['calibration'])

        if simulationParams['ventilator']:
            states['E_C_Thx'] = 3000.0 # Paralyse the thorax
            states['e_C_Thx'] = 3000.0 # Paralyse the thorax
        
        # Run 1
        cpModel = models.CardioPulmonaryModel(states, modelObjects)

        if simulationParams['ventilator']:
            states['PEEP'] = simulationParams['PEEP']
        
        pressureRes, runsRes, trun, states = runSimulation(
                states, 
                cpModel, 
                runTime=simulationParams['runTime'], 
                totalRunsToIgnore=simulationParams['base_runsToIgnore'], 
                totalRuns=simulationParams['base_runsToSave'], 
                pressureRes={}, 
                runsRes={}, 
                structures=structures
            )
        results = pressureRes | runsRes

    return states, modelObjects, modelStructure, results, structures

# Type 2
def runTestDeepBreath(simulationParams):
    with jax.default_device(jax.devices("cpu")[0]):
        jax.config.update("jax_enable_x64", True)

        totalTime = simulationParams['runTime']*(simulationParams['base_runsToSave'])
        totalTime1 = simulationParams['runTime']*(simulationParams['test_runsToSave'] * simulationParams['test_nrRuns'])

        totalTime = totalTime + totalTime1
        simulationParams['totalTime'] = totalTime
        
        states, modelObjects, modelStructure, structures = initModelWithTree(ventilator=False, control = simulationParams['control'], calibration = simulationParams['calibration'])
        
        states['E_C_Plr'] = 4500.0 # Paralyse the thorax
        states['e_C_Plr'] = 3800.0 # Paralyse the thorax

        ###################################################################################################################
        # Run 1
        cpModel = models.CardioPulmonaryModel(states, modelObjects)
        
        pressureRes, runsRes, trun, states = runSimulation(
                states, 
                cpModel, 
                runTime=simulationParams['runTime'], 
                totalRunsToIgnore=simulationParams['base_runsToIgnore'], 
                totalRuns=simulationParams['base_runsToSave'], 
                pressureRes={}, 
                runsRes={}, 
                structures=structures
            )
        results = pressureRes | runsRes
        
        ###################################################################################################################
        # Run 2
        for i in range(0,simulationParams['test_nrRuns']):
            if i < simulationParams['Cond1']:
                states['E_C_Plr'] = 7500.0
                #states['e_C_Plr'] = 3800.0
                states['RC'] = 10.0
            elif i < simulationParams['Cond2']:
                pass
            else:
                states['RC'] = 5.0
                states['E_C_Plr'] = 4500.0
                #states['e_C_Plr'] = 3800.0
                
            
            pressureRes, runsRes, trun, states = runSimulation(
                    states, 
                    cpModel, 
                    runTime=simulationParams['runTime'], 
                    totalRunsToIgnore=simulationParams['test_runsToIgnore'], 
                    totalRuns=simulationParams['test_runsToSave'], 
                    pressureRes=pressureRes, 
                    runsRes=runsRes, 
                    structures=structures
                )
            results = pressureRes | runsRes



    return states, modelObjects, modelStructure, results, structures

# Type 3
def runTestPEEPChallenge(simulationParams):
    with jax.default_device(jax.devices("cpu")[0]):
        jax.config.update("jax_enable_x64", True)

        totalTime = simulationParams['runTime']*(simulationParams['base_runsToSave'])
        totalTime1 = simulationParams['runTime']*(simulationParams['test_runsToSave'] * simulationParams['test_nrRuns'])

        totalTime = totalTime + totalTime1
        simulationParams['totalTime'] = totalTime

        states, modelObjects, modelStructure, structures = initModelWithTree(ventilator=True, control = simulationParams['control'], calibration = simulationParams['calibration'])
        states['E_C_Plr'] = 3500.0 # Paralyse the thorax
        states['e_C_Plr'] = 3500.0 # Paralyse the thorax
        
        
        
        # Run 1
        cpModel = models.CardioPulmonaryModel(states, modelObjects)
        
        pressureRes, runsRes, trun, states = runSimulation(
                states, 
                cpModel, 
                runTime=simulationParams['runTime'], 
                totalRunsToIgnore=simulationParams['base_runsToIgnore'], 
                totalRuns=simulationParams['base_runsToSave'], 
                pressureRes={}, 
                runsRes={}, 
                structures=structures
            )
        results = pressureRes | runsRes

        # Run 2
        for i in range(0,simulationParams['test_nrRuns']):
            if i < simulationParams['Cond1']:
                states['PEEP'] = states['PEEP'] + 1.0 # Paralyse the thorax
            elif i < simulationParams['Cond2']:
                states['PEEP'] = states['PEEP'] - 1.0 # Paralyse the thorax
                
            
            pressureRes, runsRes, trun, states = runSimulation(
                    states, 
                    cpModel, 
                    runTime=simulationParams['runTime'], 
                    totalRunsToIgnore=simulationParams['test_runsToIgnore'], 
                    totalRuns=simulationParams['test_runsToSave'], 
                    pressureRes=pressureRes, 
                    runsRes=runsRes, 
                    structures=structures
                )
            results = pressureRes | runsRes



    return states, modelObjects, modelStructure, results, structures

# Type 4
def runSelfAndVent(simulationParams):
    with jax.default_device(jax.devices("cpu")[0]):
        jax.config.update("jax_enable_x64", True)

        totalTime = simulationParams['runTime']*(simulationParams['base_runsToSave'])
        totalTime1 = simulationParams['runTime']*simulationParams['test_runsToSave'] 

        totalTime = totalTime + totalTime1
        simulationParams['totalTime'] = totalTime

        states, modelObjects, modelStructure, structures = initModelWithTree(ventilator=False, control = simulationParams['control'], calibration = simulationParams['calibration'])
        
        # Run 1
        cpModel = models.CardioPulmonaryModel(states, modelObjects)
        
        pressureRes, runsRes, trun, states = runSimulation(
                states, 
                cpModel, 
                runTime=simulationParams['runTime'], 
                totalRunsToIgnore=simulationParams['base_runsToIgnore'], 
                totalRuns=simulationParams['base_runsToSave'], 
                pressureRes={}, 
                runsRes={}, 
                structures=structures
            )
        results = pressureRes | runsRes
        

        ignoreStates, modelObjects, modelStructure, structures = initModelWithTree(ventilator=True, control = simulationParams['control'], calibration = simulationParams['calibration'])
        cpModel = models.CardioPulmonaryModel(states, modelObjects)
        
        states['E_C_Plr'] = 3600.0 # Paralyse the thorax
        states['e_C_Plr'] = 3600.0 # Paralyse the thorax
        states['PEEP'] = states['PEEP'] + 5.0

        pressureRes, runsRes, trun, states = runSimulation(
                states, 
                cpModel, 
                runTime=simulationParams['runTime'], 
                totalRunsToIgnore=simulationParams['test_runsToIgnore'], 
                totalRuns=simulationParams['test_runsToSave'], 
                pressureRes=pressureRes, 
                runsRes=runsRes, 
                structures=structures
            )
        results = pressureRes | runsRes



    return states, modelObjects, modelStructure, results, structures

# Type 5
def runVentricleCompliance(simulationParams):
    with jax.default_device(jax.devices("cpu")[0]):
        jax.config.update("jax_enable_x64", True)

        totalTime = simulationParams['runTime']*(simulationParams['base_runsToSave'])
        totalTime1 = simulationParams['runTime']*simulationParams['test_runsToSave'] 

        totalTime = totalTime + totalTime1
        simulationParams['totalTime'] = totalTime

        
        states, modelObjects, modelStructure, structures = initModelWithTree(ventilator=simulationParams['ventilator'], control = False, calibration = False)
        # Run 1
        cpModel = models.CardioPulmonaryModel(states, modelObjects)
        
        pressureRes, runsRes, trun, states = runSimulation(
                states, 
                cpModel, 
                runTime=simulationParams['runTime'], 
                totalRunsToIgnore=simulationParams['base_runsToIgnore'], 
                totalRuns=simulationParams['base_runsToSave'], 
                pressureRes={}, 
                runsRes={}, 
                structures=structures
            )
        results = pressureRes | runsRes
        
        # Run 2 
        states, modelObjects, modelStructure, structures = initModelWithTree(ventilator=simulationParams['ventilator'], control = True, calibration = False)
       
        cpModel = models.CardioPulmonaryModel(states, modelObjects)
        pressureRes, runsRes, trun, states = runSimulation(
                states, 
                cpModel, 
                runTime=simulationParams['runTime'], 
                totalRunsToIgnore=simulationParams['test_runsToIgnore'], 
                totalRuns=simulationParams['test_runsToSave'], 
                pressureRes=pressureRes, 
                runsRes=runsRes, 
                structures=structures
            )
        results = pressureRes | runsRes


    return states, modelObjects, modelStructure, results, structures

# Type 6
def runTestPleuraThoraxCompliance(simulationParams):
    with jax.default_device(jax.devices("cpu")[0]):
        jax.config.update("jax_enable_x64", True)

        totalTime = simulationParams['runTime']*(simulationParams['base_runsToSave'])
        totalTime1 = simulationParams['runTime']*(simulationParams['test_runsToSave'] * simulationParams['test_nrRuns'])

        totalTime = totalTime + totalTime1
        simulationParams['totalTime'] = totalTime

        states, modelObjects, modelStructure, structures = initModelWithTree(ventilator=True, control = simulationParams['control'], calibration = simulationParams['calibration'])
        states['E_C_Plr'] = 3400.0 # Paralyse the thorax
        states['e_C_Plr'] = 3400.0 # Paralyse the thorax
        
        
        
        # Run 1
        cpModel = models.CardioPulmonaryModel(states, modelObjects)
        
        pressureRes, runsRes, trun, states = runSimulation(
                states, 
                cpModel, 
                runTime=simulationParams['runTime'], 
                totalRunsToIgnore=simulationParams['base_runsToIgnore'], 
                totalRuns=simulationParams['base_runsToSave'], 
                pressureRes={}, 
                runsRes={}, 
                structures=structures
            )
        results = pressureRes | runsRes

        # Run 2
        for i in range(0,simulationParams['test_nrRuns']):
            if i < simulationParams['Cond1']:
                states['PEEP'] = states['PEEP'] + 1.0 # Paralyse the thorax
            elif i < simulationParams['Cond2']:
                states['PEEP'] = states['PEEP'] - 1.0 # Paralyse the thorax
                
            
            pressureRes, runsRes, trun, states = runSimulation(
                    states, 
                    cpModel, 
                    runTime=simulationParams['runTime'], 
                    totalRunsToIgnore=simulationParams['test_runsToIgnore'], 
                    totalRuns=simulationParams['test_runsToSave'], 
                    pressureRes=pressureRes, 
                    runsRes=runsRes, 
                    structures=structures
                )
            results = pressureRes | runsRes



    return states, modelObjects, modelStructure, results, structures


'''
███    ███  █████  ██ ███    ██ 
████  ████ ██   ██ ██ ████   ██ 
██ ████ ██ ███████ ██ ██ ██  ██
██  ██  ██ ██   ██ ██ ██  ██ ██ 
██      ██ ██   ██ ██ ██   ████
'''

t0 = time.time()


test = [1,2,3,4]

# Calibration Test
if 0 in test:
    simulationParams = {
        'runTime': 5,
        'base_runsToIgnore': 2000,
        'base_runsToSave': 1,
        'sampPeriod': 0.01,
        'atmPressure': 760.0,
        'idx': 0,
        'ventilator': False,
        'control': True,
        'calibration': True,
        'PEEP': 760 + 5.0,
    }
    states, modelObjects, modelStructure, results, structures = runBaselineSimulation(simulationParams)
    
    #plotSigmoidsTree(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'], modelObjects,simulationParams['idx'],structures)
    #plotLungResults(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'],modelObjects,simulationParams['idx'],structures)
    downsampled_result = downsample_dictionary(results, 1/simulationParams['sampPeriod'], 1)

    #plots.plotPressureVolumeHeart(downsampled_result, simulationParams['totalTime'], 1, simulationParams['atmPressure'],modelObjects)
    #plots.plotControllers(downsampled_result, simulationParams['totalTime'], 1, simulationParams['atmPressure'])

    plots.plotPressureVolumeHeart(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'],modelObjects)
    plots.plotControllers(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'])

    totalBloodVolume = 5000.0
    resultsTableVol, resultsTablePre = generateCardioResultsTable(states,results,totalBloodVolume,simulationParams['atmPressure'])
    
# Baseline Simulation
if 1 in test:  
    simulationParams = {
        'runTime': 5,
        'base_runsToIgnore': 2,
        'base_runsToSave': 2,
        'sampPeriod': 0.01,
        'atmPressure': 760.0,
        'idx': 0,
        'ventilator': False,
        'PEEP': 760 + 5.0,
        'control': True,
        'calibration': False,
    }
    states, modelObjects, modelStructure, results, structures = runBaselineSimulation(simulationParams)
    #plotSigmoidsTree(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'], modelObjects,simulationParams['idx'],structures)
    plotSigmoid(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'],modelObjects,2,structures)
    plots.plotPressureVolumeHeart(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'],modelObjects)
    plotLungResults(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'],modelObjects,simulationParams['idx'],structures)
    #plots.plotControllers(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'])

##Deep Breath Test 
if 2 in test:
    simulationParams = {
        'runTime': 5.0,
        'base_runsToIgnore': 10,
        'base_runsToSave': 1,
        'test_runsToIgnore': 0,
        'test_runsToSave': 1,
        'test_nrRuns': 3,
        'sampPeriod': 0.01,
        'atmPressure': 760.0,
        'Cond1': 2,
        'Cond2': 0,
        'idx': 0,
        'totalTime': 0,
        'control': True,
        'calibration': False,
    }
    states, modelObjects, modelStructure, results, structures = runTestDeepBreath(simulationParams)
    #plotSigmoidsTree(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'], modelObjects,simulationParams['idx'],structures)
    plotTreeDeepBreath(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'],modelObjects,simulationParams['idx'],structures)
    #plots.plotPressureVolumeHeart(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'],modelObjects)

##PEEP Challenge Test
if 3 in test:
    simulationParams = {
        'runTime': 5,
        'base_runsToIgnore': 10,
        'base_runsToSave': 10,
        'test_runsToIgnore': 1,
        'test_runsToSave': 1,
        'test_nrRuns': 160,
        'sampPeriod': 0.01,
        'atmPressure': 760.0,
        'Cond1': 75,
        'Cond2': 150,
        'idx': 0,
        'totalTime': 0,
        'control': True,
        'calibration': False,
    }
    states, modelObjects, modelStructure, results, structures = runTestPEEPChallenge(simulationParams)
    plotSigmoidsTree(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'], modelObjects,simulationParams['idx'],structures)
    #plotSigmoid(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'],modelObjects,2,structures)
    #plots.plotPressureVolumeHeart(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'],modelObjects)
    plotTreePEEP(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'],modelObjects,simulationParams['idx'],structures)
    #plots.plotPressureVolumeHeart(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'],modelObjects)
    #plots.plotControllers(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'])

##Self + Vent
if 4 in test:
    simulationParams = {
        'runTime': 5,
        'base_runsToIgnore': 16,
        'base_runsToSave': 2,
        'test_runsToIgnore': 15,
        'test_runsToSave': 2,
        'sampPeriod': 0.01,
        'atmPressure': 760.0,
        'idx': 0,
        'control': True,
        'calibration': False,
    }
    states, modelObjects, modelStructure, results, structures = runSelfAndVent(simulationParams)
    #plotSigmoidsTree(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'], modelObjects,simulationParams['idx'],structures)
    plotLungVentComparison(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'],modelObjects,simulationParams['idx'],structures)
    #plots.plotPressureVolumeHeart(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'],modelObjects)

##Ventricle Compliance Test
if 5 in test:
    simulationParams = {
        'runTime': 5,
        'base_runsToIgnore': 16,
        'base_runsToSave': 1,
        'test_runsToIgnore': 16,
        'test_runsToSave': 1,
        'sampPeriod': 0.01,
        'atmPressure': 760.0,
        'idx': 0,
        'ventilator': False,
    }
    states, modelObjects, modelStructure, results, structures = runVentricleCompliance(simulationParams)
    #plotSigmoidsTree(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'], modelObjects,simulationParams['idx'],structures)
    plotLungVentComparison(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'],modelObjects,simulationParams['idx'],structures)
    #plots.plotPressureVolumeHeart(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'],modelObjects)


##PEEP Pleura & Thorax Complaince Test
if 6 in test:
    simulationParams = {
        'runTime': 5,
        'base_runsToIgnore': 10,
        'base_runsToSave': 10,
        'test_runsToIgnore': 0,
        'test_runsToSave': 1,
        'test_nrRuns': 104,
        'sampPeriod': 0.01,
        'atmPressure': 760.0,
        'Cond1': 47,
        'Cond2': 94,
        'idx': 0,
        'totalTime': 0,
        'control': True,
        'calibration': False,
    }
    states, modelObjects, modelStructure, results, structures = runTestPleuraThoraxCompliance(simulationParams)
    #plotSigmoidsTree(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'], modelObjects,simulationParams['idx'],structures)
    plotLungResults(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'],modelObjects,simulationParams['idx'],structures)
    plots.plotPressureVolumeHeart(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'],modelObjects)
    
    #plots.plotPressureVolumeHeart(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'],modelObjects)
    #plots.plotControllers(results, simulationParams['totalTime'], simulationParams['sampPeriod'], simulationParams['atmPressure'])




