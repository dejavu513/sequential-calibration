import matplotlib.pyplot as mpl
import matplotlib.gridspec as gridspec
import h5py as h5
import numpy as np
import os

# TODO add write metohds to be able to sve resuilts to HDF5 file

def getHdf5Data(session):
    # Sets up the folder where the data is stored and the session, on the file, to be read
    directory = os.getcwd() + '/data/'  

    # Creates a dictionary with the information to read the file
    hdf5Info = {
        'fileName' : 'VentilatorData.hdf5',
        'filePath' : directory,
        'datasetPath' : '',
        'mode' : 'r',
    }
    # Reads the file and returns the descriptor and the all the data on the file
    #descriptor,data = HDF5API.readFile(**hdf5Info)

    # Reads the data from the file from a single dataset
    # the data is flow, pressure and volume from a single session
    hdf5Info['datasetPath'] = '/data/' + session + '/Paw'
    pressureData = readDataset(**hdf5Info)
    hdf5Info['datasetPath'] = '/data/' + session + '/V'
    VOLUMEData = readDataset(**hdf5Info)
    hdf5Info['datasetPath'] = '/data/' + session + '/FLOW'
    flowData = readDataset(**hdf5Info)
    hdf5Info['datasetPath'] = '/data/' + session + '/throughs'
    startOfBreath = readDataset(**hdf5Info)



    return pressureData, VOLUMEData, flowData, startOfBreath

def plotHdf5Data(pressureData, VOLUMEData, flowData):

    t = np.arange(len(flowData))

    fig = mpl.figure(figsize=(15, 6))
    grid = gridspec.GridSpec(3, 2, figure=fig)

    ax_left = fig.add_subplot(grid[:, 0])
    ax_left.set_title('Pressure-Volume Curves')
    ax_left.plot(VOLUMEData,pressureData, label='PV', color='k')
    ax_left.legend()

    ax_right1 = fig.add_subplot(grid[0, 1])
    ax_right1.set_title('Pressure Curve')
    ax_right1.plot(t,pressureData, label='Pressure', color='k')
    ax_right1.legend()

    ax_right2 = fig.add_subplot(grid[1, 1])
    ax_right2.set_title('Volume Curve')
    ax_right2.plot(t,VOLUMEData, label='Volume', color='k')
    ax_right2.legend()

    ax_right3 = fig.add_subplot(grid[2, 1])
    ax_right3.set_title('Flow Curve')
    ax_right3.plot(t,flowData, label='Flow', color='k')
    ax_right3.legend()

    ax_right3.set_xlabel('Time [s]')

    mpl.tight_layout()
    mpl.show()

def hdf5Example():
    # Sets up the folder where the data is stored and the session, on the file, to be read
    directory = os.getcwd() + '/data/' 
    session = 'S4_1'  

    # Creates a dictionary with the information to read the file
    hdf5Info = {
        'fileName' : 'VentilatorData.hdf5',
        'filePath' : directory,
        'datasetPath' : '',
        'mode' : 'r',
    }
    # Reads the file and returns the descriptor and the all the data on the file
    #descriptor,data = HDF5API.readFile(**hdf5Info)

    # Reads the data from the file from a single dataset
    # the data is flow, pressure and volume from a single session
    hdf5Info['datasetPath'] = '/data/' + session + '/Paw'
    pressureData = readDataset(**hdf5Info)
    hdf5Info['datasetPath'] = '/data/' + session + '/V'
    VOLUMEData = readDataset(**hdf5Info)
    hdf5Info['datasetPath'] = '/data/' + session + '/FLOW'
    flowData = readDataset(**hdf5Info)

    # Plots the data
    t = np.arange(len(flowData))
    '''
    fig, (ax1, ax2, ax3) = mpl.subplots(3, 1)
    ax1.set_title('Flow, Pressure, Volume from session: ' + session)
    ax1.plot(t,flowData , label='flow', color='k')
    ax2.plot(t,pressureData, label='pressure', color='k')
    ax3.plot(t,VOLUMEData, label='volume', color='k')
    fig.show()
    '''
    fig = mpl.figure(figsize=(15, 6))
    grid = gridspec.GridSpec(3, 2, figure=fig)

    ax_left = fig.add_subplot(grid[:, 0])
    ax_left.set_title('Pressure-Volume Curves')
    ax_left.plot(VOLUMEData,pressureData, label='PV', color='k')
    ax_left.legend()

    ax_right1 = fig.add_subplot(grid[0, 1])
    ax_right1.set_title('Pressure Curve')
    ax_right1.plot(t,pressureData, label='Pressure', color='k')
    ax_right1.legend()

    ax_right2 = fig.add_subplot(grid[1, 1])
    ax_right2.set_title('Volume Curve')
    ax_right2.plot(t,VOLUMEData, label='Volume', color='k')
    ax_right2.legend()

    ax_right3 = fig.add_subplot(grid[2, 1])
    ax_right3.set_title('Flow Curve')
    ax_right3.plot(t,flowData, label='Flow', color='k')
    ax_right3.legend()

    ax_right3.set_xlabel('Time [s]')

    mpl.tight_layout()
    mpl.show()

def readFile(filePath, fileName, datasetPath, mode='r'):
    print('Reading File: ' + fileName)
    fileObject = h5.File(filePath + fileName, mode)
    #################################################################
    ###################### Descriptor ###############################
    #################################################################
    # Is a Compound dataset composed of strings and numbers
    # Use variable 'descriptor' to interact with the data on this dataset

    # Object to interact with the descriptor dataset
    descriptorObject = fileObject['/descriptor']

    # Object describing the headers of the descriptor
    descTypes = descriptorObject.dtype

    # List with the names of the descriptor headers
    headerNames = descTypes.names
    #print(headerNames)

    # Empty Dictionary to hold the data on the descriptor
    descriptor = {}

    # Loop the header names in order to ascertain the data type of a particular 
    #   header and add it to a dictionary
    for header in headerNames:
        # For byte strings dtype='b' -> loop the string in order to convert 
        #   to regular string then add them to the dictionary
        if np.dtype(descTypes[header]) == np.dtype('O'): # the type stated is 'O' bit in reality is 'b'
            arr=descriptorObject[header]
            strList = []
            for byteStr in arr:
                strList.append(byteStr.decode('UTF-8'))
            descriptor[header]=strList   
        # For floats -> we can add them directly to the dictionary
        elif np.dtype(descTypes[header]) == np.dtype('<f8'): # For Floats
            descriptor[header]=descriptorObject[header]
        # If more types are present we ignore for now
        else:
            #print('other')   
            pass

    # Alternatively one can also call the data directly form the object but the 
    # hdf5 file must be open
    #print(descriptorObject['RR'])

    # and the strings will be with the byteString type
    #print(descriptorObject['FileName'])

    #################################################################
    ###################### DATA #####################################
    #################################################################
    # Creates a dictionary with all the data
    # Use the variable 'data' to interact with the data on this group 


    # Creates an object for the 'data' group
    dataGroup = fileObject['/data']
    data = {}

    # Iterates the groups (sessions) of datasets on the data group
    for group in dataGroup:
        #print(group) # to keep track of where in the file we are
        
        # Holds the names of the groups containing the sessions
        sessionGroup = dataGroup[group]
        # Holds the data for the current session
        Session={}
        
        # Iterates the datasets on a session group
        for datasetName in sessionGroup:
            # Gets the object for the dataset
            dataset = sessionGroup[datasetName]
            
            # Checks the type of the dataset
            if dataset.dtype == 'float64': # If Float
                # Convert the dataset into an array and save to the dictionary 
                Session[datasetName]=dataset[()]
                # if not a pulse nor a throughs array (therefore a Raw data array)
                #   -> create the Raw data pulses and save to a dictionary
                if ('Pulses' not in datasetName) and ('throughs' not in datasetName): #Is a Raw data dataset
                    # Read the throughs data set to get the points where a pulse starts
                    throughsDataset = sessionGroup['throughs']
                    throughs = throughsDataset[()]
                    rawPulse = {}
                    # Loop through the points in order to cut the raw data
                    for i in np.arange(1,len(throughs),1):
                        # get the raw data array
                        arr = dataset[()]
                        # Cut the Pulse and save into a new array
                        newarr = arr[int(throughs[i-1]):int(throughs[i])]
                        # Save into dictionary (did it like this because the pulses will have different lengths)
                        rawPulse['P_' + str(i)] = newarr
                    # Save the whole dictionary into the session dictionary
                    Session[datasetName + '_RawPulse'] = rawPulse     
            else: # Must be the SAX string dataset (should be testing for the actual compound type)
                # Dictionary to hold the SAX string for this session
                SAX_dictionary={}
                # Iterates the headers of the SAX compound dataset (ex. w=6 a=6)
                for name in dataset.dtype.names:
                    # List to hold the processed byte strings for every modality
                    strList=[]
                    # Iterates every SAX string in order to convert it from 
                    #   byteString into regular string
                    for byteStr in dataset[name]:
                        # Converts the string from byteString into regular string
                        strList.append(byteStr.decode('UTF-8'))
                    # Appends the whole column into the SAX dictionary
                    SAX_dictionary[name] = strList
                
                # Appends the SAX Disctionary this modality into the session Dectionary 
                Session[datasetName] = SAX_dictionary
        # Appends the session dictionary into the data dictionary
        data[group]=Session

    print('File read')
        
    return descriptor,data

def readDataset(filePath, fileName, datasetPath, mode='r'):
    fileObject = h5.File(filePath + fileName, mode)
    dataset = fileObject[datasetPath]
    data=[] 
    # Checks the type of the dataset
    if dataset.dtype == 'float64': # If Float 
        data=dataset[()]    
    else: # Must be a compound dataset (should be testing for the actual type)
        # Dictionary to hold the data
        dictionary={}
        # Iterates the headers of the compound dataset
        for name in dataset.dtype.names:
            #print(dataset[name].dtype)
            if dataset[name].dtype == 'float64':
                dictionary[name] = dataset[name]
            else:
                # List to hold the processed byte strings for every modality
                strList=[]
                # Iterates every string in order to convert it from 
                #   byteString into regular string
                for byteStr in dataset[name]:
                    # Converts the string from byteString into regular string
                    strList.append(byteStr.decode('UTF-8'))
                    # Appends the whole column into the SAX dictionary
                    dictionary[name] = strList
        
        # Appends the SAX Disctionary this modality into the session Dectionary 
        data = dictionary
    
    return data
