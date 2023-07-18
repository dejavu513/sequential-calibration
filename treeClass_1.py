# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 18:03:11 2022

@author: mtc86
"""
import math as m
import numpy as np


##################### Class for the base nodes ################################
class baseNode:
    level = 0
    branch = 0
    start_coordinate = [0, 0]
    angle = 0
    plane = ''
    name = ''
    node_Input = ''

    
    pressure = ''
    node_Current = ''
    

    
    

    maxLevels = 0
    isLeaf = False

    
    end_coordinate = [0, 0]

    radius = 0
    length = 0
    area = 0
    

    children = []

    #  ___       _ _     __  __      _   _               _
    # |_ _|_ __ (_) |_  |  \/  | ___| |_| |__   ___   __| |___
    #  | || '_ \| | __| | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
    #  | || | | | | |_  | |  | |  __/ |_| | | | (_) | (_| \__ \
    # |___|_| |_|_|\__| |_|  |_|\___|\__|_| |_|\___/ \__,_|___/

    def __init__(self, params):

        self.name = self.buildName( params['branch'], params['name'])  # ID
        self.level = params['level']
        self.branch = params['branch']
        self.conf = params['conf']

        if self.level > 0:
            self.parent = params['name']
        else:
            self.parent = ''

        # --------- circuit ######################################################
        self.node_Input = params['inputName']
        self.pressure = self.addPrefix('V_')
        self.node_Current = self.addPrefix('I_')
        self.capacitor = {
            'id': self.addPrefix('C_'),
            'pressure': self.addPrefix('V_'),
            'typeC': 0,
            'typeC_params': {
                'C': self.conf[self.level]['capacity']
            },
            'y0': 0,
        }
        self.resistorIn = {
            'id': self.addPrefix('R_'),
            'current': self.addPrefix('I_'),
            'pressureIn': self.node_Input,
            'pressureOut': self.addPrefix('V_'),
            'typeR': 0,
            'typeR_params': {
                'R': self.conf[self.level]['resistance']
            },
        }

        self.params = params

        # --------- gas exchange #################################################
        self.gases = params['gases']
        self.gasConcentrations = params['gasConcentrations']
        if self.gases:
            self.partialPressures = []
            for gas in self.gases:
                self.partialPressures.append(self.addPrefix('V' + gas + '_'))
            self.partialPressuresY0 = []
            for concentration in self.gasConcentrations:
                self.partialPressuresY0.append(concentration)

        # --------- representation ###############################################
        self.radius = self.conf[self.level]['radius']
        self.length = self.conf[self.level]['length']
        self.angle = params['angle']
        self.area = m.pi * (self.radius ** 2)
        self.plane = params['plane']

        self.start_coordinate = params['start_coordinate']
        if len(self.start_coordinate) == 2:
            self.end_coordinate = self.calculateEndcoordinate()
        elif len(self.start_coordinate) == 3:
            self.end_coordinate = self.calculate3DEndcoordinate()


    # generates the name/id of the node    
    def buildName(self, branch, Pname):
        name = Pname + '|' + str(branch)
        return name

    def addPrefix(self, prefix):
        name = prefix + self.name
        return name

    # Calculates the coordinate of the next junction using the start coordinate and the length    
    def calculateEndcoordinate(self):
        x = self.start_coordinate[0] + self.length * m.cos(self.angle)
        y = self.start_coordinate[1] + self.length * m.sin(self.angle)

        return [x, y]

    # Calculates the coordinate of the next junction using the start coordinate and the length    
    # But in 3D with a 90deg rotation 
    def calculate3DEndcoordinate(self):

        ang = self.angle
        l = self.length
        x = self.start_coordinate[0]
        y = self.start_coordinate[1]
        z = self.start_coordinate[2]

        startLevel = 1

        if self.name.find('S|0|0'):
            if (self.plane == 'xz') & (self.level > startLevel):
                x = x + l * m.cos(ang)
                z = z + l * m.sin(ang)
                # elif (self.plane == 'xy') & (self.level > 2):
                # x = x + l * m.cos(ang)
                # y = y + l * m.sin(ang)
            elif (self.plane == 'yz') & (self.level > startLevel):
                y = y + l * m.cos(ang)
                z = z + l * m.sin(ang)
            else:
                y = y + l * m.cos(ang)
                z = z + l * m.sin(ang)
        else:
            if (self.plane == 'xz') & (self.level > startLevel):
                x = x + l * -m.cos(ang)
                z = z + l * m.sin(ang)
                # elif (self.plane == 'xy') & (self.level > 2):
                # x = x + l * m.cos(ang)
                # y = y + l * m.sin(ang)
            elif (self.plane == 'yz') & (self.level > startLevel):
                y = y + l * m.cos(ang)
                z = z + l * m.sin(ang)
            else:
                y = y + l * m.cos(ang)
                z = z + l * m.sin(ang)

        return [x, y, z]

    #  _____                __  __      _   _               _
    # |_   _| __ ___  ___  |  \/  | ___| |_| |__   ___   __| |___
    #   | || '__/ _ \/ _ \ | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
    #   | || | |  __/  __/ | |  | |  __/ |_| | | | (_) | (_| \__ \
    #   |_||_|  \___|\___| |_|  |_|\___|\__|_| |_|\___/ \__,_|___/

    # Builds a 2D tree of nodes
    def buildTree(self, maxLevels, branchAngle):
        self.maxLevels = maxLevels
        self.children = []
        self.isLeaf = self.checkLeaf()

        newAngles = np.linspace(self.angle - branchAngle,self.angle + branchAngle,2)

        # ---------- Initialise new Branch #####################################
        if self.level <= (maxLevels - 1):  # BrancRches
            for i in np.arange(0, 2, 1):
                newNode = baseNode(
                    {
                        # -------------------------------- Coordinate Parameters
                        'level': self.level + 1,
                        'branch': i,
                        'start_coordinate': self.end_coordinate,
                        'angle': newAngles[i],
                        'branchAngle': np.pi / 2.5,
                        'plane': '',
                        # -------------------------------------- Tree parameters
                        'name': self.name,
                        'conf': self.conf,
                        'inputName': self.pressure,
                        # --------------------------------------- Gas Parameters
                        'gasExchange': False,
                        'gases': self.gases,
                        'gasConcentrations': self.gasConcentrations,
                    }
                )

                newNode.buildTree(maxLevels, branchAngle)
                self.children.append(newNode)

        elif self.level == maxLevels:
            if self.gases:
                self.gasExchangeResistors = []
                for gas in self.gases:
                    resistor = {
                        'id': self.addPrefix('R' + gas + '_'),
                        'current': self.addPrefix('I' + gas + '_'),
                        'pressureIn': self.addPrefix('V' + gas + '_'),
                        'pressureOut': 'V' + gas + '_Blood',
                        'typeR': 1,
                        'typeR_params': {},
                    }
                    self.gasExchangeResistors.append(resistor)

        # Builds a 2D tree of nodes

    def buildTreeForParameterEstimation(self, maxLevels, branchAngle):
        self.maxLevels = maxLevels
        self.children = []
        self.isLeaf = self.checkLeaf()

        newAngles = np.linspace(self.angle - branchAngle, self.angle + branchAngle, 2)

        #print(self.level)
        # ---------- Initialise new Branch #####################################
        if self.level <= (maxLevels - 1):  # BrancRches
            for i in np.arange(0, 2, 1):
                newNode = baseNode(
                    {
                        # -------------------------------- Coordinate Parameters
                        'level': self.level + 1,
                        'branch': i,
                        'start_coordinate': self.end_coordinate,
                        'angle': newAngles[i],
                        'branchAngle': np.pi / 2.5,
                        'plane': '',
                        # -------------------------------------- Tree parameters
                        'name': self.name,
                        'conf': self.conf,
                        'inputName': self.pressure,
                        # --------------------------------------- Gas Parameters
                        'gasExchange': False,
                        'gases': self.gases,
                        'gasConcentrations': self.gasConcentrations,
                        'nrBranches': self.params['nrBranches'],
                        'nrOriginalVessels': self.params['nrOriginalVessels']
                    }
                )

                newNode.buildTreeForParameterEstimation(maxLevels, branchAngle)
                self.children.append(newNode)

        elif self.level == maxLevels:
            if self.gases:
                self.gasExchangeResistors = []
                for gas in self.gases:
                    resistor = {
                        'id': self.addPrefix('R' + gas + '_'),
                        'current': self.addPrefix('I' + gas + '_'),
                        'pressureIn': self.addPrefix('V' + gas + '_'),
                        'pressureOut': 'V' + gas + '_Blood',
                        'typeR': 1,
                        'typeR_params': {},
                    }
                    self.gasExchangeResistors.append(resistor)

        
        if self.level == maxLevels:
            if len(self.params['nrOriginalVessels'] ) > 0:
                self.capacitor['typeC_params']['C'] = self.capacitor['typeC_params']['C'] * self.params['nrOriginalVessels'][self.level]
        else:
            childrenResistors = self.getResistorsOut()
            R1 = childrenResistors[0]['typeR_params']['R']
            R2 = childrenResistors[1]['typeR_params']['R']

            childrenResistors = self.getChildCapacitors()
            C1 = childrenResistors[0]['typeC_params']['C']
            C2 = childrenResistors[1]['typeC_params']['C']
            

            if self.params['nrBranches'][self.level] == 2:
                childrenResistorsEquivalent = (R1 * R2) / (R1 + R2)
                self.resistorIn['typeR_params']['R'] = self.resistorIn['typeR_params']['R'] + childrenResistorsEquivalent

                if len(self.params['nrOriginalVessels'] ) > 0:
                    self.capacitor['typeC_params']['C'] = C1  + self.capacitor['typeC_params']['C']* self.params['nrOriginalVessels'][self.level]
                else:
                    self.capacitor['typeC_params']['C'] = C1 + C2 + self.capacitor['typeC_params']['C']
            else:
                childrenResistorsEquivalent = R1/self.params['nrBranches'][self.level]
                self.resistorIn['typeR_params']['R'] = self.resistorIn['typeR_params']['R'] + childrenResistorsEquivalent
                
                
                self.capacitor['typeC_params']['C'] = (C1) + self.capacitor['typeC_params']['C'] * self.params['nrOriginalVessels'][self.level]



            if self.level == 0:
                # print('Resistance for: ' + self.name + ' is ' + str(self.resistorIn['typeR_params']['R']))
                pass

    # TODO fix the method
    # Builds a 3D tree of nodes
    def build3DTree(self, maxLevels, branchAngle, lengthDecay):
        self.maxLevels = maxLevels
        self.children = []
        self.isLeaf = self.checkLeaf()

        newAngles = np.linspace(self.angle - branchAngle,
                                self.angle + branchAngle,
                                2
                                )

        # According to Murray's Law
        newRadius = ((self.radius ** 3) / 2) ** (1 / 3)

        # New Length of the tube
        # newLength = self.length/(lengthDecay)
        newLength = 0.12 / (2 * (self.level + 1))

        if self.level < maxLevels:
            newPlane = ''
            if self.plane == 'xz':
                newPlane = 'yz'
            elif self.plane == 'xy':
                newPlane = 'yz'
            elif self.plane == 'yz':
                newPlane = 'xz'

            for i in np.arange(0, 2, 1):
                newNode = baseNode(
                    self.end_coordinate,
                    newLength,
                    newAngles[i],
                    newRadius,
                    self.level + 1,
                    i,
                    self.pressure,
                    self.name,
                    newPlane,
                    self.gases,
                    self.gasConcentrations,
                )
                newNode.build3DTree(maxLevels, branchAngle, lengthDecay)

                self.children.append(newNode)

    # gets the node information of the tree as a 2D dictionary
    def getTreeArray(self):
        treeArr = []
        treeArr.append(self.getTreeNode())
        for node in self.getChildArray():
            treeArr.append(node)
        return treeArr

    #  _____                ____            _            __  __      _   _               _
    # |_   _| __ ___  ___  |  _ \ _ __ ___ | |__   ___  |  \/  | ___| |_| |__   ___   __| |___
    #   | || '__/ _ \/ _ \ | |_) | '__/ _ \| '_ \ / _ \ | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
    #   | || | |  __/  __/ |  __/| | | (_) | |_) |  __/ | |  | |  __/ |_| | | | (_) | (_| \__ \
    #   |_||_|  \___|\___| |_|   |_|  \___/|_.__/ \___| |_|  |_|\___|\__|_| |_|\___/ \__,_|___/

    # gets the information of the node
    def getTreeNode(self):
        children = self.getChildNode()
        childrenNames = [child['id'] for child in children]
        if self.gases:
            if self.isLeaf:
                currentsOut = []
                for res in self.gasExchangeResistors:
                    currentsOut.append(res['current'])

                thisNode = {
                    'id': self.name,
                    'isLeaf': self.isLeaf,
                    'level': self.level,
                    'pressure': self.pressure,
                    'currentsIn': [self.node_Current],
                    # 'currentsOut':self.getCurrentsOut(),
                    'currentsOut': currentsOut,
                    'capacitor': self.capacitor,
                    'resistorsIn': [self.resistorIn],
                    # 'resistorsOut': self.getResistorsOut(),
                    'resistorsOut': self.gasExchangeResistors,
                    'resistorsgasExchange': self.gasExchangeResistors,
                    'partialPressures': self.partialPressures,
                    'partialPressuresY0': self.partialPressuresY0,
                    'parent': self.parent,
                    'children': childrenNames,
                    'rep': {
                        'startXY': self.start_coordinate,
                        'endXY': self.end_coordinate,
                        'length': self.length,
                        'radius': self.radius,
                        'area': self.area,
                        'plane': self.plane,
                        'isLeaf': self.isLeaf,
                    },
                }
            else:
                thisNode = {
                    'id': self.name,
                    'isLeaf': self.isLeaf,
                    'level': self.level,
                    'pressure': self.pressure,
                    'currentsIn': [self.node_Current],
                    'currentsOut': self.getCurrentsOut(),
                    'capacitor': self.capacitor,
                    'resistorsIn': [self.resistorIn],
                    'resistorsOut': self.getResistorsOut(),
                    'partialPressures': self.partialPressures,
                    'partialPressuresY0': self.partialPressuresY0,
                    'parent': self.parent,
                    'children': childrenNames,
                    'rep': {
                        'startXY': self.start_coordinate,
                        'endXY': self.end_coordinate,
                        'length': self.length,
                        'radius': self.radius,
                        'area': self.area,
                        'plane': self.plane,
                        'isLeaf': self.isLeaf,
                    },
                }
        else:
            thisNode = {
                'id': self.name,
                'isLeaf': self.isLeaf,
                'level': self.level,
                'pressure': self.pressure,
                'currentsIn': [self.node_Current],
                'currentsOut': self.getCurrentsOut(),
                'capacitor': self.capacitor,
                'resistorsIn': [self.resistorIn],
                'resistorsOut': self.getResistorsOut(),
                # 'partialPressures': self.partialPressures,
                # 'partialPressuresY0': self.partialPressuresY0,
                # 'parent': self.parent,
                # 'children': childrenNames,
                'rep': {
                    'startXY': self.start_coordinate,
                    'endXY': self.end_coordinate,
                    'length': self.length,
                    'radius': self.radius,
                    'area': self.area,
                    'plane': self.plane,
                    'isLeaf': self.isLeaf,
                },
            }

        return thisNode

    # gets the information of the children nodes
    def getCurrentsOut(self):
        currentsOut = []
        if len(self.children) > 0:
            for child in self.children:
                childCurrent = child.node_Current
                currentsOut.append(childCurrent)
        return currentsOut

    # gets the information of the children resistors
    def getResistorsOut(self):
        resistorsOut = []
        if len(self.children) > 0:
            for child in self.children:
                childResistor = child.resistorIn
                resistorsOut.append(childResistor)
        return resistorsOut

    # gets the information of the children capacitors
    def getChildCapacitors(self):
        capacitors = []
        if len(self.children) > 0:
            for child in self.children:
                childCapacitor = child.capacitor
                capacitors.append(childCapacitor)
        return capacitors


    # gets the information of the children nodes
    def getChildNode(self):
        childNodes = []
        if len(self.children) > 0:
            for child in self.children:
                childNode = child.getTreeNode()
                childNodes.append(childNode)
        return childNodes

    # recursive method to extract the node information of the whole tree
    def getChildArray(self):
        dicData = []
        if len(self.children) > 0:
            for node in self.getChildNode():
                dicData.append(node)
            for child in self.children:
                childtree = child.getChildArray()
                if len(childtree) > 0:
                    for node in childtree:
                        dicData.append(node)

        return dicData

    # Check if this node is an alveoli
    def checkLeaf(self):
        isLeaf = False
        if self.level == self.maxLevels:
            isLeaf = True
        return isLeaf

    #  ____  _        _               __  __      _   _               _
    # / ___|| |_ _ __(_)_ __   __ _  |  \/  | ___| |_| |__   ___   __| |___
    # \___ \| __| '__| | '_ \ / _` | | |\/| |/ _ \ __| '_ \ / _ \ / _` / __|
    #  ___) | |_| |  | | | | | (_| | | |  | |  __/ |_| | | | (_) | (_| \__ \
    # |____/ \__|_|  |_|_| |_|\__, | |_|  |_|\___|\__|_| |_|\___/ \__,_|___/
    #                         |___/

    def __repr__(self):
        return "baseNode('{}', '{}', '{}', '{}', '{}', '{}')".format(
            self.start_coordinate,
            self.length,
            self.angle,
            self.node_Input,
            self.branch,
            self.level)

    def __str__(self):
        string = [
            'node_Input - ' + self.node_Input,
            'pressure - ' + self.pressure,
            'name - ' + self.name,
            'level - ' + str(self.level),
            'branch - ' + str(self.branch),
            'start_coordinate - ' + str(self.start_coordinate),
            'end_coordinate - ' + str(self.end_coordinate),
            'length - ' + str(self.length),
            'angle - ' + str(self.angle * 180 / m.pi)
        ]
        return "\n".join(string)
