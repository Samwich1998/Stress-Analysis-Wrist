
# Basic Modules
import math
import numpy as np
# Peak Detection Modules
import scipy
import scipy.signal
# Data Filtering Modules
from scipy.signal import butter
from scipy.signal import savgol_filter
# Matlab Plotting Modules
import matplotlib.pyplot as plt
# Feature Extraction Modules
from scipy.stats import skew
from scipy.stats import entropy
from scipy.stats import kurtosis
from scipy.fft import fft, ifft
        
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class signalProcessing:
    
    def __init__(self, startStimulus, stimulusDuration, stimulusBuffer, plotData = False):
        self.glucoseFeatures = []
        self.lactateFeatures = []
        self.uricAcidFeatures = []
        
        self.featureLabelsGlucose = []
        self.featureLabelsLactate = []
        self.featureLabelsUricAcid = []
        
        self.lowPassCutoff = 0.01
        
        self.minPeakDuration = 20
        
        self.minLeftBoundaryInd = 100
        
        self.plotData = plotData
        
        self.startStimulus = startStimulus
        self.stimulusDuration = stimulusDuration + stimulusBuffer
        
        self.peakData = {"Lactate":[], "Glucose":[], "Uric Acid":[]}
    
    def analyzeData(self, xData, yData, chemicalName = ""):
        if not self.continueAnalysis:
            return []
        
        # ------------------------- Filter the Data ------------------------- #
        # Apply a Low Pass Filter
        self.samplingFreq = len(xData)/(xData[-1] - xData[0])
        yData = self.butterFilter(yData, self.lowPassCutoff, self.samplingFreq, order = 4, filterType = 'low')
        # ------------------------------------------------------------------- #
        
        # ------------------- Find and Remove the Baseline ------------------ #
        # Find Peaks in the Data
        chemicalPeakInd = self.findPeak(xData, yData)
        # Return None if No Peak Found
        if chemicalPeakInd == None:
            print("No Peak Found in " + chemicalName + " Data")
            self.plot(xData, yData, [], [], 0, 0, 0, chemicalName + " NO PEAK FOUND")
            self.continueAnalysis = False
            #sys.exit()
            return [] #[0]*74 if len(self.glucoseFeatures) == 0 else [0]*len(self.glucoseFeatures[0])
        # ------------------------------------------------------------------- #

        # ------------------- Find and Remove the Baseline ------------------ #
        # Get Baseline from Best Linear Fit
        leftCutInd, rightCutInd = self.findLinearBaseline(xData, yData, chemicalPeakInd)
        if None in [leftCutInd, rightCutInd] or abs(leftCutInd - rightCutInd) < self.minPeakDuration or self.minLeftBoundaryInd + 1 >= leftCutInd:
            print("No Baseline Found in " + chemicalName + " Data")
            self.plot(xData, yData, [], [], chemicalPeakInd, 0, 0, chemicalName + " NO BASELINE FOUND")
            self.continueAnalysis = False
            return []
        
        # Fit Lines to Ends of Graph
        lineSlope, slopeIntercept = np.polyfit(xData[[leftCutInd, rightCutInd]], yData[[leftCutInd, rightCutInd]], 1)
        linearFit = lineSlope*xData + slopeIntercept
        
        # Apply a Smoothing Function
        yData = savgol_filter(yData, max(3, self.convertToOddInt((rightCutInd-leftCutInd)/9)), 2)  # 61/3,2
            
        # Piece Together yData's Baseline
        baseline = np.concatenate((yData[0:leftCutInd], linearFit[leftCutInd: rightCutInd+1], yData[rightCutInd+1:len(yData)]))
        # Find yData After Baseline Subtraction
        baselineData = yData - baseline
        # ------------------------------------------------------------------- #
        
        # -------------------- Extract Features from Peak ------------------- #
        # Adjust the Peak's Ind After Baseline Subtraction
        chemicalPeakInd = self.findNearbyMaximum(baselineData, chemicalPeakInd, binarySearchWindow = 4)
        chemicalPeakInd = self.findNearbyMaximum(baselineData, chemicalPeakInd, binarySearchWindow = -4)
        
        # Establish New Peak Boundaries; But Dont Take Past the CutOffs (as it is Zero)
        leftBaseInd = max(leftCutInd, self.findNearbyMinimum(baselineData, chemicalPeakInd, binarySearchWindow = -2))
        rightBaseInd = min(rightCutInd, self.findNearbyMinimum(baselineData, chemicalPeakInd, binarySearchWindow = 2))
        
        if rightBaseInd - leftBaseInd  < 250:
            print("Peak too small duration in " + chemicalName + " Data")
            self.continueAnalysis = False
            return []
        # Extract the Features from the Data
        peakFeatures = self.extractFeatures(xData[leftBaseInd:rightBaseInd+1] - xData[leftBaseInd], baselineData[leftBaseInd:rightBaseInd+1], chemicalPeakInd-leftBaseInd, chemicalName)
        #peakFeatures = self.extractFeatures_Pointwise(xData[leftBaseInd:rightBaseInd+1] - xData[leftBaseInd], baselineData[leftBaseInd:rightBaseInd+1], chemicalPeakInd-leftBaseInd, chemicalName)
        # ------------------------------------------------------------------- #
        
        # -------------------------- Plot the Data -------------------------- #
        if self.plotData:
            self.plot(xData, yData, baselineData, linearFit, chemicalPeakInd, leftCutInd, rightCutInd, chemicalName)
        # ------------------------------------------------------------------- #
        
        if len(peakFeatures) == 0:
            print("No Features Found in " + chemicalName + " Data")
            self.continueAnalysis = False
        else:  
            self.peakData[chemicalName].append((xData[leftBaseInd:rightBaseInd+1] - xData[leftBaseInd], baselineData[leftBaseInd:rightBaseInd+1]))
        return peakFeatures
    
    def analyzeChemicals(self, timePoints, glucose, lactate, uricAcid, label, analyzeTogether):
        # Assert All Chemical Data is Well-Formed 
        self.continueAnalysis = True
        
        # Analyze Each Chemical
        if len(glucose) > 0:
            glucoseFeatures = self.analyzeData(timePoints, glucose, chemicalName = "Glucose")
            if not analyzeTogether:
                self.continueAnalysis = True
            
        if self.continueAnalysis and len(lactate) > 0:
            lactateFeatures = self.analyzeData(timePoints, lactate, chemicalName = "Lactate")
            if not analyzeTogether:
                self.continueAnalysis = True
        
        if self.continueAnalysis and len(uricAcid) > 0:
            uricAcidFeatures = self.analyzeData(timePoints, uricAcid, chemicalName = "Uric Acid")
            if not analyzeTogether:
                self.continueAnalysis = True
            
        # If Features Found for ALL Chemicals
        if self.continueAnalysis:
            # Keep Track of the Features
            if len(glucose) > 0 and len(glucoseFeatures) > 0:
                self.glucoseFeatures.extend(glucoseFeatures)
                for _ in range(len(glucoseFeatures)):
                    self.featureLabelsGlucose.append(label)
                    
            if len(lactate) > 0 and len(lactateFeatures) > 0:
                self.lactateFeatures.extend(lactateFeatures)
                for _ in range(len(lactateFeatures)):
                    self.featureLabelsLactate.append(label)

            if len(uricAcid) > 0 and len(uricAcidFeatures) > 0:
                self.uricAcidFeatures.extend(uricAcidFeatures)
                for _ in range(len(uricAcidFeatures)):
                    self.featureLabelsUricAcid.append(label)

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    
    def convertToOddInt(self, x):
        return 2*math.floor((x+1)/2) - 1
    
    def findPeak(self, xData, yData, ignoredBoundaryPoints = 10, deriv = False):
        # Find All Peaks in the Data
        peakInfo = scipy.signal.find_peaks(yData, prominence=10E-10, width=20, distance = 20)
        # Extract the Peak Information
        peakProminences = peakInfo[1]['prominences']
        peakIndices = peakInfo[0]
        
        # Remove Peaks Nearby Boundaries
        allProminences = peakProminences[np.logical_and(peakIndices < len(xData) - ignoredBoundaryPoints, peakIndices >= ignoredBoundaryPoints)]
        peakIndices = peakIndices[np.logical_and(peakIndices < len(xData) - ignoredBoundaryPoints, peakIndices >= ignoredBoundaryPoints)]
        # Seperate Out the Stimulus Window
        allProminences = allProminences[self.startStimulus < xData[peakIndices]]
        peakIndices = peakIndices[self.startStimulus < xData[peakIndices]]
        allProminences = allProminences[self.startStimulus + self.stimulusDuration > xData[peakIndices]]
        peakIndices = peakIndices[self.startStimulus + self.stimulusDuration > xData[peakIndices]]

        # If Peaks are Found
        if len(peakIndices) > 0:
            # Take the Most Prominent Peak
            bestPeak = allProminences.argmax()
            peakInd = peakIndices[bestPeak]
            return peakInd
        elif not deriv:
            filteredVelocity = savgol_filter(np.gradient(yData), 251, 3)
            return self.findPeak(xData, filteredVelocity, deriv = True)
        # If No Peak is Found, Return None
        return None
    
    def butterParams(self, cutoffFreq = [0.1, 7], samplingFreq = 800, order=3, filterType = 'band'):
        nyq = 0.5 * samplingFreq
        if filterType == "band":
            normal_cutoff = [freq/nyq for freq in cutoffFreq]
        else:
            normal_cutoff = cutoffFreq / nyq
        sos = butter(order, normal_cutoff, btype = filterType, analog = False, output='sos')
        return sos
    
    def butterFilter(self, data, cutoffFreq, samplingFreq, order = 3, filterType = 'band'):
        sos = self.butterParams(cutoffFreq, samplingFreq, order, filterType)
        return scipy.signal.sosfiltfilt(sos, data)
    
    def findClosestExtrema(self, data, xPointer, binarySearchWindow = 1, maxPointsSearch = 500):
        if binarySearchWindow < 0:
            print("Setting binarySearchWindow to Positive")
            binarySearchWindow = abs(binarySearchWindow)
            
        # Check the Trends on Both Sides
        leftPoint = data[xPointer - abs(binarySearchWindow)]
        rightPoint = data[xPointer + abs(binarySearchWindow)]
        leftSlopesDown = leftPoint < data[xPointer]
        rightSlopesDown = rightPoint < data[xPointer]
                
        # If Both Sides Slope the Same Way, its an Extrema
        if leftSlopesDown == rightSlopesDown:
            return xPointer
        # If the Left Goes Down and the Right Goes Up -> its a positive slope
        elif leftSlopesDown and not rightSlopesDown:
            newPointer_Left = self.findNearbyMinimum(data, xPointer, -binarySearchWindow, maxPointsSearch)
            newPointer_Right = self.findNearbyMaximum(data, xPointer, binarySearchWindow, maxPointsSearch)
        # If the Left Goes Up and the Right Goes Down -> its a Negative slope
        else:
            newPointer_Left = self.findNearbyMinimum(data, xPointer, binarySearchWindow, maxPointsSearch)
            newPointer_Right = self.findNearbyMaximum(data, xPointer, -binarySearchWindow, maxPointsSearch)
        # Return the Furthest Pointer Away
        return min([newPointer_Left, newPointer_Right], key = lambda ind: abs(xPointer - ind))  
    
    def findClosestMax(self, data, xPointer, binarySearchWindow = 1, maxPointsSearch = 500):
        newPointer_Left = self.findNearbyMaximum(data, xPointer, -binarySearchWindow, maxPointsSearch)
        newPointer_Right = self.findNearbyMaximum(data, xPointer, binarySearchWindow, maxPointsSearch)
        if newPointer_Right == xPointer and newPointer_Left == xPointer:
            return self.findClosestMax(data, xPointer, binarySearchWindow*2, maxPointsSearch)
        if newPointer_Right == xPointer:
            return newPointer_Left
        elif newPointer_Left == xPointer:
            return newPointer_Right
        else:
            return min([newPointer_Left, newPointer_Right], key = lambda ind: abs(xPointer - ind))  
            
    def findNearbyMinimum(self, data, xPointer, binarySearchWindow = 5, maxPointsSearch = 500):
        """
        Search Right: binarySearchWindow > 0
        Search Left: binarySearchWindow < 0
        """
        # Base Case
        if abs(binarySearchWindow) < 1 or maxPointsSearch == 0:
            searchSegment = data[max(0,xPointer-1):min(xPointer+2, len(data))]
            xPointer -= np.where(searchSegment==data[xPointer])[0][0]
            return xPointer + np.argmin(searchSegment) 
        
        maxHeightPointer = xPointer
        maxHeight = data[xPointer]; searchDirection = binarySearchWindow//abs(binarySearchWindow)
        # Binary Search Data to Find the Minimum (Skip Over Minor Fluctuations)
        for dataPointer in range(max(xPointer, 0), max(0, min(xPointer + searchDirection*maxPointsSearch, len(data))), binarySearchWindow):
            # If the Next Point is Greater Than the Previous, Take a Step Back
            if data[dataPointer] >= maxHeight and xPointer != dataPointer:
                return self.findNearbyMinimum(data, dataPointer - binarySearchWindow, round(binarySearchWindow/4), maxPointsSearch - searchDirection*(abs(dataPointer - binarySearchWindow)) - xPointer)
            # Else, Continue Searching
            else:
                maxHeightPointer = dataPointer
                maxHeight = data[dataPointer]

        # If Your Binary Search is Too Large, Reduce it
        return self.findNearbyMinimum(data, maxHeightPointer, round(binarySearchWindow/2), maxPointsSearch-1)
    
    def findNearbyMaximum(self, data, xPointer, binarySearchWindow = 5, maxPointsSearch = 500):
        """
        Search Right: binarySearchWindow > 0
        Search Left: binarySearchWindow < 0
        """
        # Base Case
        xPointer = min(max(xPointer, 0), len(data)-1)
        if abs(binarySearchWindow) < 1 or maxPointsSearch == 0:
            searchSegment = data[max(0,xPointer-1):min(xPointer+2, len(data))]
            xPointer -= np.where(searchSegment==data[xPointer])[0][0]
            return xPointer + np.argmax(searchSegment)
        
        minHeightPointer = xPointer; minHeight = data[xPointer];
        searchDirection = binarySearchWindow//abs(binarySearchWindow)
        # Binary Search Data to Find the Minimum (Skip Over Minor Fluctuations)
        for dataPointer in range(xPointer, max(0, min(xPointer + searchDirection*maxPointsSearch, len(data))), binarySearchWindow):
            # If the Next Point is Greater Than the Previous, Take a Step Back
            if data[dataPointer] < minHeight and xPointer != dataPointer:
                return self.findNearbyMaximum(data, dataPointer - binarySearchWindow, round(binarySearchWindow/2), maxPointsSearch - searchDirection*(abs(dataPointer - binarySearchWindow)) - xPointer)
            # Else, Continue Searching
            else:
                minHeightPointer = dataPointer
                minHeight = data[dataPointer]

        # If Your Binary Search is Too Large, Reduce it
        return self.findNearbyMaximum(data, minHeightPointer, round(binarySearchWindow/2), maxPointsSearch-1)
    
    
    def findLinearBaseline(self, xData, yData, peakInd):
        # Define a threshold for distinguishing good/bad lines
        maxBadPointsTotal = int(len(xData)/10)
        # Store Possibly Good Tangent Indexes
        goodTangentInd = [[] for _ in range(maxBadPointsTotal)]
        
        # For Each Index Pair on the Left and Right of the Peak
        for rightInd in range(peakInd+2, len(yData), 1):
            for leftInd in range(peakInd-2, self.minLeftBoundaryInd, -1):
                
                # Initialize range of data to check
                checkPeakBuffer = int((rightInd - leftInd)/4)
                xDataCut = xData[max(0, leftInd - checkPeakBuffer):rightInd + checkPeakBuffer]
                yDataCut = yData[max(0, leftInd - checkPeakBuffer):rightInd + checkPeakBuffer]
                
                # Draw a Linear Line Between the Points
                lineSlope = (yData[leftInd] - yData[rightInd])/(xData[leftInd] - xData[rightInd])
                slopeIntercept = yData[leftInd] - lineSlope*xData[leftInd]
                linearFit = lineSlope*xDataCut + slopeIntercept

                # Find the Number of Points Above the Tangent Line
                numWrongSideOfTangent = len(linearFit[linearFit - yDataCut > 0])
                
                # If a Tangent Line is Drawn Correctly, Return the Tangent Points' Indexes
                # if numWrongSideOfTangent == 0 and rightInd - leftInd > self.minPeakDuration:
                #     return (leftInd, rightInd)
                # Define a threshold for distinguishing good/bad lines
                maxBadPoints = int(len(linearFit)/10) # Minimum 1/6
                if numWrongSideOfTangent < maxBadPoints and rightInd - leftInd > self.minPeakDuration:
                    goodTangentInd[numWrongSideOfTangent].append((leftInd, rightInd))
                    
        # If Nothing Found, Try and Return a Semi-Optimal Tangent Position
        for goodInd in range(maxBadPointsTotal):
            if len(goodTangentInd[goodInd]) != 0:
                return max(goodTangentInd[goodInd], key=lambda tangentPair: tangentPair[1]-tangentPair[0])
        return None, None
    
    def findLineIntersectionPoint(self, leftLineParams, rightLineParams):
        xPoint = (rightLineParams[1] - leftLineParams[1])/(leftLineParams[0] - rightLineParams[0])
        yPoint = leftLineParams[0]*xPoint + leftLineParams[1]
        return xPoint, yPoint
    
    def pinpointExtrema(self, yData, xPointer, binarySearchMag, bufferSearch = 5):
        if yData[xPointer - bufferSearch]  < yData[xPointer + bufferSearch]:
            return xPointer
        elif yData[xPointer - bufferSearch] < yData[xPointer + bufferSearch]:
            return xPointer
    
    def calculateCurvature(self, xData, yData):
        # Calculate Derivatives
        dx_dt = np.gradient(xData); dx_dt2 = np.gradient(dx_dt); 
        dy_dt = np.gradient(yData); dy_dt2 = np.gradient(dy_dt);
        
        # Calculate Peak Shape parameters
        speed = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
        curvature = np.abs((dx_dt2 * dy_dt - dx_dt * dy_dt2)) / speed**3  # Units 1/Volts
        # Return the Curvature
        return curvature
        
    def extractFeatures_Pointwise(self, xData, baselineData, peakInd, chemicalName):
        # ----------------------- Derivative Analysis ----------------------- #   
        # Normalize the Data
        baselineData = baselineData/baselineData[peakInd]
        xData = xData/baselineData[peakInd]
        # Calculate the Signal Derivatives
        velocity = np.gradient(baselineData, xData, edge_order = 2)
        acceleration = np.gradient(velocity, xData, edge_order = 2)
        thirdDeriv = np.gradient(acceleration, xData, edge_order = 2)
        forthDeriv = np.gradient(thirdDeriv, xData, edge_order = 2)

        # Find the Velocity Extremas
        leftVelPeakInd = self.findNearbyMaximum(velocity, peakInd, binarySearchWindow = -1)
        rightVelPeakInd = self.findNearbyMinimum(velocity, peakInd, binarySearchWindow = 1)
        # ------------------------------------------------------------------- #
        
        peakHeight = baselineData[peakInd]
        plt.show()
        plt.plot(xData, baselineData, 'k', linewidth= 2)
        plt.plot(xData[peakInd], baselineData[peakInd], 'bo')
        plt.plot(xData, peakHeight*velocity/max(abs(velocity)), 'tab:blue')
        plt.plot(xData, peakHeight*acceleration/max(abs(acceleration)), 'tab:red')
        plt.plot(xData[[leftVelPeakInd, rightVelPeakInd]], (peakHeight*velocity/max(abs(velocity)))[[leftVelPeakInd, rightVelPeakInd]], 'o')
        plt.ylim([-peakHeight*1.1, peakHeight*1.1])
        plt.show()

        # ----------------------- Store Peak Features ----------------------- #
        peakFeatures = []
        for peakInd in range(leftVelPeakInd, rightVelPeakInd+1):
            peakFeatures.append([baselineData[peakInd], velocity[peakInd], acceleration[peakInd], thirdDeriv[peakInd], forthDeriv[peakInd]])
        return peakFeatures
        # ------------------------------------------------------------------- #
    
    def extractFeatures(self, xData, baselineData, peakInd, chemicalName):
        # Normalize the Data
        baselineData = baselineData/baselineData[peakInd]
        xData = xData/baselineData[peakInd]
        # ----------------------- Derivative Analysis ----------------------- #        
        # Calculate the Signal Derivatives
        velocity = np.gradient(baselineData, xData, edge_order = 2)
        acceleration = np.gradient(velocity, xData, edge_order = 2)
        thirdDeriv = np.gradient(acceleration, xData, edge_order = 2)
        forthDeriv = np.gradient(thirdDeriv, xData, edge_order = 2)

        # Find the Velocity Extremas
        leftVelPeakInd = self.findNearbyMaximum(velocity, peakInd, binarySearchWindow = -1)
        rightVelPeakInd = self.findNearbyMinimum(velocity, peakInd, binarySearchWindow = 1)

        # Find Third Derivative Extremas on Right
        thirdDerivRightMax_Temp = self.findNearbyMaximum(thirdDeriv, rightVelPeakInd, binarySearchWindow = 2)
        thirdDerivRightMax = self.findNearbyMaximum(thirdDeriv, rightVelPeakInd, binarySearchWindow = -2)
        thirdDerivRightMax = max([thirdDerivRightMax, thirdDerivRightMax_Temp], key = lambda ind: thirdDeriv[ind])
        thirdDerivRightMin = self.findNearbyMinimum(thirdDeriv, thirdDerivRightMax, binarySearchWindow = 2)
        
        # Find Third Derivative Extremas on Left: Min
        thirdDerivLeftMin = self.findNearbyMinimum(thirdDeriv, leftVelPeakInd, binarySearchWindow = -2)
        if abs(thirdDerivLeftMin - leftVelPeakInd) < 5:
            thirdDerivLeftMin = self.findNearbyMinimum(thirdDeriv, leftVelPeakInd, binarySearchWindow = 2)
        # Find Third Derivative Extremas on Left: Max
        thirdDerivLeftMax = self.findNearbyMaximum(thirdDeriv, thirdDerivLeftMin, binarySearchWindow = 3)
        if abs(thirdDerivLeftMin - thirdDerivLeftMax) < 5:
            thirdDerivLeftMax = self.findNearbyMaximum(thirdDeriv, thirdDerivLeftMin, binarySearchWindow = 5)

        # Find the Acceleration Extremas on Left: Max
        maxAccelLeftInd = self.findNearbyMaximum(acceleration, leftVelPeakInd, binarySearchWindow = -2)
        # fourthDerivStartLeftSlopeInd = self.findNearbyMaximum(forthDeriv, leftVelPeakInd, binarySearchWindow = -2)
        # fourthDerivStartLeftSlopeInd = self.findNearbyMinimum(forthDeriv, fourthDerivStartLeftSlopeInd, binarySearchWindow = -2)
        # Find the Acceleration Extremas on Left: Min
        minAccelLeftInd_Temp = self.findNearbyMinimum(acceleration, peakInd, binarySearchWindow = -1)
        minAccelLeftInd = self.findNearbyMinimum(acceleration, peakInd, binarySearchWindow = 1)
        minAccelLeftInd = min(minAccelLeftInd_Temp, minAccelLeftInd, key = lambda ind: acceleration[ind])

        # Find the Acceleration Extremas on Right: Max
        maxAccelRightInd = self.findNearbyMaximum(acceleration, thirdDerivRightMin, binarySearchWindow = -2)
        # Find the Drop in the Acceleration IF PRESENT (If Not, IT IS THE SAME AS maxAccelRightInd)
        minAccelRightInd = self.findNearbyMinimum(acceleration, thirdDerivRightMin, binarySearchWindow = 1)
        # If Accel Peak on the Right is a Saddle Point, Get the Boundaries
        if abs(minAccelRightInd - maxAccelRightInd) <= 5:
            maxAccelRightInd = self.findNearbyMinimum(forthDeriv, maxAccelRightInd, binarySearchWindow = -1)
            minAccelRightInd = self.findNearbyMaximum(forthDeriv, minAccelRightInd, binarySearchWindow = 1)
        # ------------------------------------------------------------------- #
        
        peakHeight = baselineData[peakInd]
        plt.show()
        plt.plot(xData, baselineData, 'k', linewidth= 2)
        plt.plot(xData[peakInd], baselineData[peakInd], 'bo')
        plt.plot(xData, peakHeight*velocity/max(abs(velocity)), 'tab:blue')
        plt.plot(xData, peakHeight*acceleration/max(abs(acceleration)), 'tab:red')
        plt.plot(xData, peakHeight*thirdDeriv/max(abs(thirdDeriv[maxAccelLeftInd:minAccelRightInd])), 'tab:brown')
        plt.plot(xData, peakHeight*forthDeriv/max(abs(forthDeriv[maxAccelLeftInd:minAccelRightInd])), 'tab:purple')
        plt.plot(xData[[leftVelPeakInd, rightVelPeakInd]], (peakHeight*velocity/max(abs(velocity)))[[leftVelPeakInd, rightVelPeakInd]], 'o')
        plt.plot(xData[[maxAccelRightInd, minAccelLeftInd, minAccelRightInd, maxAccelLeftInd]], (peakHeight*acceleration/max(abs(acceleration)))[[maxAccelRightInd, minAccelLeftInd, minAccelRightInd, maxAccelLeftInd]], 'o')
        plt.plot(xData[[thirdDerivLeftMax, thirdDerivLeftMin, thirdDerivRightMin, thirdDerivRightMax]], (peakHeight*thirdDeriv/max(abs(thirdDeriv[maxAccelLeftInd:minAccelRightInd])))[[thirdDerivLeftMax, thirdDerivLeftMin, thirdDerivRightMin, thirdDerivRightMax]], 'o')
        plt.ylim([-peakHeight*1.1, peakHeight*1.1])
        plt.show()
        
        # -------------------------- Time Features -------------------------- #   
        # Peak Durations
        peakDurationFull = xData[-1] - xData[0]
        peakRiseDuration = xData[peakInd] - xData[0]
        peakFallDuration = peakDurationFull - peakRiseDuration
        peakDurationRatio = peakRiseDuration/peakFallDuration
        
        # Velocity Intervals
        velInterval = xData[rightVelPeakInd] - xData[leftVelPeakInd]
        velRiseToPeak = xData[peakInd] - xData[leftVelPeakInd]
        velFallToPeak = xData[rightVelPeakInd] - xData[peakInd]
        
        # Acceleration Intervals
        minAccelToPeak = xData[peakInd] - xData[minAccelLeftInd]
        minAccelToVelLeft = xData[minAccelLeftInd] - xData[leftVelPeakInd]
        minAccelToVelRight = xData[rightVelPeakInd] - xData[minAccelLeftInd]
        leftAccelInterval = xData[minAccelLeftInd] - xData[maxAccelLeftInd]
        rightAccelInterval = xData[minAccelRightInd] - xData[maxAccelRightInd]

        # Third Deriv Intervals
        thirdDerivRightInterval = xData[thirdDerivRightMin] - xData[thirdDerivRightMax]
        thirdDerivLeftInterval = xData[thirdDerivLeftMax] - xData[thirdDerivLeftMin]
        # ------------------------------------------------------------------- #
        
        # -------------------------- Slope Features --------=---------------- #     
        # Get Full Slope Parameters
        upSlope_Full = np.polyfit(xData[maxAccelLeftInd:leftVelPeakInd], baselineData[maxAccelLeftInd:leftVelPeakInd], 1)
        downSlope_Full = np.polyfit(xData[thirdDerivRightMax:maxAccelRightInd], baselineData[thirdDerivRightMax:maxAccelRightInd], 1)
        # Get Slope Features
        upSlope = upSlope_Full[0]
        downSlope = downSlope_Full[0]
        # ------------------------------------------------------------------- #
        
        # ------------------------ Amplitude Features ----------------------- #  
        # Amplitudes Based on the Baseline Extremas
        peakHeight = baselineData[peakInd]
        # Amplitudes Based on the Velocity Extremas
        maxUpSlopeConc, maxDownSlopeConc = baselineData[[leftVelPeakInd, rightVelPeakInd]]
        maxUpSlopeVel, maxDownSlopeVel = velocity[[leftVelPeakInd, rightVelPeakInd]]
        # Amplitudes Based on the Acceleration Extremas
        maxAccelLeftConc, minAccelLeftConc, maxAccelRightConc, minAccelRightConc = baselineData[[maxAccelLeftInd, minAccelLeftInd, maxAccelRightInd, minAccelRightInd]]
        maxAccelLeftAccel, minAccelLeftAccel, maxAccelRightAccel, minAccelRightAccel = acceleration[[maxAccelLeftInd, minAccelLeftInd, maxAccelRightInd, minAccelRightInd]]

        # Velocity Amplitudes
        velDiffConc = maxUpSlopeConc - maxDownSlopeConc
        # Acceleration Amplitudes
        accelDiffRightConc = maxAccelRightConc - minAccelRightConc
        # ------------------------------------------------------------------- #

        # --------------------- Under the Curve Features -------------------- #        
        # Calculate the Area Under the Curve
        peakArea = scipy.integrate.simpson(baselineData, xData)
        leftArea = scipy.integrate.simpson(baselineData[0:peakInd], xData[0:peakInd])
        rightArea = peakArea - leftArea
        
        # General Areas
        velToVelArea = scipy.integrate.simpson(abs(baselineData[leftVelPeakInd:rightVelPeakInd]), xData[leftVelPeakInd:rightVelPeakInd])
        # ------------------------------------------------------------------- #
        
        # ----------------------- Peak Shape Features ----------------------- #        
        # Find Peak's Tent Features
        peakTentX, peakTentY = self.findLineIntersectionPoint(upSlope_Full, downSlope_Full)
        tentDeviationX = peakTentX - xData[peakInd]
        tentDeviationY = peakTentY - baselineData[peakInd]

        # Calculate the New Baseline of the Peak
        startBlinkX, _ = self.findLineIntersectionPoint([0, 0], upSlope_Full)
        endBlinkX, _ = self.findLineIntersectionPoint(downSlope_Full, [0, 0])
        
        # Calculate the New Baseline's Index
        # startBlinkInd = np.argmin(abs(xData - startBlinkX))
        # endBlinkInd = np.argmin(abs(xData - endBlinkX))
        
        blinkDuration_Final = endBlinkX - startBlinkX
        
        # Shape Parameters
        peakAverage = np.mean(baselineData[leftVelPeakInd:rightVelPeakInd])
        peakSTD = np.std(baselineData[leftVelPeakInd:rightVelPeakInd], ddof=1)
        peakEntropy = entropy(baselineData[leftVelPeakInd:rightVelPeakInd])
        peakSkew = skew(baselineData[leftVelPeakInd:rightVelPeakInd], bias=False)
        peakKurtosis = kurtosis(baselineData[leftVelPeakInd:rightVelPeakInd], fisher=True, bias = False)
        
        baselineDataFFT = fft(baselineData)[leftVelPeakInd:rightVelPeakInd]
        peakAverageFFT = abs(np.mean(baselineDataFFT))
        peakSTD_FFT = np.std(baselineDataFFT, ddof=1)
        peakEntropyFFT = entropy(abs(baselineDataFFT))
        
        peakHeightFFT = abs(baselineDataFFT[peakInd - leftVelPeakInd])
        leftVelHeightFFT = abs(baselineDataFFT[peakInd])
        rightVelHeightFFT = abs(baselineDataFFT[peakInd - leftVelPeakInd + rightVelPeakInd])
        
        curvature = self.calculateCurvature(xData, baselineData)
        peakCurvature = curvature[peakInd]
        leftVelCurvature = curvature[leftVelPeakInd]
        rightVelCurvature = curvature[rightVelPeakInd]
        # ------------------------------------------------------------------- #
        
        # -------------------------- Ratio Features ------------------------- #        
        # ------------------------------------------------------------------- #
        
        # ------------------------ Cull Bad Features ------------------------ #  
        if blinkDuration_Final > 2000:
            print("Bad blinkDuration_Final:", blinkDuration_Final)
            return []
        # ------------------------------------------------------------------- #

        # ----------------------- Store Peak Features ----------------------- #
        peakFeatures = []
        # Saving Features from Section: Time Features
        peakFeatures.extend([peakDurationFull, peakRiseDuration, peakFallDuration, peakDurationRatio])
        peakFeatures.extend([velInterval, velRiseToPeak, velFallToPeak])
        peakFeatures.extend([minAccelToPeak, minAccelToVelLeft, minAccelToVelRight, leftAccelInterval, rightAccelInterval])
        peakFeatures.extend([thirdDerivRightInterval, thirdDerivLeftInterval])
        
        # Saving Features from Section: Amplitude Features
        peakFeatures.extend([peakHeight])
        peakFeatures.extend([maxUpSlopeConc, maxDownSlopeConc, maxUpSlopeVel, maxDownSlopeVel])
        peakFeatures.extend([maxAccelLeftConc, maxAccelRightConc, minAccelRightConc])
        peakFeatures.extend([maxAccelLeftAccel, minAccelLeftAccel, maxAccelRightAccel, minAccelRightAccel])
        peakFeatures.extend([velDiffConc, accelDiffRightConc])
        
        # Saving Features from Section: Slope Features
        peakFeatures.extend([upSlope, downSlope])
        
        # Saving Features from Section: Under the Curve Features
        peakFeatures.extend([leftArea, rightArea, velToVelArea])
        
        # Saving Features from Section: Peak Shape Features
        peakFeatures.extend([peakTentX, peakTentY, tentDeviationX, tentDeviationY, blinkDuration_Final])
        peakFeatures.extend([peakAverage, peakSTD, peakSkew, peakKurtosis])

        # ------------------------------------------------------------------- #
        
        if False:
            plt.plot(xData, baselineData, 'k', linewidth= 2)
            plt.plot(xData[peakInd], baselineData[peakInd], 'bo')
            plt.plot(xData, peakHeight*velocity/max(abs(velocity)), 'tab:blue')
            plt.plot(xData, peakHeight*acceleration/max(abs(acceleration)), 'tab:red')
            plt.plot(xData, peakHeight*thirdDeriv/max(abs(thirdDeriv[maxAccelLeftInd:minAccelRightInd])), 'tab:brown')
            plt.plot(xData, peakHeight*forthDeriv/max(abs(forthDeriv[maxAccelLeftInd:minAccelRightInd])), 'tab:purple')

            plt.plot(xData[[leftVelPeakInd, rightVelPeakInd]], (peakHeight*velocity/max(abs(velocity)))[[leftVelPeakInd, rightVelPeakInd]], 'o')
            plt.plot(xData[[maxAccelRightInd, minAccelLeftInd, minAccelRightInd, maxAccelLeftInd]], (peakHeight*acceleration/max(abs(acceleration)))[[maxAccelRightInd, minAccelLeftInd, minAccelRightInd, maxAccelLeftInd]], 'o')
            plt.plot(xData[[thirdDerivLeftMax, thirdDerivLeftMin, thirdDerivRightMin, thirdDerivRightMax]], (peakHeight*thirdDeriv/max(abs(thirdDeriv[maxAccelLeftInd:minAccelRightInd])))[[thirdDerivLeftMax, thirdDerivLeftMin, thirdDerivRightMin, thirdDerivRightMax]], 'o')
        
        return [peakFeatures]
    
    def plot(self, xData, yData, baselineData, linearFit, peakInd, leftCutInd, rightCutInd, chemicalName = "Chemical"):
        plt.plot(xData, yData, 'k', linewidth=2)
        plt.plot(xData[peakInd], yData[peakInd], 'bo')
        plt.plot(xData[[leftCutInd,rightCutInd]], yData[[leftCutInd,rightCutInd]], 'ro')
        if len(linearFit) > 0:
            plt.plot(xData, linearFit, 'r', alpha=0.5)
        if len(baselineData) > 0:
            plt.plot(xData, baselineData, 'tab:brown', linewidth=1.5)
         # Add Figure Title and Labels
        plt.title(chemicalName + " Data")
        plt.xlabel("Time (Sec)")
        plt.ylabel("Concentration (uM)")
        # Display the Plot
        plt.show()


    
    
    