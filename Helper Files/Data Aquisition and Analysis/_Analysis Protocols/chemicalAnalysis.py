
# Basic Modules
import sys
import numpy as np
# Peak Detection
import scipy
import scipy.signal
# High/Low Pass Filters
from scipy.signal import butter
# Matlab Plotting API
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.stats.mstats import gmean
from scipy.signal import butter, lfilter 
from scipy.interpolate import UnivariateSpline
        
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
        
        self.lowPassCutoff = 0.007
        
        self.plotData = plotData
        
        self.startStimulus = startStimulus
        self.stimulusDuration = stimulusDuration + stimulusBuffer
    
    def analyzeData(self, xData, yData, chemicalName = ""):
        if not self.continueAnalysis:
            return []
        
        # ------------------------- Filter the Data ------------------------- #
        # Apply a Low Pass Filter
        self.samplingFreq = len(xData)/(xData[-1] - xData[0])
        yData = self.butterFilter(yData, self.lowPassCutoff, self.samplingFreq, order = 3, filterType = 'low')
        # ------------------------------------------------------------------- #
        
        # ------------------- Find and Remove the Baseline ------------------ #
        # Find Peaks in the Data
        chemicalPeakInd = self.findPeak(xData, yData)
        # Return None if No Peak Found
        if chemicalPeakInd == None:
            print("No Peak Found in " + chemicalName + " Data")
            return [0]*57 if len(self.glucoseFeatures) == 0 else [0]*len(self.glucoseFeatures[0])
        # ------------------------------------------------------------------- #
        self.plot(xData, yData, yData, yData, chemicalPeakInd, chemicalPeakInd, chemicalPeakInd, chemicalName)
        # ------------------- Find and Remove the Baseline ------------------ #
        # Get Baseline from Best Linear Fit
        leftCutInd, rightCutInd = self.findLinearBaseline(xData, yData, chemicalPeakInd)
        if None in [leftCutInd, rightCutInd] or abs(leftCutInd - rightCutInd) < 10:
            print("No Baseline Found in " + chemicalName + " Data")
            self.continueAnalysis = False
            return []
        
        # Fit Lines to Ends of Graph
        lineSlope, slopeIntercept = np.polyfit(xData[[leftCutInd, rightCutInd]], yData[[leftCutInd, rightCutInd]], 1)
        linearFit = lineSlope*xData + slopeIntercept
            
        # Piece Together yData's Baseline
        baseline = np.concatenate((yData[0:leftCutInd+1], linearFit[leftCutInd+1: rightCutInd], yData[rightCutInd:len(yData)]))
        # Find yData After Baseline Subtraction
        baselineData = yData - baseline
        # ------------------------------------------------------------------- #
        
        # -------------------- Extract Features from Peak ------------------- #
        # Adjust the Peak's Ind After Baseline Subtraction
        chemicalPeakInd = self.findNearbyMaximum(baselineData, chemicalPeakInd, binarySearchWindow = 2)
        chemicalPeakInd = self.findNearbyMaximum(baselineData, chemicalPeakInd, binarySearchWindow = -2)
        
        # Establish New Peak Boundaries
        leftBaseInd = self.findNearbyMinimum(baselineData, chemicalPeakInd, binarySearchWindow = -1)
        rightBaseInd = self.findNearbyMinimum(baselineData, chemicalPeakInd, binarySearchWindow = 1)

        # Extract the Features from the Data
        peakFeatures = self.extractFeatures(xData[leftBaseInd:rightBaseInd+1] - xData[leftBaseInd], baselineData[leftBaseInd:rightBaseInd+1], chemicalPeakInd-leftBaseInd, chemicalName)
        # ------------------------------------------------------------------- #
        
        # -------------------------- Plot the Data -------------------------- #
       # if self.plotData:
       #     self.plot(xData, yData, baselineData, linearFit, chemicalPeakInd, leftCutInd, rightCutInd, chemicalName)
        # ------------------------------------------------------------------- #
        
        if len(peakFeatures) == 0:
            print("No Features Found in " + chemicalName + " Data")
            self.continueAnalysis = False
        return peakFeatures
    
    def analyzeChemicals(self, timePoints, glucose, lactate, uricAcid, label):
        # Assert All Chemical Data is Well-Formed 
        self.continueAnalysis = True
        
        # Analyze Each Chemical
        glucoseFeatures = self.analyzeData(timePoints, glucose, chemicalName = "Glucose")
        self.continueAnalysis = True
        lactateFeatures = self.analyzeData(timePoints, lactate, chemicalName = "Lactate")
        self.continueAnalysis = True
        uricAcidFeatures = self.analyzeData(timePoints, uricAcid, chemicalName = "Uric Acid")
        self.continueAnalysis = True
        
        # If Features Found for ALL Chemicals
        if self.continueAnalysis:
            # Keep Track of the Features
            if len(glucoseFeatures) > 0:
                self.glucoseFeatures.append(glucoseFeatures)
                self.featureLabelsGlucose.append(label)
            if len(lactateFeatures) > 0:
                self.lactateFeatures.append(lactateFeatures)
                self.featureLabelsLactate.append(label)
            if len(uricAcidFeatures) > 0:
                self.uricAcidFeatures.append(uricAcidFeatures)
                self.featureLabelsUricAcid.append(label)

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    
    def findPeak(self, xData, yData, ignoredBoundaryPoints = 10):
        # Find All Peaks in the Data
        peakInfo = scipy.signal.find_peaks(yData, prominence=.0001, width=30, distance = 20)
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
            return xPointer - 1 + np.argmin(data[max(0,xPointer-1):min(xPointer+2, len(data))]) 
        
        maxHeightPointer = xPointer
        maxHeight = data[xPointer]; searchDirection = binarySearchWindow//abs(binarySearchWindow)
        # Binary Search Data to Find the Minimum (Skip Over Minor Fluctuations)
        for dataPointer in range(max(xPointer, 0), max(0, min(xPointer + searchDirection*maxPointsSearch, len(data))), binarySearchWindow):
            # If the Next Point is Greater Than the Previous, Take a Step Back
            if data[dataPointer] >= maxHeight and xPointer != dataPointer:
                return self.findNearbyMinimum(data, dataPointer - binarySearchWindow, round(binarySearchWindow/8), maxPointsSearch - searchDirection*(abs(dataPointer - binarySearchWindow)) - xPointer)
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
            return xPointer - 1 + np.argmax(data[max(0,xPointer-1):xPointer+2]) 
        
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
        maxBadPoints = int(len(xData)/5)
        ignoreBoundaryPoints = int(self.startStimulus*self.samplingFreq/2)
        
        # Divide the yData into Two Groups: Left and Right
        leftData = yData[ignoreBoundaryPoints:peakInd]
        rightData = yData[peakInd+1:len(yData)]
        # Store Possibly Good Tangent Indexes
        goodTangentInd = [[] for _ in range(maxBadPoints+1)]
                                
        # For Each Index Pair on the Left and Right of the Peak
        for rightInd, rightPoint in enumerate(rightData):
            rightInd = rightInd + peakInd + 1
            
            for leftInd, leftPoint in enumerate(leftData):
                leftInd = leftInd + ignoreBoundaryPoints
                
                # Draw a Linear Line Between the Points
                lineSlope = (yData[leftInd] - yData[rightInd])/(xData[leftInd] - xData[rightInd])
                slopeIntercept = yData[leftInd] - lineSlope*xData[leftInd]
                linearFit = lineSlope*xData + slopeIntercept

                # # Find the Number of Points Above the Tangent Line
                numWrongSideOfTangent = len(linearFit[(linearFit - yData) > 0])
                        
                # If a Tangent Line is Drawn Correctly, Return the Tangent Points' Indexes
                if numWrongSideOfTangent <= maxBadPoints:
                    goodTangentInd[numWrongSideOfTangent].append((leftInd, rightInd))
                    
        # If Nothing Found, Try and Return a Semi-Optimal Tangent Position
        for goodInd in range(maxBadPoints):
            if len(goodTangentInd[goodInd]) != 0:
                return min(goodTangentInd[goodInd], key=lambda tangentPair: tangentPair[1]-tangentPair[0])
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
    
    def extractFeatures(self, xData, baselineData, peakInd, chemicalName):
        # ----------------------- Derivative Analysis ----------------------- #        
        # Calculate the Signal Derivatives
        velocity = np.gradient(baselineData, xData)
        acceleration = np.gradient(velocity, xData)
        thirdDeriv = np.gradient(acceleration, xData)
        forthDeriv = np.gradient(thirdDeriv, xData)

        # Find the Velocity Extremas
        leftVelPeakInd = self.findNearbyMaximum(velocity, peakInd, binarySearchWindow = -2)
        rightVelPeakInd = self.findNearbyMinimum(velocity, peakInd, binarySearchWindow = 1)

        # Find Third Derivative Extremas
        thirdDerivRightMax_Temp = self.findNearbyMaximum(thirdDeriv, rightVelPeakInd, binarySearchWindow = 2)
        thirdDerivRightMax = self.findNearbyMaximum(thirdDeriv, rightVelPeakInd, binarySearchWindow = -2)
        thirdDerivRightMax = max([thirdDerivRightMax, thirdDerivRightMax_Temp], key = lambda ind: thirdDeriv[ind])
        thirdDerivRightMin = self.findNearbyMinimum(thirdDeriv, thirdDerivRightMax, binarySearchWindow = 25)
        
        # Find the Acceleration Extremas
        maxAccelLeftInd = self.findNearbyMaximum(acceleration, leftVelPeakInd, binarySearchWindow = -2)
        # Find the First Maximum in the Acceleration After the Peak (Two Ways)
        maxAccelRightInd_Temp = self.findNearbyMaximum(acceleration, thirdDerivRightMin, binarySearchWindow = -2)
        maxAccelRightInd = thirdDerivRightMax + np.argmin(abs(thirdDeriv[thirdDerivRightMax: thirdDerivRightMin]))
        # If They are Similar, Trust the findMaximum Way More
        if abs(maxAccelRightInd_Temp - maxAccelRightInd) < 5:
            maxAccelRightInd = maxAccelRightInd_Temp
        # Find the Drop in the Acceleration IF PRESENT (If Not, IT IS THE SAME AS maxAccelRightInd)
        minAccelRightInd = self.findNearbyMinimum(acceleration, thirdDerivRightMin, binarySearchWindow = 1)
        # If Accel Peak on the Right is a Saddle Point, Get the Boundaries
        if abs(minAccelRightInd - maxAccelRightInd) <= 5:
            maxAccelRightInd = self.findNearbyMinimum(forthDeriv, maxAccelRightInd, binarySearchWindow = -1)
            minAccelRightInd = self.findNearbyMaximum(forthDeriv, minAccelRightInd, binarySearchWindow = 1)
        # ------------------------------------------------------------------- #
        
        # ------------------------ Cull Bad Features ------------------------ #  
        # Cull Noisy Data -> Increase Sampling Frequency
        accelInds = scipy.signal.find_peaks(abs(acceleration), prominence=.00001)[0]
        if len(accelInds) > 6:
            print("Data Too Noisy")
            return []
        
        # If Any Index is Near the Border, the Data Could be MisCut
        if len(xData) - rightVelPeakInd < 50 or len(xData) - minAccelRightInd < 50:
            return []
        # ------------------------------------------------------------------- #

        # -------------------------- Time Features -------------------------- #   
        # Peak Durations
        peakDuration = xData[-1] - xData[0]
        peakRiseDuration = xData[peakInd] - xData[0]
        peakFallDuration = peakDuration - peakRiseDuration
        
        # Velocity Intervals
        velInterval = xData[rightVelPeakInd] - xData[leftVelPeakInd]
        velRiseToPeak = xData[peakInd] - xData[leftVelPeakInd]
        velFallToPeak = xData[rightVelPeakInd] - xData[peakInd]
        # Acceleration Intervals
        accelInterval1 = xData[peakInd] - xData[maxAccelLeftInd]
        accelInterval2 = xData[maxAccelRightInd] - xData[peakInd]
        accelInterval3 = xData[minAccelRightInd] - xData[maxAccelRightInd]
        accelInterval4 = xData[minAccelRightInd] - xData[maxAccelLeftInd]
        accelInterval5 = xData[maxAccelLeftInd] - xData[maxAccelLeftInd]
        # Third Deriv Intervals
        thirdDerivInterval = xData[thirdDerivRightMin] - xData[thirdDerivRightMax]
        
        # Mixed Intervals
        thirdDerivAccelInterval1 = xData[rightVelPeakInd] - xData[thirdDerivRightMax]
        accelToVelLeft = xData[leftVelPeakInd] - xData[maxAccelLeftInd]
        accelToVelRight = xData[rightVelPeakInd] - xData[maxAccelRightInd]

        # Time Ratios
        peakDurationRatio = peakRiseDuration/peakFallDuration
        # ------------------------------------------------------------------- #
        
        # -------------------------- Slope Features --------=---------------- #     
        # Peak Slope Features
        upSlope_Full = np.polyfit(xData[maxAccelLeftInd:leftVelPeakInd], baselineData[maxAccelLeftInd:leftVelPeakInd], 1)
        downSlope_Full = np.polyfit(xData[thirdDerivRightMax:maxAccelRightInd], baselineData[thirdDerivRightMax:maxAccelRightInd], 1)
        upSlope = upSlope_Full[0]; downSlope = downSlope_Full[0]
        
        # Velocity Slope
        velSlope = np.polyfit(xData[leftVelPeakInd:rightVelPeakInd], baselineData[leftVelPeakInd:rightVelPeakInd], 1)[0]
        # Acceleration Slope Features
        accelEndStimulusSlope = np.polyfit(xData[thirdDerivRightMax:thirdDerivRightMin], baselineData[thirdDerivRightMax:thirdDerivRightMin], 1)[0]
        # Third Derivative Slope Features
        thirdDerivEndStimulusSlope = np.polyfit(xData[maxAccelRightInd:minAccelRightInd], baselineData[maxAccelRightInd:minAccelRightInd], 1)[0]
        # ------------------------------------------------------------------- #
        
        # ------------------------ Amplitude Features ----------------------- #  
        # Amplitudes Based on the Baseline Extremas
        peakAmp = baselineData[peakInd]
        velAmp = velocity[peakInd]
        accelAmp = acceleration[peakInd]
        # Amplitudes Based on the Velocity Extremas
        maxUpSlopeConc, maxDownSlopeConc = baselineData[[leftVelPeakInd, rightVelPeakInd]]
        maxUpSlopeVel, maxDownSlopeVel = velocity[[leftVelPeakInd, rightVelPeakInd]]
        maxUpSlopeAccel, maxDownSlopeAccel = acceleration[[leftVelPeakInd, rightVelPeakInd]]
        # Amplitudes Based on the Acceleration Extremas
        maxAccelLeftConc, maxAccelRightConc, minAccelRightConc = baselineData[[maxAccelLeftInd, maxAccelRightInd, minAccelRightInd]]
        maxAccelLeftVel, maxAccelRightVel, minAccelRightVel = velocity[[maxAccelLeftInd, maxAccelRightInd, minAccelRightInd]]
        maxAccelLeftAccel, maxAccelRightAccel, minAccelRightAccel = acceleration[[maxAccelLeftInd, maxAccelRightInd, minAccelRightInd]]
        # Amplitudes Based on the Velocity Extremas
        thirdDerivRightMaxConc, thirdDerivRightMinConc = baselineData[[thirdDerivRightMax, thirdDerivRightMin]]
        thirdDerivRightMaxVel, thirdDerivRightMinVel = velocity[[thirdDerivRightMax, thirdDerivRightMin]]
        thirdDerivRightVel, thirdDerivRightVel = acceleration[[thirdDerivRightMax, thirdDerivRightMin]]
        
        # Velocity Amplitudes
        velDiffConc = maxUpSlopeConc - maxDownSlopeConc
        velDiffVel = maxUpSlopeVel - maxDownSlopeVel
        velDiffAccel = maxUpSlopeAccel - maxDownSlopeAccel
        
        # Find Peak's Tent Features
        peakTentX, peakTentY = self.findLineIntersectionPoint(upSlope_Full, downSlope_Full)
        tentDeviationX = peakTentX - xData[peakInd]
        tentDeviationY = peakTentY - baselineData[peakInd]
        # ------------------------------------------------------------------- #

        # --------------------- Under the Curve Features -------------------- #        
        # Calculate the Area Under the Curve
        peakArea = scipy.integrate.simpson(baselineData, xData)
        leftArea = scipy.integrate.simpson(baselineData[0:peakInd], xData[0:peakInd])
        rightArea = peakArea - leftArea
        
        # General Areas
        velToVelArea = scipy.integrate.simpson(baselineData[leftVelPeakInd:rightVelPeakInd], xData[leftVelPeakInd:rightVelPeakInd])

        # Average of the Pulse
        geometricMean = gmean(baselineData)
        peakAverage = np.mean(baselineData)
        # ------------------------------------------------------------------- #
        
        # -------------------------- Ratio Features ------------------------- #        
        # ------------------------------------------------------------------- #
        
        # ----------------------- Store Peak Features ----------------------- #
        peakFeatures = []
        # Saving Features from Section: Time Features
        peakFeatures.extend([peakDuration, peakRiseDuration, peakFallDuration])
        peakFeatures.extend([velInterval, velRiseToPeak, velFallToPeak, peakDurationRatio])
        peakFeatures.extend([accelInterval1, accelInterval2, accelInterval3, accelInterval4, accelInterval5])
        peakFeatures.extend([thirdDerivInterval, thirdDerivAccelInterval1, accelToVelLeft, accelToVelRight])
        # Saving Features from Section: Slope Features
        peakFeatures.extend([upSlope, downSlope, velSlope, accelEndStimulusSlope, thirdDerivEndStimulusSlope])
        # Saving Features from Section: Assmplitude Features
        peakFeatures.extend([peakAmp, velAmp, accelAmp])
        peakFeatures.extend([maxUpSlopeConc, maxDownSlopeConc, maxUpSlopeVel, maxDownSlopeVel, maxUpSlopeAccel, maxDownSlopeAccel])
        peakFeatures.extend([maxAccelLeftConc, maxAccelRightConc, minAccelRightConc, maxAccelLeftVel, maxAccelRightVel, minAccelRightVel, maxAccelLeftAccel, maxAccelRightAccel, minAccelRightAccel])
        peakFeatures.extend([thirdDerivRightMaxConc, thirdDerivRightMinConc, thirdDerivRightMaxVel, thirdDerivRightMinVel, thirdDerivRightVel, thirdDerivRightVel])
        peakFeatures.extend([velDiffConc, velDiffVel, velDiffAccel])
        peakFeatures.extend([peakTentX, peakTentY, tentDeviationX, tentDeviationY])
        # Saving Features from Section: Under the Curve Features
        peakFeatures.extend([peakArea, leftArea, rightArea, velToVelArea])
        peakFeatures.extend([geometricMean, peakAverage])
        # ------------------------------------------------------------------- #
        
        if True:
            plt.plot(xData, baselineData, 'k', linewidth= 2)
            plt.plot(xData[peakInd], baselineData[peakInd], 'bo')
            plt.plot(xData, peakAmp*velocity/max(abs(velocity)), 'tab:blue')
            plt.plot(xData, peakAmp*acceleration/max(abs(acceleration)), 'tab:red')
            plt.plot(xData, peakAmp*thirdDeriv/max(abs(thirdDeriv[maxAccelLeftInd:minAccelRightInd])), 'tab:brown')
            #plt.plot(xData, peakAmp*forthDeriv/max(abs(forthDeriv[maxAccelLeftInd:minAccelRightInd])), 'tab:purple')

            plt.plot(xData[[leftVelPeakInd, rightVelPeakInd]], (peakAmp*velocity/max(abs(velocity)))[[leftVelPeakInd, rightVelPeakInd]], 'o')
            plt.plot(xData[[maxAccelRightInd, minAccelRightInd, maxAccelLeftInd]], (peakAmp*acceleration/max(abs(acceleration)))[[maxAccelRightInd, minAccelRightInd, maxAccelLeftInd]], 'o')
            plt.plot(xData[[thirdDerivRightMin, thirdDerivRightMax]], (peakAmp*thirdDeriv/max(abs(thirdDeriv[maxAccelLeftInd:minAccelRightInd])))[[thirdDerivRightMin, thirdDerivRightMax]], 'o')

            velInds = scipy.signal.find_peaks(abs(velocity), prominence=.00001)[0]
            plt.title(chemicalName + " VelInds: " + str(len(velInds)) + " AccelInds: " + str(len(accelInds)))
            plt.ylim([-peakAmp*1.1, peakAmp*1.1])
            plt.show()
        
        return peakFeatures
    
    def plot(self, xData, yData, baselineData, linearFit, peakInd, leftCutInd, rightCutInd, chemicalName = "Chemical"):
        plt.plot(xData, yData, 'k', linewidth=2)
        plt.plot(xData[peakInd], yData[peakInd], 'bo')
        plt.plot(xData[[leftCutInd,rightCutInd]], yData[[leftCutInd,rightCutInd]], 'ro')
        plt.plot(xData, linearFit, 'r', alpha=0.5)
        plt.plot(xData, baselineData, 'tab:brown', linewidth=1.5)
         # Add Figure Title and Labels
        plt.title(chemicalName + " Data")
        plt.xlabel("Time (Sec)")
        plt.ylabel("Concentration (uM)")
        # Display the Plot
        plt.show()


    
    
    