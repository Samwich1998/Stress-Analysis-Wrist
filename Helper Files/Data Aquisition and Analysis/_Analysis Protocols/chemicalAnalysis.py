
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

from scipy.signal import butter, lfilter 
from scipy.interpolate import UnivariateSpline
        
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class signalProcessing:
    
    def __init__(self, startStimulus, stimulusDuration, plotData = False):
        self.featureList = []
        
        self.lowPassCutoff = 0.005
        
        self.plotData = plotData
        
        self.startStimulus = startStimulus
        self.stimulusDuration = stimulusDuration
        

    def analyzeGlucose(self, xData, yData, stimulusBuffer):
        # ------------------------- Filter the Data ------------------------- #
        # Apply a Low Pass Filter
        samplingFreq = len(xData)/(xData[-1] - xData[0])
        yData = self.butterFilter(yData, self.lowPassCutoff, samplingFreq, order = 3, filterType = 'low')
        # ------------------------------------------------------------------- #
        
        # ------------------- Find and Remove the Baseline ------------------ #
        # Find Peaks in the Data
        chemicalPeakInd = self.findPeak(xData, yData, stimulusBuffer)
        # Return None if No Peak Found
        if chemicalPeakInd == None:
            return None
        # ------------------------------------------------------------------- #
        
        # ------------------- Find and Remove the Baseline ------------------ #
        # Get Baseline from Best Linear Fit
        leftCutInd, rightCutInd = self.findLinearBaseline(xData, yData, chemicalPeakInd)
        
        # Fit Lines to Ends of Graph
        lineSlope, slopeIntercept = np.polyfit(xData[[leftCutInd, rightCutInd]], yData[[leftCutInd, rightCutInd]], 1)
        linearFit = lineSlope*xData + slopeIntercept
            
        # Piece Together yData's Baseline
        baseline = np.concatenate((yData[0:leftCutInd+1], linearFit[leftCutInd+1: rightCutInd], yData[rightCutInd:len(yData)]))
        # Find yData After Baseline Subtraction
        baselineyData = yData - baseline
        # ------------------------------------------------------------------- #
        
        # -------------------- Extract Features from Peak ------------------- #
        # Adjust the Peak's Ind After Baseline Subtraction
        chemicalPeakInd = self.findNearbyMaximum(baselineyData, chemicalPeakInd, binarySearchWindow = 2)
        chemicalPeakInd = self.findNearbyMaximum(baselineyData, chemicalPeakInd, binarySearchWindow = -2)

        # Extract the Features from the Data
        self.extractFeatures(xData, baselineyData, chemicalPeakInd, leftCutInd, rightCutInd)
        # ------------------------------------------------------------------- #
        
        # -------------------------- Plot the Data -------------------------- #
        if self.plotData:
            self.plot(xData, yData, baselineyData, linearFit, chemicalPeakInd, leftCutInd, rightCutInd)
        # ------------------------------------------------------------------- #
        return 1
    
    def analyzeLactate(self, xData, yData):
      #  plt.plot(xData, yData, 'k', linewidth=2)
        return 1
    
    def analyzeUricAcid(self, xData, yData):
      #  plt.plot(xData, yData, 'k', linewidth=2)
        return 1
            
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    
    def findPeak(self, xData, yData, stimulusBuffer, ignoredBoundaryPoints = 10):
        # Find All Peaks in the Data
        peakInfo = scipy.signal.find_peaks(yData, prominence=.01, width=30, distance = 20)
        # Extract the Peak Information
        peakProminences = peakInfo[1]['prominences']
        peakIndices = peakInfo[0]
        
        # Remove Peaks Nearby Boundaries
        allProminences = peakProminences[np.logical_and(peakIndices < len(xData) - ignoredBoundaryPoints, peakIndices >= ignoredBoundaryPoints)]
        peakIndices = peakIndices[np.logical_and(peakIndices < len(xData) - ignoredBoundaryPoints, peakIndices >= ignoredBoundaryPoints)]
        # Seperate Out the Stimulus Window
        allProminences = allProminences[self.startStimulus < xData[peakIndices]]
        peakIndices = peakIndices[self.startStimulus < xData[peakIndices]]
        allProminences = allProminences[self.startStimulus + self.stimulusDuration + stimulusBuffer > xData[peakIndices]]
        peakIndices = peakIndices[self.startStimulus + self.stimulusDuration + stimulusBuffer > xData[peakIndices]]

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
        print(normal_cutoff)
        sos = butter(order, normal_cutoff, btype = filterType, analog = False, output='sos')
        return sos
    
    def butterFilter(self, data, cutoffFreq, samplingFreq, order = 3, filterType = 'band'):
        sos = self.butterParams(cutoffFreq, samplingFreq, order, filterType)
        return scipy.signal.sosfiltfilt(sos, data)
    
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
            if data[dataPointer] > maxHeight:
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
            if data[dataPointer] < minHeight:
                return self.findNearbyMaximum(data, dataPointer - binarySearchWindow, round(binarySearchWindow/2), maxPointsSearch - searchDirection*(abs(dataPointer - binarySearchWindow)) - xPointer)
            # Else, Continue Searching
            else:
                minHeightPointer = dataPointer
                minHeight = data[dataPointer]

        # If Your Binary Search is Too Large, Reduce it
        return self.findNearbyMaximum(data, minHeightPointer, round(binarySearchWindow/2), maxPointsSearch-1)
    
    
    def findLinearBaseline(self, xData, yData, peakInd, maxBadPoints = 4):
        # Divide the yData into Two Groups: Left and Right
        leftData = yData[0:peakInd]
        rightData = yData[peakInd+1:len(yData)]
        # Store Possibly Good Tangent Indexes
        goodTangentInd = [[] for _ in range(maxBadPoints+1)]
                                
        # For Each Index Pair on the Left and Right of the Peak
        for rightInd, rightPoint in enumerate(rightData):
            rightInd = rightInd + peakInd + 1
            
            for leftInd, leftPoint in enumerate(leftData):
                
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
    
    def extractFeatures(self, xData, baselineData, peakInd, leftCutInd, rightCutInd):
        # ----------------------- Analyze Derivatives ----------------------- #
        # Calculate the Signal Derivatives
        velocity = np.gradient(baselineData, xData)
        acceleration = np.gradient(velocity, xData)
        
        # Find Velocity Points
        maxVelRise = self.findNearbyMaximum(velocity, peakInd, binarySearchWindow = -2)
        minVelFall = self.findNearbyMinimum(velocity, peakInd, binarySearchWindow = 2)
        # FInd Acceleration Points
        maxAccelRight = self.findNearbyMaximum(acceleration, peakInd, binarySearchWindow = 2)
        minAccelRight = self.findNearbyMinimum(acceleration, maxAccelRight, binarySearchWindow = 2)
        minAccelLeft = self.findNearbyMinimum(acceleration, peakInd, binarySearchWindow = -2)
        maxAccelLeft = self.findNearbyMaximum(acceleration, minAccelLeft, binarySearchWindow = -2)
        # ------------------------------------------------------------------- #
        
        
        # Find the Peak's Amplitude
        peakAmp = baselineData[peakInd]
        velAmp = velocity[peakInd]
        accelAmp = acceleration[peakInd]
        
        # Find Peak's Tent Features
        # peakTentX, peakTentY = self.findLineIntersectionPoint(startBlinkLineParams, endBlinkLineparams1)
        # tentDeviationX = peakTentX - xData[peakInd]
        # tentDeviationY = peakTentY - baselineData[peakInd]
        
        
        
        
        
        plt.plot(xData, baselineData)
        plt.plot(xData[peakInd], baselineData[peakInd], 'bo')
        plt.plot(xData, peakAmp*velocity/max(velocity))
        plt.plot(xData, peakAmp*acceleration/max(acceleration))
        
        plt.plot(xData[[maxVelRise, minVelFall]], (peakAmp*velocity/max(velocity))[[maxVelRise, minVelFall]], 'o')
        plt.plot(xData[[maxAccelRight, minAccelRight, minAccelLeft, maxAccelLeft]], (peakAmp*acceleration/max(acceleration))[[maxAccelRight, minAccelRight, minAccelLeft, maxAccelLeft]], 'o')

        plt.show()
    
    def plot(self, xData, yData, baselineyData, linearFit, peakInd, leftCutInd, rightCutInd):
        plt.plot(xData, yData, 'k', linewidth=2)
        plt.plot(xData[peakInd], yData[peakInd], 'bo')
        plt.plot(xData[[leftCutInd,rightCutInd]], yData[[leftCutInd,rightCutInd]], 'ro')
        plt.plot(xData, linearFit, 'r', alpha=0.5)
        plt.plot(xData, baselineyData, 'tab:brown', linewidth=1.5)
         # Add Figure Title and Labels
        plt.title("Glucose Data")
        plt.xlabel("Time (Sec)")
        plt.ylabel("Concentration (uM)")
        # Display the Plot
        plt.show()


    
    
    