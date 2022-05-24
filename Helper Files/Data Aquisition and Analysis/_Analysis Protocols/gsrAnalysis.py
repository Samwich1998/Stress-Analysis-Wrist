
# Basic Modules
import sys
import math
import numpy as np
# Peak Detection Modules
import scipy
import scipy.signal
# Data Filtering Modules
from scipy.signal import butter
from scipy.signal import savgol_filter
# Matlab Plotting Modules
import matplotlib as mpl
import matplotlib.pyplot as plt
# Gaussian Decomposition
from lmfit import Model
from sklearn.metrics import r2_score
# Feature Extraction Modules
from scipy.stats import skew
from scipy.stats import entropy
from scipy.stats import kurtosis
from scipy.fft import fft, ifft
from scipy.interpolate import UnivariateSpline

from lmfit.models import GaussianModel
from lmfit.models import SkewedGaussianModel, SkewedVoigtModel

# Import Files
import _filteringProtocols as filteringMethods # Import Files with Filtering Methods

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class plot:
    
    def __init__(self):
        self.sectionColors = ['red','orange', 'blue','green', 'black']
    
    def plotData(self, xData, yData, title, ax = None, axisLimits = [], peakSize = 3, lineWidth = 2, lineColor = "tab:blue"):
        # Create Figure
        showFig = False
        if ax == None:
            plt.figure()
            ax = plt.gca()
            showFig = True
        # Plot the Data
        ax.plot(xData, yData, linewidth = lineWidth, color = lineColor)
        # Add Axis Labels and Figure Title
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Current (uAmps)")
        ax.set_title(title)
        # Change Axis Limits If Given
        if axisLimits:
            ax.set_xlim(axisLimits)
        # Increase DPI (Clarity); Will Slow Down Plotting
        mpl.rcParams['figure.dpi'] = 300
        # Show the Plot
        if showFig:
            plt.show()
        
# ---------------------------------------------------------------------------#
# ---------------------------------------------------------------------------#

class signalProcessing:
    
    def __init__(self, stimulusTimes):
        # General Parameters
        self.samplingFreq = None
        self.lowPassCutoff = 0.01
        
        # Specify the Stimulus Start/End Times
        self.endStimulusTime = stimulusTimes[1]
        self.startStimulusTime = stimulusTimes[0]
        self.stimulusBufferInd = 500
        
        # Define the Class with all the Filtering Methods
        self.filteringMethods = filteringMethods.filteringMethods()
    
    def analyzeGSR(self, xData, yData):
        
        # ------------------------- Filter the Data ------------------------ #
        # Apply a Low Pass Filter
        self.samplingFreq = len(xData)/(xData[-1] - xData[0])
        yData = self.filteringMethods.savgolFilter.savgolFilter(yData, 21, 2)
        yData = self.filteringMethods.bandPassFilter.butterFilter(yData, self.lowPassCutoff, self.samplingFreq, order = 4, filterType = 'low')
        yData = self.filteringMethods.savgolFilter.savgolFilter(yData, 21, 2)
        # ------------------------------------------------------------------ #
        
        # ------------------ Find and Remove the Baseline ------------------ #
        # Find the Stimulus Start/Stop Ind
        startStimulusInd = np.argmin(abs(xData - self.startStimulusTime))
        endStimulusInd = np.argmin(abs(xData - self.endStimulusTime))
        
        # Calculate the Baseline Using the Rest Data
        baselineDataParams = np.polyfit(xData[0:int(startStimulusInd/2)], yData[0:int(startStimulusInd/2)], 1)
        baselineFit = baselineDataParams[0]*xData + baselineDataParams[1]
        
        # Remove the Baseline Data
        yData -= baselineFit
        # ------------------------------------------------------------------ #

        # ---------------------- Analyze the GSR Peaks --------------------- #
        # Find the GSR Peak
        peakInd = np.argmax(yData[startStimulusInd:])
        
        # Find All Peaks in the Data
        peakInfo = scipy.signal.find_peaks(yData, prominence=10E-8, width = 20)
        # Extract the Peak Information
        peakProminences = peakInfo[1]['prominences'][peakInfo[0] > startStimulusInd]
        # peakIndices = peakInfo[0][peakInfo[0] > startStimulusInd]
        # plt.plot(xData, yData/max(yData))
        # plt.plot(xData[peakIndices], yData[peakIndices]/max(yData), 'o', markersize = 5)
        # ------------------------------------------------------------------ #
        
        # ----------------------- Feature Extraction ----------------------- #
        # Extract Peak Features
        numPeaks = len(peakProminences)
        sumPromineces = np.sum(peakProminences)
        bestprominence = np.max(peakProminences)
        prominenceRatio = bestprominence/sumPromineces
        
        # General Features
        maxHeight = yData[peakInd]
        riseTime = xData[peakInd] - self.startStimulusTime
        halfAmpRecoveryInd = peakInd + np.argmin(abs(yData[peakInd:] - maxHeight/2))
        recoveryTime = xData[halfAmpRecoveryInd] - xData[peakInd]
        
        # Extract Mean of the Data
        meanSignal = np.mean(yData)
        meanSignalRest = np.mean(yData[0:startStimulusInd])
        meanSignalStress = np.mean(yData[startStimulusInd:endStimulusInd])
        meanSignalRecovery = np.mean(yData[endStimulusInd:])
        meanStressIncrease = meanSignalStress - meanSignalRest
        
        # Extract Slope Features
        relaxationSlope = np.polyfit(xData[endStimulusInd + self.stimulusBufferInd:], yData[endStimulusInd + self.stimulusBufferInd:], 1)[0]
        stressSlope = np.polyfit(xData[startStimulusInd:endStimulusInd - int((endStimulusInd-startStimulusInd)/2)], yData[startStimulusInd:endStimulusInd - int((endStimulusInd-startStimulusInd)/2)], 1)[0]
        
        # Extract Shape Features
        peakSTD = np.std(yData[startStimulusInd:], ddof=1)
        peakEntropy = entropy(yData[startStimulusInd:])
        
        # Compile the Features
        gsrFeatures = []
        gsrFeatures.extend([numPeaks, sumPromineces, bestprominence, prominenceRatio])
        gsrFeatures.extend([maxHeight, riseTime, recoveryTime])
        gsrFeatures.extend([meanSignal, meanSignalRest, meanSignalStress, meanSignalRecovery, meanStressIncrease])
        gsrFeatures.extend([peakSTD, peakEntropy, relaxationSlope, stressSlope])
        # ------------------------------------------------------------------ #
        
        return gsrFeatures

    def downsizeDataPoint(self, xData, yData, downsizeWindow = 5):
        
        yDownsizedData = []; xDownsizedData = [];
        yDataHolder = []; xDataHolder = []
        for dataPoint in range(len(xData)):
            xPoint = xData[dataPoint]
            yPoint = yData[dataPoint]
            
            yDataHolder.append(yPoint)
            xDataHolder.append(xPoint)
            
            if len(yDataHolder) == downsizeWindow:
                yDownsizedData.append(np.mean(yDataHolder))
                xDownsizedData.append(np.mean(xDataHolder))
                # Reset Data Holder
                yDataHolder = []; xDataHolder = []

        return xDownsizedData, yDownsizedData
    
    def downsizeDataTime(self, xData, yData, downsizeWindow = 5):
        
        yDownsizedData = []; xDownsizedData = [];
        yDataHolder = []; xDataHolder = []
        for dataPoint in range(len(xData)):
            xPoint = xData[dataPoint]
            yPoint = yData[dataPoint]
            
            yDataHolder.append(yPoint)
            xDataHolder.append(xPoint)
            
            if xPoint >= downsizeWindow*(len(xDownsizedData) + 1):
                yDownsizedData.append(np.mean(yDataHolder))
                xDownsizedData.append(np.mean(xDataHolder))
                # Reset Data Holder
                yDataHolder = []; xDataHolder = []

        return xDownsizedData, yDownsizedData
            




    
    
    
    
    