
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

# Import Files
import _filteringProtocols as filteringMethods # Import Files with Filtering Methods

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
    
    def analyzeTemperature(self, xData, yData):
        
        # ------------------------- Filter the Data ------------------------ #
        # Apply a Low Pass Filter
        self.samplingFreq = len(xData)/(xData[-1] - xData[0])
        yData = self.filteringMethods.bandPassFilter.butterFilter(yData, self.lowPassCutoff, self.samplingFreq, order = 4, filterType = 'low')
        yData = self.filteringMethods.savgolFilter.savgolFilter(yData, 21, 2)
        # ------------------------------------------------------------------ #
        
        # ----------------------- Feature Extraction ----------------------- #
        # Find the Stimulus Start/Stop Ind
        startStimulusInd = np.argmin(abs(xData - self.startStimulusTime))
        endStimulusInd = np.argmin(abs(xData - self.endStimulusTime))
        
        # Extract Mean of the Data
        meanSignal = np.mean(yData)
        meanSignalRest = np.mean(yData[0:startStimulusInd])
        meanSignalStress = np.mean(yData[startStimulusInd:endStimulusInd])
        meanSignalRecovery = np.mean(yData[endStimulusInd:])
        meanStressIncrease = meanSignalStress - meanSignalRest
        
        # Extract Shape Features
        peakSTD = np.std(yData[startStimulusInd:], ddof=1)
        peakEntropy = entropy(yData[startStimulusInd:])
        
        # Extract Slopes
        stressSlope = np.polyfit(xData[startStimulusInd + int((startStimulusInd-endStimulusInd)/4):endStimulusInd], yData[startStimulusInd + int((startStimulusInd-endStimulusInd)/4):endStimulusInd], 1)[0]
        relaxationSlope = np.polyfit(xData[endStimulusInd + self.stimulusBufferInd:], yData[endStimulusInd + self.stimulusBufferInd:], 1)[0]

        # Compile the Features
        temperatureFeatures = []
        temperatureFeatures.extend([meanSignal, meanSignalRest, meanSignalStress, meanSignalRecovery, meanStressIncrease])
        temperatureFeatures.extend([peakSTD, peakEntropy, stressSlope, relaxationSlope])
        # ------------------------------------------------------------------ #
        
        return temperatureFeatures


    
    
    
    
    