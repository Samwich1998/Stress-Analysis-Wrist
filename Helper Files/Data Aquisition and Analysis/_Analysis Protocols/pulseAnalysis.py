
# Basic Modules
import math
import numpy as np
from scipy import stats
from bisect import bisect
# Peak Detection
import scipy
import scipy.signal
# Filter the Data
from scipy.signal import butter
from scipy.signal import savgol_filter
# Baseline Subtraction
from BaselineRemoval import BaselineRemoval
# Gaussian Decomposition
from lmfit import Model
from sklearn.metrics import r2_score
# Matlab Plotting API
import matplotlib as mpl
import matplotlib.pyplot as plt


class plot:
    
    def __init__(self):
        self.sectionColors = ['red','orange', 'blue','green', 'black']
    
    def plotData(self, xData, yData, title, ax = None, axisLimits = [], topPeaks = {}, bottomPeaks = {}, peakSize = 3, lineWidth = 2, lineColor = "tab:blue", pulsePeakInds = []):
        # Create Figure
        showFig = False
        if ax == None:
            plt.figure()
            ax = plt.gca()
            showFig = True
        # Plot the Data
        ax.plot(xData, yData, linewidth = lineWidth, color = lineColor)
        if topPeaks:
            ax.plot(topPeaks[1], topPeaks[2], 'or', markersize=peakSize)
        if bottomPeaks:
            ax.plot(bottomPeaks[1], bottomPeaks[2], 'ob', markersize=peakSize)
        if len(pulsePeakInds) > 0:
            for groupInd in range(len(self.sectionColors)):
                if pulsePeakInds[groupInd] in [np.nan, None] or pulsePeakInds[groupInd+1] in [np.nan, None]: 
                    continue
                ax.fill_between(xData[pulsePeakInds[groupInd]:pulsePeakInds[groupInd+1]+1], min(yData), yData[pulsePeakInds[groupInd]:pulsePeakInds[groupInd+1]+1], color=self.sectionColors[groupInd], alpha=0.15)
        # Add Axis Labels and Figure Title
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Capacitance (pF)")
        ax.set_title(title)
        # Change Axis Limits If Given
        if axisLimits:
            ax.set_xlim(axisLimits)
        # Increase DPI (Clarity); Will Slow Down Plotting
        mpl.rcParams['figure.dpi'] = 300
        # Show the Plot
        if showFig:
            plt.show()
    

    def plotPulses(self, bloodPulse, numSubPlotsX = 3, firstPeakPlotting = 1, maxPulsesPlot = 9, figWidth = 25, figHeight = 13, finalPlot = False):
        # Create One Plot with First 9 Pulse Curves
        numSubPlots = min(maxPulsesPlot, len(bloodPulse) - firstPeakPlotting + 1)
        scaleGraph = math.ceil(numSubPlots/numSubPlotsX) / (maxPulsesPlot/numSubPlotsX)
        figHeight = int(figHeight*scaleGraph); figWidth = int(figWidth*min(numSubPlots,numSubPlotsX)/numSubPlotsX)
        
        fig, ax = plt.subplots(math.ceil(numSubPlots/numSubPlotsX), min(numSubPlotsX, numSubPlots), sharey=False, sharex = False, figsize=(figWidth,figHeight))
        fig.suptitle("Indivisual Pulse Peaks", fontsize=20, fontweight ="bold", yData=0.98)
        for figNum, pulseNum in enumerate(list(bloodPulse.keys())[firstPeakPlotting-1:]):
            if figNum == numSubPlots:
                break
            # Keep Running Order of Subplots
            if numSubPlots == 1:
                currentAxes = ax
            elif numSubPlots <= numSubPlotsX:
                currentAxes = ax[figNum]
            else:
                currentAxes = ax[figNum//numSubPlotsX][figNum%numSubPlotsX]
            # Get the Data
            time = bloodPulse[pulseNum]['time']
            filterData = bloodPulse[pulseNum]["normalizedPulse"]
            # Get the Pulse peaks
            bottomInd = []
            pulsePeakInds = bloodPulse[pulseNum]['indicesTop']
            # Plot with Pulses Sectioned Off into Regions
            if finalPlot:
                pulsePeakInds = bloodPulse[pulseNum]['pulsePeakInds']
                self.plotData(time, filterData, title = "Peak Number " + str(pulseNum), ax = currentAxes, topPeaks = {1:time[pulsePeakInds], 2:filterData[pulsePeakInds]}, bottomPeaks = {1:time[bottomInd], 2:filterData[bottomInd]}, peakSize = 5,  lineWidth = 2, lineColor = "black", pulsePeakInds = pulsePeakInds)
            # General Plot
            else:
                # Plot the Data 
                self.plotData(time, filterData, title = "Peak Number " + str(pulseNum), ax = currentAxes, topPeaks = {1:time[pulsePeakInds], 2:filterData[pulsePeakInds]}, bottomPeaks = {1:time[bottomInd], 2:filterData[bottomInd]}, peakSize = 5, lineWidth = 2, lineColor = "black")
        fig.tight_layout(pad= 2.0)
        plt.show()
    
    
    def plotPulseNum(self, bloodPulse, pulseNum, finalPlot = False):
        # Get Data
        time = bloodPulse[pulseNum]['time']
        filterData = bloodPulse[pulseNum]["normalizedPulse"]
        # Get the Pulse peaks 
        bottomInd = []
        pulsePeakInds = bloodPulse[pulseNum]['indicesTop']
        # Plot with Pulses Sectioned Off into Regions
        if finalPlot:
            pulsePeakInds = bloodPulse[pulseNum]['pulsePeakInds']
            self.plotData(time, filterData, title = "Peak Number " + str(pulseNum), topPeaks = {1:time[pulsePeakInds], 2:filterData[pulsePeakInds]}, bottomPeaks = {1:time[bottomInd], 2:filterData[bottomInd]}, peakSize = 5,  lineWidth = 2, lineColor = "black", pulsePeakInds = pulsePeakInds)
        else:
            self.plotData(time, filterData, title = "Peak Number " + str(pulseNum), topPeaks = {1:time[pulsePeakInds], 2:filterData[pulsePeakInds]}, bottomPeaks = {1:time[bottomInd], 2:filterData[bottomInd]}, peakSize = 3, lineWidth = 2, lineColor = "black")
        
# ---------------------------------------------------------------------------#
# ---------------------------------------------------------------------------#

class signalProcessing:
    
    def __init__(self, alreadyFilteredData = False, plotGaussFit = False, plotSeperation = False):
        """
        ----------------------------------------------------------------------
        Input Parameters:
            alreadyFilteredData: Do Not Reprocess Data That has Already been Processed; Just Extract Features
            plotSeperation: Display the Indeces Identified as Around Mid-Sysolic Along with the Data
            plotGaussFit: Display the Gaussian Decomposition of Each Pulse
        ----------------------------------------------------------------------
        """        
        # Program Flags
        self.plotGaussFit = plotGaussFit                # Plot the Guassian Decomposition
        self.plotSeperation = plotSeperation            # Plot the First Derivative and Labeled Systolic Peak Location (General)
        self.alreadyFilteredData = alreadyFilteredData  # If the Data is Already Filtered and Normalize, Do NOT Filter Again
        
        # Data Processing Parameters
        self.minGaussianWidth = 10E-5   # THe Minimum Gaussian Width During Guassian Decomposition
        self.minPeakIndSep = 10         # The Minimum Points Between the Dicrotic and Tail Peak
        
        self.resetGlobalVariables()
        
    def resetGlobalVariables(self):
        # Feature Tracking Parameters
        self.timeOffset = 0             # Store the Time Offset Between Files
        self.numSecondsAverage = 60     # The Number of Swconds to Consider When Taking the Averaging Data
        self.incomingPulseTimes = []    # An Ongoing List Representing the Times of Each Pulse's Peak
        self.heartRateListAverage = []  # An Ongoing List Representing the Heart Rate
        # Feature Lists
        self.featureListExact = []      # List of Lists of Features; Each Index Represents a Pulse; Each Pulse's List Represents its Features
        self.featureListAverage = []    # List of Lists of Features Averaged in Time by self.numSecondsAverage; Each Index Represents a Pulse; Each Pulse's List Represents its Features

        # Peak Seperation Parameters
        self.peakStandard = 0;          # The Max First Deriviative of the Previous Pulse's Systolic Peak
        self.peakStandardInd = 0        # The Index of the Max Derivative in the Previous Pulse's Systolic Peak
        # Peak Filtering Parameters
        self.lowPassCutoff = 18         # Low Pass Filter Cutoff; Used on SEPERATED Pulses
        
        # Systolic and Diastolic References
        self.systolicPressure0 = None   # The Calibrated Systolic Pressure
        self.diastolicPressure0 = None  # The Calibrated Diastolic Pressure
        self.diastolicPressure = None   # The Current Diastolic Pressure
        self.calibratedSystolicAmplitude = None    # The Average Amplitude for the Calibrated Systolic/Diastolic Pressure
        self.calibratedSystolicAmplitudeList = []  # A List of Systolic Amplitudes for the Calibration
        
        # Save Each Filtered Pulse
        self.time = []
        self.signalData = []
        self.filteredData = []
        
    def setPressureCalibration(self, systolicPressure0, diastolicPressure0):
        self.systolicPressure0 = systolicPressure0    # The Calibrated Systolic Pressure
        self.diastolicPressure0 = diastolicPressure0  # The Calibrated Diastolic Pressure
        
    def convertToOddInt(self, x):
        return 2*math.floor((x+1)/2) - 1
        
    def seperatePulses(self, time, firstDer):
        self.peakStandardInd = 0
        # Take First Derivative of Smoothened Data
        systolicPeaks = [];
        for pointInd in range(len(firstDer)):
            # Calcuate the Derivative at pointInd
            firstDerVal = firstDer[pointInd]
            
            # If the Derivative Stands Out, Its the Systolic Peak
            if firstDerVal > self.peakStandard*0.5:
                
                # Use the First Few Peaks as a Standard
                if (self.timeOffset != 0 or 1.5 < time[pointInd]) and self.minPointsPerPulse < pointInd:
                    # If the Point is Sufficiently Far Away, its a New R-Peak
                    if self.peakStandardInd + self.minPointsPerPulse < pointInd:
                        systolicPeaks.append(pointInd)
                    # Else, Find the Max of the Peak
                    elif firstDer[systolicPeaks[-1]] < firstDer[pointInd]:
                        systolicPeaks[-1] = pointInd
                    # Else, Dont Update Pointer
                    else:
                        continue
                    self.peakStandardInd = pointInd
                    self.peakStandard = firstDerVal
                else:
                    self.peakStandard = max(self.peakStandard, firstDerVal)

        return systolicPeaks
        
    
    def analyzePulse(self, time, signalData, minBPM = 27, maxBPM = 480):
        """
        ----------------------------------------------------------------------
        Input Parameters:
            time: xData-Axis Data for the Blood Pulse (Seconds)
            signalData:  yData-Axis Data for Blood Pulse (Capacitance)
            minBPM = Minimum Beats Per Minute Possible. 27 BPM is the lowest recorded; 30 is a good threshold
            maxBPM: Maximum Beats Per Minute Possible. 480 is the maximum recorded. 220 is a good threshold
        Use Case: Seperate the Pulses, Gaussian Decompositions, Feature Extraction
        ----------------------------------------------------------------------
        """       
        print("\nSeperating Pulse Data")
        # ------------------------- Set Up Analysis ------------------------- #
        # Calculate the Sampling Frequency, if None Present
        self.samplingFreq = len(signalData)/(time[-1]-time[0])
        print("\tSampling Frequency: " + str(self.samplingFreq))
  
        # Estimate that Defines the Number of Points in a Pulse
        self.minPointsPerPulse = math.floor(self.samplingFreq*60/maxBPM)
        self.maxPointsPerPulse = math.ceil(self.samplingFreq*60/minBPM)
        
        # Save the Data
        previousData = len(self.time)
        self.time.extend(time + self.timeOffset)
        self.signalData.extend(signalData)
        self.filteredData.extend([0]*len(time))
        # ------------------------------------------------------------------- #

        # ------------------------- Seperate Pulses ------------------------- #
        # Calculate Derivatives
        firstDer = savgol_filter(signalData, 9, 2, mode='nearest', deriv=1)
        # Take First Derivative of Smoothened Data
        systolicPeaks = self.seperatePulses(time, firstDer)
        # If no Systolic peaks found, it is likely there was a noise artifact with a high derivative
        while len(systolicPeaks) == 0:
            self.peakStandard = self.peakStandard/2;
            systolicPeaks = self.seperatePulses(time, firstDer)
        
        # If Questioning: Plot to See How the Pulses Seperated
        if self.plotSeperation:
            systolicPeaks = np.array(systolicPeaks); firstDer = np.array(firstDer)
            scaledData = signalData*max(np.abs(firstDer))/(max(signalData) - min(signalData))
            plt.figure()
            plt.plot(time, scaledData - np.mean(scaledData), label = "Centered + Scaled Signal Data", zorder = 3)
            plt.plot(time, firstDer, label = "First Derivative of Signal Data", zorder = 2)
            plt.plot(time[systolicPeaks], firstDer[systolicPeaks], 'o', label = "Mid-Pulse Rise Identification")
            plt.legend(loc=9, bbox_to_anchor=(1.35, 1));
            plt.hlines(0,time[0], time[-1])
            #plt.xlim(3,5)
            plt.show()
        # ------------------------------------------------------------------- #
                    
        # -------------------------- Pulse Analysis ------------------------- #
        print("\tAnalyzing Pulses")
        # Seperate Peaks Based on the Minimim Before the R-Peak Rise
        pulseStartInd = self.findNearbyMinimum(signalData, systolicPeaks[0], binarySearchWindow=-1, maxPointsSearch=self.maxPointsPerPulse)
        for pulseNum in range(1, len(systolicPeaks)):
            pulseEndInd = self.findNearbyMinimum(signalData, systolicPeaks[pulseNum], binarySearchWindow=-1, maxPointsSearch=self.maxPointsPerPulse)
            self.timePoint = time[pulseEndInd] + self.timeOffset
            
            # -------------------- Calculate Heart Rate --------------------- #
            # Save the Pulse's Time
            self.incomingPulseTimes.append(self.timePoint)
                        
            # Average Heart Rate in Time
            numPulsesAverage = len(self.incomingPulseTimes) - bisect(self.incomingPulseTimes, self.timePoint - self.numSecondsAverage)
            self.heartRateListAverage.append(numPulsesAverage*60/self.numSecondsAverage)
            # --------------------------------------------------------------- #
            
            # ---------------------- Cull Bad Pulses ------------------------ #
            # Check if the Pulse is Too Big: Likely Double Pulse
            if pulseEndInd - pulseStartInd > self.maxPointsPerPulse:
                print("Pulse Too Big; THIS SHOULDNT HAPPEN")
                pulseStartInd = pulseEndInd; continue
            # Check if the Pulse is Too Small; Likely Not an R-Peak
            elif pulseEndInd - pulseStartInd < self.minPointsPerPulse:
                print("Pulse Too Small; THIS SHOULDNT HAPPEN")
                pulseStartInd = pulseEndInd; continue
            # --------------------------------------------------------------- #
            
            # ----------------------- Filter the Pulse ---------------------- #
            # Extract Indivisual Pulse Data
            pulseData = signalData[pulseStartInd:pulseEndInd+1]
            # Filter the pulse, if not already filtered
            if not self.alreadyFilteredData:
                # Apply Low Pass Filter and then Smoothing Function
                pulseData = self.butterFilter(pulseData, self.lowPassCutoff, self.samplingFreq, order = 3, filterType = 'low')
                pulseData = savgol_filter(pulseData, self.convertToOddInt(len(pulseData)/8), 2, mode='nearest')
            # --------------------------------------------------------------- #

            # ------------------ PreProcess the Pulse Data ------------------ #
            # Calculate the Pulse Derivatives
            pulseTime = time[pulseStartInd:pulseEndInd+1] - time[pulseStartInd]
            pulseVelocity = savgol_filter(pulseData, 3, 2, mode='nearest', deriv=1)
            pulseAcceleration = savgol_filter(pulseData, 3, 2, mode='nearest', deriv=2)
            thirdDeriv = savgol_filter(pulseAcceleration, 3, 1, mode='nearest', deriv=1)

            # Normalize the Pulse's Baseline to Zero
            normalizedPulse = pulseData.copy()
            if not self.alreadyFilteredData:
                normalizedPulse = self.normalizePulseBaseline(normalizedPulse, polynomialDegree = 1)
                
            # Calculate the Diastolic Pressure
            self.diastolicPressure = pulseData[0]
            # Calculate Diastolic and Systolic Reference of the First Pulse
            if not self.diastolicPressure0:
                diastolicPressure0 = self.diastolicPressure
                systolicPressure0 = self.findNearbyMaximum(signalData, systolicPeaks[pulseNum-1], binarySearchWindow=1, maxPointsSearch=self.maxPointsPerPulse)
                self.setPressureCalibration(systolicPressure0, diastolicPressure0)
            # --------------------------------------------------------------- #
            
            # -------------------- Extract Pulse Features ------------------- #
            if self.calibratedSystolicAmplitude != None:
                normalizedPulse = self.calibrateAmplitude(normalizedPulse)
                self.filteredData[previousData+pulseStartInd:previousData+pulseEndInd+1] = normalizedPulse
                # Label Systolic, Tidal Wave, Dicrotic, and Tail Wave Peaks Using Gaussian Decomposition   
                self.extractPulsePeaks(pulseTime, normalizedPulse, pulseVelocity, pulseAcceleration, thirdDeriv)
            else:
                self.calibratedSystolicAmplitudeList.append(max(normalizedPulse))
            # --------------------------------------------------------------- #
            
            # Reste for Next Pulse
            pulseStartInd = pulseEndInd
        # ------------------------------------------------------------------- #
        self.timeOffset += time[-1]
        
        if self.calibratedSystolicAmplitude == None:
            self.calibratedSystolicAmplitude = np.mean(self.calibratedSystolicAmplitudeList)
        
        # plt.plot(self.heartRateListAverage, 'k-', linewidth=2)
        # plt.ylim(60, 100)
        # plt.show()
    
    def calibrateAmplitude(self, normalizedPulse):
        scaleAmp = (self.systolicPressure0 - self.diastolicPressure0)/self.calibratedSystolicAmplitude
        return normalizedPulse*scaleAmp
    
    def extractPulsePeaks(self, pulseTime, normalizedPulse, pulseVelocity, pulseAcceleration, thirdDeriv):
        
        # ----------------------- Detect Systolic Peak ---------------------- #        
        # Find Systolic Peak
        systolicPeakInd = self.findNearbyMaximum(normalizedPulse, 0, binarySearchWindow = 4, maxPointsSearch = len(pulseTime))
        # Find UpStroke Peaks
        systolicUpstrokeVelInd = self.findNearbyMaximum(pulseVelocity, 0, binarySearchWindow = 1, maxPointsSearch = systolicPeakInd)
        systolicUpstrokeAccelMaxInd = self.findNearbyMaximum(pulseAcceleration, systolicUpstrokeVelInd, binarySearchWindow = -1, maxPointsSearch = systolicPeakInd)
        systolicUpstrokeAccelMinInd = self.findNearbyMinimum(pulseAcceleration, systolicUpstrokeVelInd, binarySearchWindow = 1, maxPointsSearch = systolicPeakInd)
        # ------------------------------------------------------------------- #
                
        # ---------------------- Detect Tidal Wave Peak --------------------- #     
        bufferToTidal = self.findNearbyMinimum(thirdDeriv, systolicPeakInd+1, binarySearchWindow = 1, maxPointsSearch = int(len(pulseTime)/2))
        # Find Tidal Peak Beginning
        tidalStartInd = self.findNearbyMaximum(thirdDeriv, bufferToTidal+2, binarySearchWindow = 1, maxPointsSearch = int(len(pulseTime)/2))
        tidalStartInd_OPTION2 = self.findNearbyMaximum(pulseAcceleration, systolicPeakInd, binarySearchWindow = 2, maxPointsSearch = int(len(pulseTime)/2))
        # Find Tidal Peak
        tidalPeakInd = self.findNearbyMinimum(thirdDeriv, min(tidalStartInd_OPTION2, tidalStartInd+1), binarySearchWindow = 4, maxPointsSearch = int(len(pulseTime)/2))
        # Find Tidal Peak Ending
        tidalEndInd = self.findNearbyMaximum(thirdDeriv, tidalPeakInd+1, binarySearchWindow = 2, maxPointsSearch = int(len(pulseTime)/2))
        # ------------------------------------------------------------------- #
        
        # ----------------------  Detect Dicrotic Peak ---------------------- #
        dicroticNotchInd = self.findNearbyMinimum(normalizedPulse, tidalEndInd, binarySearchWindow = 1, maxPointsSearch = int(len(pulseTime)/2))
        dicroticPeakInd = self.findNearbyMaximum(normalizedPulse, dicroticNotchInd, binarySearchWindow = 1, maxPointsSearch = int(len(pulseTime)/2))
        
        # Other Extremas Nearby
        dicroticInflectionInd = self.findNearbyMaximum(pulseVelocity, dicroticNotchInd, binarySearchWindow = 2, maxPointsSearch = int(len(pulseTime)/2))
        dicroticFallVelMinInd = self.findNearbyMinimum(pulseVelocity, dicroticInflectionInd, binarySearchWindow = 2, maxPointsSearch = int(len(pulseTime)/2))
        # ------------------------------------------------------------------- #
        
        def plotIt(badReason = ""):
            normalizedPulse1 = normalizedPulse/max(normalizedPulse)
            pulseVelocity1 = pulseVelocity/ max(pulseVelocity)
            pulseAcceleration1 = pulseAcceleration/max(pulseAcceleration)
            thirdDeriv1 = thirdDeriv/max(thirdDeriv)
            
            plt.plot(pulseTime, normalizedPulse1, linewidth = 2, color = "black")
            plt.plot(pulseTime, pulseVelocity1, alpha=0.5)
            plt.plot(pulseTime, pulseAcceleration1, alpha=0.5)
            plt.plot(pulseTime, thirdDeriv1, alpha=0.5)
            #plt.plot(pulseTime, fourthDeriv1, alpha=0.5)

            plt.plot(pulseTime[systolicPeakInd], normalizedPulse1[systolicPeakInd],  'ko')
            plt.plot(pulseTime[systolicUpstrokeVelInd], normalizedPulse1[systolicUpstrokeVelInd],  'ko')
            plt.plot(pulseTime[systolicUpstrokeAccelMaxInd], normalizedPulse1[systolicUpstrokeAccelMaxInd],  'ko')
            plt.plot(pulseTime[systolicUpstrokeAccelMinInd], normalizedPulse1[systolicUpstrokeAccelMinInd],  'ko')

            plt.plot(pulseTime[tidalStartInd], normalizedPulse1[tidalStartInd],  'ro')
            plt.plot(pulseTime[tidalPeakInd], normalizedPulse1[tidalPeakInd],  'go')
            plt.plot(pulseTime[tidalEndInd], normalizedPulse1[tidalEndInd],  'ro')
            
            plt.plot(pulseTime[dicroticNotchInd], normalizedPulse1[dicroticNotchInd],  'bo')
            plt.plot(pulseTime[dicroticPeakInd], normalizedPulse1[dicroticPeakInd],  'bo')
            
            plt.plot(pulseTime[[dicroticInflectionInd, dicroticFallVelMinInd]], normalizedPulse1[[dicroticInflectionInd, dicroticFallVelMinInd]],  'bo')
            
            plt.title("Time: " + str(self.timePoint) + "; " + badReason)
            plt.show()

        # ------------------------- Cull Bad Pulses ------------------------- #
        # Check The Order of the Systolic Peaks
        if not systolicUpstrokeAccelMaxInd < systolicUpstrokeVelInd < systolicUpstrokeAccelMinInd < systolicPeakInd:
            print("\t\tBad Systolic Sequence. Time = ", self.timePoint); 
            # plotIt("SYSTOLIC")
            return None
        # Check The Order of the Tidal Peaks
        elif not tidalPeakInd < tidalEndInd:
            print("\t\tBad Tidal Sequence. Time = ", self.timePoint); 
            # plotIt("TIDAL")
            return None
        # Check The Order of the Dicrotic Peaks
        elif not dicroticNotchInd < dicroticInflectionInd < dicroticPeakInd < dicroticFallVelMinInd:
            print("\t\tBad Dicrotic Sequence. Time = ", self.timePoint); 
            # plotIt("DICROTIC")
            return None
        # Check The Order of the Peaks
        elif not systolicPeakInd < tidalEndInd < dicroticNotchInd - 2:
            print("\t\tBad Peak Sequence. Time = ", self.timePoint); 
            # plotIt("GENERAL")
            return None
        
        # Check If the Dicrotic Peak was Skipped
        if pulseTime[-1]*0.75 < pulseTime[dicroticPeakInd] - pulseTime[systolicUpstrokeAccelMaxInd]:
            print("\t\tDicrotic Peak Likely Skipped Over. Time = ", self.timePoint);
            return None
        # ------------------------------------------------------------------- #
        
        # ----------------------- Feature Extraction ------------------------ #
        allSystolicPeaks = [systolicUpstrokeAccelMaxInd, systolicUpstrokeVelInd, systolicUpstrokeAccelMinInd, systolicPeakInd]
        allTidalPeaks = [tidalPeakInd, tidalEndInd]
        allDicroticPeaks = [dicroticNotchInd, dicroticInflectionInd, dicroticPeakInd, dicroticFallVelMinInd]
        
        # Extract the Pulse Features
        self.extractFeatures(normalizedPulse, pulseTime, pulseVelocity, pulseAcceleration, allSystolicPeaks, allTidalPeaks, allDicroticPeaks)
        # ------------------------------------------------------------------- #
        
        # # If the Systolic, Dicrotic, and Tail Peaks were Found (Don't Need Tidal)
        # tailPeak2 = None
        # pulsePeakInds = [0, systolicPeakInd, tidalPeakInd, dicroticPeakInd, int(dicroticPeakInd*1.5), len(pulseTime)-1]
        # if None not in pulsePeakInds:
        #     # Perform Gaussian Decomposition on the Data
        #     pulsePeakInds, gaussPeakInds, gaussPeakAmps = self.gausDecomp(pulseTime, normalizedPulse, pulsePeakInds, addExtraGauss = tailPeak2 != None)
        #     if len(pulsePeakInds) != 0:
        #         self.extractFeatures(normalizedPulse, pulseTime, pulseVelocity, pulseAcceleration, pulsePeakInds, gaussPeakInds, gaussPeakAmps)

        if self.plotGaussFit:
            normalizedPulse1 = normalizedPulse/max(normalizedPulse)
            pulseVelocity1 = pulseVelocity/ max(pulseVelocity)
            pulseAcceleration1 = pulseAcceleration/max(pulseAcceleration)
            thirdDeriv1 = thirdDeriv/max(thirdDeriv)
            
            plt.plot(pulseTime, normalizedPulse1, linewidth = 2, color = "black")
            plt.plot(pulseTime, pulseVelocity1, alpha=0.5)
            plt.plot(pulseTime, pulseAcceleration1, alpha=0.5)
            plt.plot(pulseTime, thirdDeriv1, alpha=0.5)
            
            plt.plot(pulseTime[systolicPeakInd], normalizedPulse1[systolicPeakInd],  'ko')
            plt.plot(pulseTime[systolicUpstrokeVelInd], normalizedPulse1[systolicUpstrokeVelInd],  'ko')
            plt.plot(pulseTime[systolicUpstrokeAccelMaxInd], normalizedPulse1[systolicUpstrokeAccelMaxInd],  'ko')
            plt.plot(pulseTime[systolicUpstrokeAccelMinInd], normalizedPulse1[systolicUpstrokeAccelMinInd],  'ko')

            plt.plot(pulseTime[tidalStartInd], normalizedPulse1[tidalStartInd],  'ro')
            plt.plot(pulseTime[tidalPeakInd], normalizedPulse1[tidalPeakInd],  'ro')
            plt.plot(pulseTime[tidalEndInd], normalizedPulse1[tidalEndInd],  'ro')

            plt.plot(pulseTime[dicroticNotchInd], normalizedPulse1[dicroticNotchInd],  'bo')
            plt.plot(pulseTime[dicroticPeakInd], normalizedPulse1[dicroticPeakInd],  'bo')
            
            plt.plot(pulseTime[[dicroticInflectionInd, dicroticFallVelMinInd]], normalizedPulse1[[dicroticInflectionInd, dicroticFallVelMinInd]],  'bo')
            
            plt.title("Time: " + str(self.timePoint))
            plt.show()
    
    
    def gaussModel(self, xData, amplitude, fwtm, center):
        sigma = fwtm/(2*math.sqrt(2*math.log(10)))
        return amplitude * np.exp(-(xData-center)**2 / (2*sigma**2))
            
    
    def gausDecomp(self, xData, yData, pulsePeakInds, addExtraGauss = False):
        # https://lmfit.github.io/lmfit-py/builtin_models.html#example-1-fit-peak-data-to-gaussian-lorentzian-and-voigt-profiles

        peakAmp = []; peakCenter = []; peakWidth = []
        # Extract Guesses About What the Peak Width, Center, and Amplitude Are
        for currentInd in range(1,5):
            peakInd = pulsePeakInds[currentInd]
            
            # If There Was No Tidal Pulse Detected
            if not peakInd:
                # Estimate the Pulse to be Between the Systolic and Dicrotic Peaks
                peakInd = int((pulsePeakInds[currentInd - 1] + pulsePeakInds[currentInd + 1])/2)   
            
            # Get the Peak's Amplitude and Center
            peakAmp.append(yData[peakInd])
            peakCenter.append(xData[peakInd])
            # Get the Peak's Width: Difference Between the Last Two Centers
            peakWidth.append(2*(peakCenter[currentInd-1] - peakCenter[currentInd-2]))
            
        
        # Systolic Peak Model
        gauss1 = Model(self.gaussModel, prefix = "g1_")
        pars = gauss1.make_params()
        pars['g1_center'].set(value = xData[pulsePeakInds[1]], min = xData[pulsePeakInds[1]]*.95, max = min(xData[pulsePeakInds[1]]*1.05, 0.99*xData[pulsePeakInds[2]]))
        pars['g1_fwtm'].set(value = 2*xData[pulsePeakInds[1]], min = self.minGaussianWidth, max = xData[pulsePeakInds[3]])
        pars['g1_amplitude'].set(value = yData[pulsePeakInds[1]], min = yData[pulsePeakInds[1]]*0.9, max = yData[pulsePeakInds[1]])
        
        # Tidal Wave Model
        gauss2 = Model(self.gaussModel, prefix = "g2_")
        pars.update(gauss2.make_params())
        pars['g2_center'].set(value = xData[pulsePeakInds[2]], min = max(xData[pulsePeakInds[2]]*.8, xData[pulsePeakInds[1]]), max = min(xData[pulsePeakInds[2]]*1.2, xData[pulsePeakInds[3]]))
        pars['g2_fwtm'].set(value = peakWidth[1], min = self.minGaussianWidth, max = 1.1*(peakCenter[2] - peakCenter[0]))
        pars['g2_amplitude'].set(value = peakAmp[1], min = peakAmp[1]*.8, max = peakAmp[1]*1.05)
        
        # Dicrotic Peak Model
        gauss3 = Model(self.gaussModel, prefix = "g3_")
        pars.update(gauss3.make_params())
        pars['g3_center'].set(value = peakCenter[2], min = peakCenter[2]*.9, max =min(peakCenter[2]*1.1, peakCenter[3]))
        pars['g3_fwtm'].set(value = peakWidth[2], min = self.minGaussianWidth, max = 2*(peakCenter[2] - peakCenter[0]))
        pars['g3_amplitude'].set(value = peakAmp[2], min = peakAmp[2]*.9, max = peakAmp[2]*1.02)
        
        # Tail Wave Model
        gauss4 = Model(self.gaussModel, prefix = "g4_")
        pars.update(gauss4.make_params())
        pars['g4_center'].set(value = peakCenter[3], min = min(peakCenter[2] + 0.5*(peakCenter[2]- peakCenter[1]), peakCenter[3]), max = min(peakCenter[3]*1.1, xData[-1]))
        pars['g4_fwtm'].set(value = xData[-1] - peakCenter[3], min = self.minGaussianWidth, max = xData[-1] - peakCenter[1])
        pars['g4_amplitude'].set(value = peakAmp[3], min = peakAmp[3]*.8, max = peakAmp[3]*1.2)
        
        # Add Models Together
        mod = gauss1 + gauss2 + gauss3 + gauss4
        # Add Extra Gaussian to Tail if The Previous Fit Was Bad
        if addExtraGauss:
            gauss5 = Model(self.gaussModel, prefix = "g5_")
            pars.update(gauss5.make_params())
            pars['g5_amplitude'].set(value = peakAmp[3]/6, min = 0, max = peakAmp[3]/2)
            pars['g5_fwtm'].set(value = xData[-1] - peakCenter[3], min = 0, max = xData[-1] - peakCenter[2])
            pars['g5_center'].set(value = min(peakCenter[3]*1.05, xData[-1]), min = min(peakCenter[2] + (peakCenter[2] - peakCenter[1]), peakCenter[3]*1.05, xData[-1]*.99), max = xData[-1])
            mod += gauss5
        
        # Get Fit Information
        finalFitInfo = mod.fit(yData, pars, xData=xData, method='powell')
        #fitReport = finalFitInfo.fit_report(min_correl=0.6); print(fitReport)
        
        # Calcluate Different RSquared Methods
        startCheck = 3
        rSquared1 = 1 - finalFitInfo.residual[startCheck:].var() / np.var(yData[startCheck:])
        rSquared2 = 1 - finalFitInfo.redchi / np.var(yData[startCheck:], ddof=1)
        coefficient_of_dermination = r2_score(yData[startCheck:], finalFitInfo.best_fit[startCheck:])
        # Statistics for Fit
        errorSQ = finalFitInfo.residual[2:-2]**2  # Ignore First/Last 2 Points (Bad EndPoint Fit Given Smoothing)
        meanErrorSQ = np.mean(errorSQ)
        #print(rSquared1, rSquared2, coefficient_of_dermination, meanErrorSQ)
        
        # Plot the Pulse with its Fit 
        def plotGaussianFit(xData, yData, pulsePeakInds):
            xData = np.array(xData); yData = np.array(yData)
            pulsePeakInds = np.array(pulsePeakInds, dtype = int)
            dely = finalFitInfo.eval_uncertainty(sigma=3)
            plt.plot(xData, yData, linewidth = 2, color = "black")
            plt.plot(xData[pulsePeakInds], yData[pulsePeakInds], 'o')
            plt.plot(xData, comps['g1_'], '--', color = "tab:red", alpha = 0.8, label='Systolic Pulse')
            plt.plot(xData, comps['g2_'], '--', color = "tab:green", alpha = 0.8, label='Tidal Wave Pulse')
            plt.plot(xData, comps['g3_'], '--', color = "tab:blue", alpha = 0.8, label='Dicrotic Pulse')
            plt.plot(xData, comps['g4_'], '--', color = "tab:purple", alpha = 0.8, label='Tail Wave Pulse')
            
            if addExtraGauss:
                plt.plot(xData, comps['g5_'], '--', color = "tab:orange", alpha = 0.5, label='Extra Tail Pulse')
            
            plt.fill_between(xData, finalFitInfo.best_fit-dely, finalFitInfo.best_fit+dely, color="#ABABAB",
                 label='3-$\sigma$ uncertainty band')
            
            plt.legend(loc='best')
            plt.title("Gaussian Decomposition at Time " + str(self.timePoint))
            plt.show()
        
        comps = finalFitInfo.eval_components(xData=xData)
        plotGaussianFit(xData, yData, pulsePeakInds)
        # Only Take Pulses with a Good Fit
        if rSquared1 > 0.98 and rSquared2 > 0.98 and coefficient_of_dermination > 0.98 and meanErrorSQ < 2E-2:
            # Extract Data From Gaussian's in Fit to Save
            comps = finalFitInfo.eval_components(xData=xData)
            gaussPeakInds = []; gaussPeakAmps = []
            for peakInd in range(1,5):
                # Save the Gaussian Center's Index and Amplitude
                gaussPeakInds.append(comps['g'+str(peakInd)+'_'].argmax())
                gaussPeakAmps.append(max(comps['g'+str(peakInd)+'_']))
            # If We Previously Missed the Tidal Wave, Use the Gaussian's Tidal Wave Index
            if not pulsePeakInds[2]:
                pulsePeakInds[2] = gaussPeakInds[1]
            # Plot Gaussian Fit
            if self.plotGaussFit:
                plotGaussianFit(xData, yData, pulsePeakInds)
            # Return True if it Worked
            #plotGaussianFit(xData, yData, pulsePeakInds)
            return pulsePeakInds, gaussPeakInds, gaussPeakAmps
        # If Bad, Try and Add an Extra Gaussian to the Tail
        elif not addExtraGauss:
            return self.gausDecomp(xData, yData, pulsePeakInds, addExtraGauss = True)
        # If Still Bad, Throw Out the Pulse
        return [], [], []

    def extractFeatures(self, normalizedPulse, pulseTime, pulseVelocity, pulseAcceleration, allSystolicPeaks, allTidalPeaks, allDicroticPeaks):
     
        # ------------------- Extract Data from Peak Inds ------------------- #        
        # Unpack All Peak Inds
        systolicUpstrokeAccelMaxInd, systolicUpstrokeVelInd, systolicUpstrokeAccelMinInd, systolicPeakInd = allSystolicPeaks
        tidalPeakInd, tidalEndInd = allTidalPeaks
        dicroticNotchInd, dicroticInflectionInd, dicroticPeakInd, dicroticFallVelMinInd = allDicroticPeaks
        
        # Find TimePoints of All Peaks
        systolicUpstrokeAccelMaxTime, systolicUpstrokeVelTime, systolicUpstrokeAccelMinTime, systolicPeakTime = pulseTime[allSystolicPeaks]
        tidalPeakTime, tidalEndTime = pulseTime[allTidalPeaks]
        dicroticNotchTime, dicroticRiseVelMaxTime, dicroticPeakTime, dicroticFallVelMinTime = pulseTime[allDicroticPeaks]
        # Find Amplitude of All Peaks
        systolicUpstrokeAccelMaxAmp, systolicUpstrokeVelAmp, systolicUpstrokeAccelMinAmp, systolicPeakAmp = normalizedPulse[allSystolicPeaks]
        tidalPeakAmp, tidalEndAmp = normalizedPulse[allTidalPeaks]
        dicroticNotchAmp, dicroticRiseVelMaxAmp, dicroticPeakAmp, dicroticFallVelMinAmp = normalizedPulse[allDicroticPeaks]
        # Find Velocity of All Peaks
        systolicUpstrokeAccelMaxVel, systolicUpstrokeVelVel, systolicUpstrokeAccelMinVel, systolicPeakVel = pulseVelocity[allSystolicPeaks]
        tidalPeakVel, tidalEndVel = pulseVelocity[allTidalPeaks]
        dicroticNotchVel, dicroticRiseVelMaxVel, dicroticPeakVel, dicroticFallVelMinVel = pulseVelocity[allDicroticPeaks]
        # Find Acceleration of All Peaks
        systolicUpstrokeAccelMaxAccel, systolicUpstrokeVelAccel, systolicUpstrokeAccelMinAccel, systolicPeakAccel = pulseAcceleration[allSystolicPeaks]
        tidalPeakAccel, tidalEndAccel = pulseAcceleration[allTidalPeaks]
        dicroticNotchAccel, dicroticRiseVelMaxAccel, dicroticPeakAccel, dicroticFallVelMinAccel = pulseAcceleration[allDicroticPeaks]
        # ------------------------------------------------------------------- #
        
        # -------------------------- Time Features -------------------------- #        
        # Diastole and Systole Parameters
        pulseDuration = pulseTime[-1]
        systolicTime = pulseTime[dicroticNotchInd]
        DiastolicTime = pulseDuration - systolicTime
        leftVentricularPerformance = systolicTime/DiastolicTime
        
        # Time from Systolic Peak
        maxDerivToSystolic = systolicPeakTime - systolicUpstrokeVelTime
        systolicToTidal = tidalPeakTime - systolicPeakTime
        systolicToDicroticNotch = dicroticNotchTime - systolicPeakTime
        # Time from Dicrotic Notch
        dicroticNotchToTidal = tidalPeakTime - dicroticNotchTime
        dicroticNotchToDicrotic = dicroticPeakTime - dicroticNotchTime
        
        # General Times
        systolicUpSlopeTime = systolicUpstrokeAccelMinTime - systolicUpstrokeAccelMaxTime
        midToEndTidal = tidalEndTime - tidalPeakTime
        tidalToDicroticVelPeakInterval = dicroticRiseVelMaxTime - tidalPeakTime
        # ------------------------------------------------------------------- #

        # --------------------- Under the Curve Features -------------------- #        
        # Calculate the Area Under the Curve
        pulseArea = scipy.integrate.simpson(normalizedPulse, pulseTime)
        pulseAreaSquared = scipy.integrate.simpson(normalizedPulse**2, pulseTime)
        leftVentricleLoad = scipy.integrate.simpson(normalizedPulse[0:dicroticNotchInd+1], pulseTime[0:dicroticNotchInd+1])
        diastolicArea = pulseArea - leftVentricleLoad
        
        # General Areas
        systolicUpSlopeArea = scipy.integrate.simpson(normalizedPulse[systolicUpstrokeAccelMaxInd:systolicUpstrokeAccelMinInd+1], pulseTime[systolicUpstrokeAccelMaxInd:systolicUpstrokeAccelMinInd+1])
        velToTidalArea = scipy.integrate.simpson(normalizedPulse[systolicUpstrokeVelInd:tidalPeakInd+1], pulseTime[systolicUpstrokeVelInd:tidalPeakInd+1])

        # Average of the Pulse
        pulseAverage = np.mean(normalizedPulse)
        # ------------------------------------------------------------------- #

        # -------------------------- Ratio Features ------------------------- #        
        # Systole and Diastole Ratios
        areaRatio = leftVentricleLoad/diastolicArea
        systolicDicroticNotchAmpRatio = dicroticNotchAmp/systolicPeakAmp
        systolicDicroticNotchVelRatio = dicroticNotchVel/systolicPeakVel
        systolicDicroticNotchAccelRatio = dicroticNotchAccel/systolicPeakAccel
        
        # Other Systolic Amplitude Ratios
        systolicTidalAmpRatio = tidalPeakAmp/systolicPeakAmp
        systolicDicroticAmpRatio = dicroticPeakAmp/systolicPeakAmp
        # Other Diastole Ratios
        dicroticNotchTidalAmpRatio = tidalPeakAmp/dicroticNotchAmp
        dicroticNotchDicroticAmpRatio = dicroticPeakAmp/dicroticNotchAmp
        
        # Systolic Velocty Ratios
        systolicTidalVelRatio = tidalPeakVel/systolicPeakVel
        systolicDicroticVelRatio = dicroticPeakVel/systolicPeakVel
        # Diastole Velocity Ratios
        dicroticNotchTidalVelRatio = tidalPeakVel/dicroticNotchVel
        dicroticNotchDicroticVelRatio = dicroticPeakVel/dicroticNotchVel
        
        # Systolic Acceleration Ratios
        systolicTidalAccelRatio = tidalPeakAccel/systolicPeakAccel
        systolicDicroticAccelRatio = dicroticPeakAccel/systolicPeakAccel
        # Diastole Acceleration Ratios
        dicroticNotchTidalAccelRatio = tidalPeakAccel/dicroticNotchAccel
        dicroticNotchDicroticAccelRatio = dicroticPeakAccel/dicroticNotchAccel
        # ------------------------------------------------------------------- #

        # -------------------------- Slope Features --------=---------------- #        
        # Systolic Slopes
        systolicSlopeUp = np.polyfit(pulseTime[systolicUpstrokeAccelMaxInd: systolicUpstrokeAccelMinInd], normalizedPulse[systolicUpstrokeAccelMaxInd: systolicUpstrokeAccelMinInd], 1)[0]

        # Tidal Slopes
        tidalSlope = np.polyfit(pulseTime[tidalPeakInd:tidalEndInd], normalizedPulse[tidalPeakInd:tidalEndInd], 1)[0]
        
        # Dicrotic Slopes
        DicroticSlopeUp = np.polyfit(pulseTime[dicroticNotchInd:dicroticPeakInd], normalizedPulse[dicroticNotchInd:dicroticPeakInd], 1)[0]
        
        # Tail Slopes
        endSlope =  np.polyfit(pulseTime[dicroticFallVelMinInd:], normalizedPulse[dicroticFallVelMinInd:], 1)[0]
        # ------------------------------------------------------------------- #
        
        # ----------------------- Biological Features ----------------------- #        
        # Find the Diastolic and Systolic Pressure
        diastolicPressure = self.diastolicPressure
        systolicPressure = self.diastolicPressure + systolicPeakAmp
        pressureRatio = systolicPressure/diastolicPressure

        momentumDensity = 2*pulseTime[-1]*pulseArea
        meanArterialBloodPressure = diastolicPressure + systolicPeakAmp/3
        pseudoCardiacOutput = pulseArea/pulseTime[-1]
        pseudoSystemicVascularResistance = meanArterialBloodPressure/pulseTime[-1]
        pseudoStrokeVolume = pseudoCardiacOutput/pulseTime[-1]
                
        maxSystolicVelocity = max(pulseVelocity)
        valveCrossSectionalArea = pseudoCardiacOutput/maxSystolicVelocity
        velocityTimeIntegral = scipy.integrate.simpson(pulseVelocity, pulseTime)
        velocityTimeIntegralABS = scipy.integrate.simpson(abs(pulseVelocity), pulseTime)
        velocityTimeIntegral_ALT = pseudoStrokeVolume/valveCrossSectionalArea

        # Add Index Parameters: https://www.vitalscan.com/dtr_pwv_parameters.htm
        centralAugmentationIndex = normalizedPulse[np.argmax(pulseVelocity)]/systolicPeakAmp  # Tidal Peak / Systolic Max Vel yData
        centralAugmentationIndex_EST = tidalPeakAmp/systolicPeakAmp  # Tidal Peak / Systolic Max Vel yData
        reflectionIndex = dicroticPeakAmp/systolicPeakAmp  # Dicrotic Peak / Systolic Peak
        stiffensIndex = 1/(dicroticPeakTime - systolicPeakTime)  # 1/ Time from the Systolic to Dicrotic Peaks
        # ------------------------------------------------------------------- #
        
        # ------------------------ Organize Features ------------------------ #        
        pulseFeatures = [self.timePoint]
        # Saving Features from Section: Extract Data from Peak Inds
        pulseFeatures.extend([systolicUpstrokeAccelMaxTime, systolicUpstrokeVelTime, systolicUpstrokeAccelMinTime, systolicPeakTime])
        pulseFeatures.extend([tidalPeakTime, tidalEndTime])
        pulseFeatures.extend([dicroticNotchTime, dicroticRiseVelMaxTime, dicroticPeakTime, dicroticFallVelMinTime])
        pulseFeatures.extend([systolicUpstrokeAccelMaxAmp, systolicUpstrokeVelAmp, systolicUpstrokeAccelMinAmp, systolicPeakAmp])
        pulseFeatures.extend([tidalPeakAmp, tidalEndAmp])
        pulseFeatures.extend([dicroticNotchAmp, dicroticRiseVelMaxAmp, dicroticPeakAmp, dicroticFallVelMinAmp])
        pulseFeatures.extend([systolicUpstrokeAccelMaxVel, systolicUpstrokeVelVel, systolicUpstrokeAccelMinVel, systolicPeakVel])
        pulseFeatures.extend([tidalPeakVel, tidalEndVel])
        pulseFeatures.extend([dicroticNotchVel, dicroticRiseVelMaxVel, dicroticPeakVel, dicroticFallVelMinVel])
        pulseFeatures.extend([systolicUpstrokeAccelMaxAccel, systolicUpstrokeVelAccel, systolicUpstrokeAccelMinAccel, systolicPeakAccel])
        pulseFeatures.extend([tidalPeakAccel, tidalEndAccel])
        pulseFeatures.extend([dicroticNotchAccel, dicroticRiseVelMaxAccel, dicroticPeakAccel])
        
        # Saving Features from Section: Time Features
        pulseFeatures.extend([pulseDuration, systolicTime, DiastolicTime, leftVentricularPerformance])
        pulseFeatures.extend([maxDerivToSystolic, systolicToTidal, systolicToDicroticNotch, dicroticNotchToTidal, dicroticNotchToDicrotic])
        pulseFeatures.extend([systolicUpSlopeTime, midToEndTidal, tidalToDicroticVelPeakInterval])

        # Saving Features from Section: Under the Curve Features
        pulseFeatures.extend([pulseArea, pulseAreaSquared, leftVentricleLoad, diastolicArea])
        pulseFeatures.extend([systolicUpSlopeArea, velToTidalArea, pulseAverage])
        
        # Saving Features from Section: Ratio Features
        pulseFeatures.extend([areaRatio, systolicDicroticNotchAmpRatio, systolicDicroticNotchVelRatio, systolicDicroticNotchAccelRatio])
        pulseFeatures.extend([systolicTidalAmpRatio, systolicDicroticAmpRatio, dicroticNotchTidalAmpRatio, dicroticNotchDicroticAmpRatio])
        pulseFeatures.extend([systolicTidalVelRatio, systolicDicroticVelRatio, dicroticNotchTidalVelRatio, dicroticNotchDicroticVelRatio])
        pulseFeatures.extend([systolicTidalAccelRatio, systolicDicroticAccelRatio, dicroticNotchTidalAccelRatio, dicroticNotchDicroticAccelRatio])

        # Saving Features from Section: Slope Features
        pulseFeatures.extend([systolicSlopeUp, tidalSlope, DicroticSlopeUp, endSlope])

        # Saving Features from Section: Biological Features
        pulseFeatures.extend([momentumDensity, pseudoCardiacOutput, pseudoStrokeVolume])
        pulseFeatures.extend([diastolicPressure, systolicPressure, pressureRatio, meanArterialBloodPressure, pseudoSystemicVascularResistance, pseudoStrokeVolume])
        pulseFeatures.extend([maxSystolicVelocity, valveCrossSectionalArea, velocityTimeIntegral, velocityTimeIntegralABS, velocityTimeIntegral_ALT])
        pulseFeatures.extend([centralAugmentationIndex, centralAugmentationIndex_EST, reflectionIndex, stiffensIndex])

        # Save the Pulse Features
        pulseFeatures = np.array(pulseFeatures)
        self.featureListExact.append(pulseFeatures)
        self.featureListAverage.extend(stats.trim_mean(np.array(self.featureListExact)[:,1:][ np.array(self.featureListExact)[:,0] >= self.timePoint - self.numSecondsAverage ], 0.3))
    
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

    def findNearbyMinimum(self, data, xPointer, binarySearchWindow = 5, maxPointsSearch = 500):
        """
        Search Right: binarySearchWindow > 0
        Search Left: binarySearchWindow < 0
        """
        # Base Case
        if abs(binarySearchWindow) < 1 or maxPointsSearch == 0:
            return xPointer - min(2, xPointer) + np.argmin(data[max(0,xPointer-2):min(xPointer+3, len(data))]) 
        
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
            return xPointer - min(2, xPointer) + np.argmax(data[max(0,xPointer-2):min(xPointer+3, len(data))]) 
        
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
    
    
    def window_rms(self, inputData, window_size):
        """
        ----------------------------------------------------------------------
        Input Parameters:
            inputData:  yData-Axis Data for Blood Pulse (First Derivative)
            window_size: Size of Window to Take the Root Mean Squared
        Output Parameters:
            pulseRMS: Root Mean Squared of yData-Axis Data
        Use Case: Increase the Gradient of the Systolic Peak to Differentiate it More
        Assumption for Later Use: The Window Size is Not too Big as to Average Everything
        ----------------------------------------------------------------------
        """
        dataSquared = np.power(inputData, 2)
        window = np.ones(window_size)/float(window_size)
        return np.sqrt(np.convolve(dataSquared, window, 'same'))
    
    
    def normalizePulseBaseline(self, pulseData, polynomialDegree):
        """
        ----------------------------------------------------------------------
        Input Parameters:
            pulseData:  yData-Axis Data for a Single Pulse (Start-End)
            polynomialDegree: Polynomials Used in Baseline Subtraction
        Output Parameters:
            pulseData: yData-Axis Data for a Baseline-Normalized Pulse (Start, End = 0)
        Use Case: Shift the Pulse to the xData-Axis (Removing non-Horizontal Base)
        Assumption in Function: pulseData is Positive
        ----------------------------------------------------------------------
        Further API Information Can be Found in the Following Link:
        https://pypi.org/project/BaselineRemoval/
        ----------------------------------------------------------------------
        """
        # Perform Baseline Removal Twice to Ensure Baseline is Gone
        for _ in range(2):
            # Baseline Removal Procedure
            baseObj = BaselineRemoval(pulseData)  # Create Baseline Object
            pulseData = baseObj.ModPoly(polynomialDegree) # Perform Modified multi-polynomial Fit Removal
            
        # Return the Data With Removed Baseline
        return pulseData
    
    def findRightMaximum(self, yData, xPointer, searchWindow = 50):
        currentMax = yData[xPointer]
        for dataPoint in range(xPointer+1, min(xPointer + searchWindow, len(yData))):
            if currentMax < yData[dataPoint]:
                currentMax = yData[dataPoint]
            else:
                return dataPoint - 1
        return dataPoint
            



"""
        # Find Mid-Ampltiude of the R-Peak (Systolic Peak; First Peak)
        #possiblesystolicPeaks = scipy.signal.find_peaks(firstDer, prominence = .9, distance = minPointsPerPulse, width=2)[0]
        # Cull Bad Pulse Indices
        # lastGoodInd = 1; systolicPeaks = []; gotGoodPeak = False;
        # for currentPlace, peakCompareInd in enumerate(possiblesystolicPeaks):
        #     peakStandard = firstDer[possiblesystolicPeaks[lastGoodInd]]
        #     peakCompare = firstDer[peakCompareInd]
        #     # Good Peak
        #     if peakCompare > peakStandard * 0.5:
        #         systolicPeaks.append(peakCompareInd)
        #         lastGoodInd = currentPlace
        #         gotGoodPeak = True
        #     elif not gotGoodPeak:
        #         print("Here")
        #         lastGoodInd += 1
        
        
                
        # Take First Derivative of Smoothened Data
        firstDer = [0]; peakStandard = 0
        systolicPeaks = []; lastPeakRise = 0
        for pointInd in range(1, len(signalData)):
            # Calcuate the Derivative at pointInd
            deltaY = signalData[pointInd] - signalData[max(pointInd-1, 0)]
            deltaX = time[pointInd] - time[max(pointInd-1, 0)]
            firstDer.append(deltaY/deltaX)
            
            # If the Derivative Stands Out, Its the Systolic Peak
            if firstDer[-1] > peakStandard*0.5:
                peakStandard = firstDer[-1]
                # Use the First Few Peaks as a Standard
                if 1.5 < time[pointInd]:
                    # If the Point is Sufficiently Far Away, its a New R-Peak
                    if lastPeakRise + minPointsPerPulse < pointInd:
                        systolicPeaks.append(pointInd)
                    # Else, Find the Max of the Peak
                    elif firstDer[systolicPeaks[-1]] < firstDer[pointInd]:
                        systolicPeaks[-1] = pointInd
                    lastPeakRise = pointInd
                    
        # tailPeak1 = None
        # tailPeak2 = None
        # velPeakInds = scipy.signal.find_peaks(pulseVelocity[dicroticPeakInd + 2:], prominence = 10E-5, distance = self.minPeakIndSep)[0]
        # peakInds = scipy.signal.find_peaks(normalizedPulse[dicroticPeakInd + 2:], distance = self.minPeakIndSep)[0]
        # # If There is a Dicrotic Peak, Look for Tail Peak (Else, It is Worthless to Check; Bad Pulse)
        # if dicroticPeakInd:
        #     # Get Possible Tail Wave Peaks AFTER First Gradient Hump Thats AFTER Dicrotic
        #     derivTail = velPeakInds[velPeakInds > dicroticPeakInd + self.minPeakIndSep]
        #     if len(derivTail) > 0:
        #         tailPeaks = peakInds[peakInds > derivTail[0]]
        #     else:
        #         tailPeaks = peakInds[peakInds > dicroticPeakInd + self.minPeakIndSep]
        #     # Get the Maximum Tail Peak (Small Weight to Take One Further Out) and Label it Tail Peak
        #     tailPeak1 = max(tailPeaks, key = lambda tailInd: normalizedPulse[tailInd] + tailInd/(6*peakInds[-1]), default=None)
        #     # If No peakInds After the Dicrotic Peak, Use the Gradient to Find Tail Peak Hump
        #     if not tailPeak1 and len(peakInds) != 0:
        #         tailPeak1 = max(derivTail, key = lambda tailInd: normalizedPulse[tailInd] + tailInd/(6*peakInds[-1]), default=None) 
        #     elif tailPeak1:
        #         tailPeaks = tailPeaks[tailPeak1 < tailPeaks]
        #         if len(tailPeaks) != 0:
        #             tailPeak2 = tailPeaks[0]
        # if not tailPeak1:
        #     tailPeak1 = dicroticNotchInd + int(3*(len(normalizedPulse) - dicroticNotchInd)/4)
"""
    
    
    
    
    