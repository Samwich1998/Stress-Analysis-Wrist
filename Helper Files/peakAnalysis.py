
# Basic Modules
import math
import heapq
import numpy as np
# Data Smoothing and First Derivative
from scipy.signal import savgol_filter
from scipy.interpolate import splrep, splev
# Baseline Subtraction
from BaselineRemoval import BaselineRemoval
# Peak Detection
import scipy
from scipy.signal import argrelextrema
from scipy.signal import find_peaks, peak_widths
# Gaussian Decomposition
from lmfit import Model
from sklearn.metrics import r2_score
# Matlab Plotting API
import matplotlib as mpl
import matplotlib.pyplot as plt
  

class plot:
    
    def __init__(self):
        self.sectionColors = ['red','orange', 'blue','green', 'black']
    
    def plotData(self, xData, yData, title, ax = None, axisLimits = [], topPeaks = {}, bottomPeaks = {}, peakSize = 3, lineWidth = 2, lineColor = "tab:blue", finalInd = []):
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
        if len(finalInd) > 0:
            for groupInd in range(len(self.sectionColors)):
                if finalInd[groupInd] in [np.nan, None] or finalInd[groupInd+1] in [np.nan, None]: 
                    continue
                ax.fill_between(xData[finalInd[groupInd]:finalInd[groupInd+1]+1], min(yData), yData[finalInd[groupInd]:finalInd[groupInd+1]+1], color=self.sectionColors[groupInd], alpha=0.15)
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
        fig.suptitle("Indivisual Pulse Peaks", fontsize=20, fontweight ="bold", y=0.98)
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
            filterData = bloodPulse[pulseNum]['smoothData']
            # Get the Pulse peaks
            bottomInd = []
            topInd = bloodPulse[pulseNum]['indicesTop']
            # Plot with Pulses Sectioned Off into Regions
            if finalPlot:
                finalInd = bloodPulse[pulseNum]['finalInd']
                self.plotData(time, filterData, title = "Peak Number " + str(pulseNum), ax = currentAxes, topPeaks = {1:time[topInd], 2:filterData[topInd]}, bottomPeaks = {1:time[bottomInd], 2:filterData[bottomInd]}, peakSize = 5,  lineWidth = 2, lineColor = "black", finalInd = finalInd)
            # General Plot
            else:
                # Plot the Data 
                self.plotData(time, filterData, title = "Peak Number " + str(pulseNum), ax = currentAxes, topPeaks = {1:time[topInd], 2:filterData[topInd]}, bottomPeaks = {1:time[bottomInd], 2:filterData[bottomInd]}, peakSize = 5, lineWidth = 2, lineColor = "black")
        fig.tight_layout(pad= 2.0)
        plt.show()
    
    
    def plotPulseNum(self, bloodPulse, pulseNum, finalPlot = False):
        # Get Data
        time = bloodPulse[pulseNum]['time']
        filterData = bloodPulse[pulseNum]['smoothData']
        # Get the Pulse peaks 
        bottomInd = []
        topInd = bloodPulse[pulseNum]['indicesTop']
        # Plot with Pulses Sectioned Off into Regions
        if finalPlot:
            finalInd = bloodPulse[pulseNum]['finalInd']
            self.plotData(time, filterData, title = "Peak Number " + str(pulseNum), topPeaks = {1:time[topInd], 2:filterData[topInd]}, bottomPeaks = {1:time[bottomInd], 2:filterData[bottomInd]}, peakSize = 5,  lineWidth = 2, lineColor = "black", finalInd = finalInd)
        else:
            self.plotData(time, filterData, title = "Peak Number " + str(pulseNum), topPeaks = {1:time[topInd], 2:filterData[topInd]}, bottomPeaks = {1:time[bottomInd], 2:filterData[bottomInd]}, peakSize = 3, lineWidth = 2, lineColor = "black")
    
    
    def seeFilterFit(self, bloodPulse, pulseNum):
        # Get the Data
        dataProcessing = signalProcessing([], [])
        pulseData = dataProcessing.normalizePulseBaseline(bloodPulse[pulseNum]['pulseData'], polynomialDegree = 1)
        smoothData = bloodPulse[pulseNum]['smoothData']
        firstDer = bloodPulse[pulseNum]["firstDer"]
        pulseTime = bloodPulse[pulseNum]['time']
        
        # Plot the Data
        plt.plot(pulseTime, pulseData, linewidth = 2, color = "black")
        plt.plot(pulseTime, smoothData, linewidth = 1, color = "tab:orange")
        plt.plot(pulseTime, firstDer/max(firstDer), linewidth = 1, color = "tab:red")
        plt.show()
        
# ---------------------------------------------------------------------------#
# ---------------------------------------------------------------------------#

class signalProcessing:
    
    def __init__(self):
        # Indivisial Pulse Information
        self.bloodPulse = {}      # Holder for Each Indivisual Pulse Data
        # Saving Final pulseNums: Well-Shaped Pulses for Machine Learning
        self.goodPulseNums = []
    
    
    def findLeftMinimum(self, smoothData, startInd):
        """
        ----------------------------------------------------------------------
        Input Parameters:
            smoothData:  y-Axis Data for Blood Pulse (Prefered Smoothen Filter Applied)
            startInd: Polynomials Used in Baseline Subtraction
        Output Parameters:
            startPulseInd: y-Axis Data Index for Local Minimum Before StartInd
        Use Case: Find the Start Index (Also the Previous Pulse's End Index) of a Pulse
        Assumption for Later Use: Minor Fluctuations in Pulse on the Sysolic Rise are Smoothened Out
        ----------------------------------------------------------------------
        """
        # Initiate Starting Point
        prevPoint = smoothData[startInd]
        # Loop Through Points Left of Starting Point to Find Minimum
        for devInd in range(1, startInd + 1):
            # Get the Left Adjacent Data Point
            nextInd = startInd - devInd
            dataPoint = smoothData[nextInd]
            
            # If the Left Adjacent Data Point is Greater Than the Right, It is a Minimum
            if dataPoint > prevPoint:
                # Get the Minimum's Index and Return it
                startPulseInd =  nextInd + 1
                return startPulseInd
            
            # Move Onto the Next Point
            prevPoint = dataPoint
        # Edge Case: The First Point is the Minimum
        return 0
    
    
    def window_rms(self, inputData, window_size):
        """
        ----------------------------------------------------------------------
        Input Parameters:
            inputData:  y-Axis Data for Blood Pulse (First Derivative)
            window_size: Size of Window to Take the Root Mean Squared
        Output Parameters:
            pulseRMS: Root Mean Squared of y-Axis Data
        Use Case: Increase the Gradient of the Systolic Peak to Differentiate it More
        Assumption for Later Use: The Window Size is Not too Big as to Average Everything
        ----------------------------------------------------------------------
        """
        dataSquared = np.power(inputData, 2)
        window = np.ones(window_size)/float(window_size)
        return np.sqrt(np.convolve(dataSquared, window, 'valid'))
    
    
    def normalizePulseBaseline(self, pulseData, polynomialDegree):
        """
        ----------------------------------------------------------------------
        Input Parameters:
            pulseData:  y-Axis Data for a Single Pulse (Start-End)
            polynomialDegree: Polynomials Used in Baseline Subtraction
        Output Parameters:
            pulseData: y-Axis Data for a Baseline-Normalized Pulse (Start, End = 0)
        Use Case: Shift the Pulse to the x-Axis (Removing non-Horizontal Base)
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
    
    
    def sepPulseAnalyze(self, time, signalData, plotSeperation = False):
        """
        ----------------------------------------------------------------------
        Input Parameters:
            time: x-Axis Data for the Blood Pulse (Seconds)
            signalData:  y-Axis Data for Blood Pulse (Capacitance)
            plotSeperation (Optional): Display the Indeces Identified as Around Mid-Sysolic Along with the Data
        Output Parameters:
            bloodPulse: Dictionary Hashed by the Pulse Number (Order it Appears) to Another Dictionary Containing Key Information about the Pulse
        Use Case: Seperate the Pulses, Label Peaks, Gaussian Decompositions, Return Information
        ----------------------------------------------------------------------
        """        
        # Smoothen Out the Data to Eliminate Small Peaks
        smoothData = savgol_filter(signalData, 5, 3)
        
        # Take First Derivative of Smoothened Data (Add Slight Smooting)
        pulseFit= splrep(time, smoothData, k = 5, s = 0.05) # k = [1,5], Choose 5 for Best Fit; s = Smoothing Factor
        firstDer = splev(time, pulseFit, der = 1)

        # Threshold that Defines the Slope of the R-Peak (Systolic Peak; First Peak)
        firstDerRMS = self.window_rms(firstDer, 7)
        bigPeakThreshold = heapq.nlargest(3, firstDerRMS)[2]/2
        # Find Mid-Ampltiude of the R-Peak (Systolic Peak; First Peak)
        risingPeaks = scipy.signal.find_peaks(firstDer, prominence=10E-2, height = bigPeakThreshold)[0]
        
        # If Questioning: Plot to See How the Pulses Seperated
        if plotSeperation:
            scaledData = signalData*max(abs(firstDer))/(max(signalData) - min(signalData))
            plt.plot(time, scaledData - np.mean(scaledData), label = "Centered + Scaled Signal Data", zorder = 3)
            plt.plot(time, firstDer, label = "First Derivative of Signal Data", zorder = 2)
            #plt.plot(time[0:len(firstDerRMS)], firstDerRMS, label = "Root Mean Squared of First Derivative", zorder = 1)
            plt.plot(time[risingPeaks], firstDer[risingPeaks], 'o', label = "Pulse Rise Identification")
            plt.legend(loc=9, bbox_to_anchor=(1.35, 1)); plt.show();
        
        # Seperate Peaks Based on the Minimim Before the R-Peak Rise
        pulseStartInd = self.findLeftMinimum(smoothData, risingPeaks[0])
        for pulseNum in range(1, len(risingPeaks) - 1):
            pulseEndInd = self.findLeftMinimum(smoothData, risingPeaks[pulseNum])
            
            self.bloodPulse[pulseNum] = {}
            # Store Initial Pulse Data
            self.bloodPulse[pulseNum]['pulseData'] = signalData[pulseStartInd:pulseEndInd+1]
            self.bloodPulse[pulseNum]['time'] = time[pulseStartInd:pulseEndInd+1]
            # Normalize Smooth Data Baseline to Zero and Store it Alongside first Derivative
            self.bloodPulse[pulseNum]["smoothData"] = self.normalizePulseBaseline(smoothData[pulseStartInd:pulseEndInd+1], polynomialDegree = 1)
            self.bloodPulse[pulseNum]["firstDer"] = firstDer[pulseStartInd:pulseEndInd+1]
            
            # Perform Peak Detection on Smooth Data
            minPeakIndSep = 3
            topInd = scipy.signal.find_peaks(self.bloodPulse[pulseNum]['smoothData'], distance = minPeakIndSep)[0]
            # Save Peak Indices (Indices will be For Indivisual Pulse Data)
            self.bloodPulse[pulseNum]['indicesTop'] = topInd
            
            # Save Final Peak Indices (Indices will be For Indivisual Pulse Data) and Gaussian Data
            lastInd = len(self.bloodPulse[pulseNum]['time']) - 1
            self.bloodPulse[pulseNum]["finalInd"] = np.array([0, *[None]*4, lastInd])
            self.bloodPulse[pulseNum]["finalIndGauss"] = np.array([0, *[None]*4, lastInd])
            self.bloodPulse[pulseNum]["finalAmpGauss"] = np.array([None]*4)
            # Label Systolic, Tidal Wave, Dicrotic, and Tail Wave Peaks Using Gaussian Decomposition    
            self.labelFinalPeaks(pulseNum, minPeakIndSep)
            
            # Hash Indices for Full DataSet
            self.bloodPulse[pulseNum]["dataHash"] = pulseStartInd, pulseEndInd
            
            # Reste for Next Pulse
            pulseNum += 1
            pulseStartInd = pulseEndInd
        
        return self.bloodPulse

    
    def labelFinalPeaks(self, pulseNum, minPeakIndSep = 3):
        
        # Get Relevant Data from the Pulse
        smoothData = self.bloodPulse[pulseNum]['smoothData']
        finalInd = self.bloodPulse[pulseNum]['finalInd']
        firstDer = self.bloodPulse[pulseNum]["firstDer"]
        topInd = self.bloodPulse[pulseNum]['indicesTop'].copy()
        
        # -------------------- Detect Systolic Peak ------------------------ #
        systolicPeak = max(topInd, key = lambda ind: smoothData[ind])
        finalInd[1] = systolicPeak
        topInd = topInd[topInd > systolicPeak]
        # ------------------------------------------------------------------ #
                
        # ------------------- Detect Tidal Wave Peak ----------------------- #
        # Detect Peaks in the First Derivative
        derivPeakInd = scipy.signal.find_peaks(firstDer, prominence = 10E-5, distance = minPeakIndSep)[0]
        derivPeakInd = derivPeakInd[derivPeakInd > systolicPeak + minPeakIndSep]
        
        # Initialize Possible Tidal Wave Peak
        tidalPeak = None
        # Check to See if Any Gradient Changes/Peaks are Found After the Systolic Peak
        if len(derivPeakInd) > 0 and len(topInd) > 0:
            # If So, Temporarily Label the First One Tidal Peak
            tidalPeak = min(derivPeakInd[0], topInd[0])
            # If Tidal Peak is a Maximum Instead of a Saddle, Choose the Maximum
            #betterTidal = topInd[abs(topInd - tidalPeak) <= 3]
            #if betterTidal:
            #    tidalPeak = betterTidal[0]
        # ------------------------------------------------------------------ #
        
        # --------  Detect/Distinguish the Tidal and Dicrotic Peak --------- #
        dicroticPeakInd = None
        # If There is Only One Recognized Top Peak After the Systolic Peak
        if len(topInd) == 1:
            # Assumption: If Any topInd is Found, it is Most Likely Dicrotic
            # If This Top Peak Was The Closest Peak to the Systolic Peak
            if tidalPeak and abs(topInd[0] - tidalPeak) < minPeakIndSep:
                # Then There Was No Tidal Wave Peak; It Was Actually Dicrotic
                tidalPeak = None
            # The Top Peak is a Dicrotic Peak
            dicroticPeakInd = topInd[0]
        
        # If There is More Than 1 topInd
        elif len(topInd) > 1:
            # Find the topInd After the Potential Tidal Wave
            topInd = topInd[topInd > tidalPeak + minPeakIndSep]
            # Label the Dicrotic Peak the Next Highest Peak
            dicroticPeakInd = max(topInd, key = lambda dicroticInd: smoothData[dicroticInd])

        finalInd[2] = tidalPeak
        finalInd[3] = dicroticPeakInd
        # ------------------------------------------------------------------ #
        
        # -------------------- Detect Tail Wave Peak ----------------------- #
        tailPeak = None
        # If There is a Dicrotic Peak, Look for Tail Peak (Else, It is Worthless to Check; Bad Pulse)
        if dicroticPeakInd:
            # Get Possible Tail Wave Peaks AFTER First Gradient Hump Thats AFTER Dicrotic
            derivTail = derivPeakInd[derivPeakInd > dicroticPeakInd + minPeakIndSep]
            if len(derivTail) > 0:
                tailPeaks = topInd[topInd > derivTail[0]]
            else:
                tailPeaks = topInd[topInd > dicroticPeakInd + minPeakIndSep]
            # Get the Maximum Tail Peak (Small Weight to Take One Further Out) and Label it Tail Peak
            tailPeak = max(tailPeaks, key = lambda tailInd: smoothData[tailInd] + tailInd/(6*finalInd[-1]), default=None)
            # If No topInd After the Dicrotic Peak, Use the Gradient to Find Tail Peak Hump
            if not tailPeak and len(derivPeakInd) > 0:
                tailPeak = max(derivTail, key = lambda tailInd: smoothData[tailInd] + tailInd/(6*finalInd[-1]), default=None)   
        
        finalInd[4] = tailPeak
        # ------------------------------------------------------------------ #
        
        # Save Final Indices
        self.bloodPulse[pulseNum]['finalInd'] = finalInd
        
        # If the Systolic, Dicrotic, and Tail Peaks were Found (Don't Need Tidal)
        if systolicPeak and dicroticPeakInd and tailPeak:
            # Perform Gaussian Decomposition on the Data
            self.gausDecomp(pulseNum)

    
    def gaussModel(self, x, amplitude, fwtm, center):
        sigma = fwtm/(2*math.sqrt(2*math.log(10)))
        return amplitude * np.exp(-(x-center)**2 / (2*sigma**2))
            
    
    def gausDecomp(self, pulseNum, addExtraGauss = False, plotFit = True):
        # https://lmfit.github.io/lmfit-py/builtin_models.html#example-1-fit-peak-data-to-gaussian-lorentzian-and-voigt-profiles
        
        # Get the Pulse Data (Start Pulse at Time = 0)
        x = self.bloodPulse[pulseNum]["time"] - self.bloodPulse[pulseNum]["time"][0]
        y = self.bloodPulse[pulseNum]["smoothData"]
        finalInd = self.bloodPulse[pulseNum]["finalInd"].copy()
        # Define Minimum Peak Width
        minWidth = 10E-5

        peakAmp = []; peakCenter = []; peakWidth = []
        # Extract Guesses About What the Peak Width, Center, and Amplitude Are
        for currentInd in range(1,5):
            peakInd = finalInd[currentInd]
            
            # If There Was No Tidal Pulse Detected
            if not peakInd:
                # Estimate the Pulse to be Between the Systolic and Dicrotic Peaks
                peakInd = int((finalInd[currentInd - 1] + finalInd[currentInd + 1])/2)   
            
            # Get the Peak's Amplitude and Center
            peakAmp.append(y[peakInd])
            peakCenter.append(x[peakInd])
            # Get the Peak's Width: Difference Between the Last Two Centers
            peakWidth.append(2*(peakCenter[currentInd-1] - peakCenter[currentInd-2]))
            
        
        # Systolic Peak Model
        gauss1 = Model(self.gaussModel, prefix = "g1_")
        pars = gauss1.make_params()
        pars['g1_center'].set(value = peakCenter[0], min = peakCenter[0]*.95, max = min(peakCenter[0]*1.05, peakCenter[1]))
        pars['g1_fwtm'].set(value = peakWidth[0], min = minWidth, max = peakCenter[3])
        pars['g1_amplitude'].set(value = peakAmp[0], min = peakAmp[0]*.95, max = peakAmp[0])
        
        # Tidal Wave Model
        gauss2 = Model(self.gaussModel, prefix = "g2_")
        pars.update(gauss2.make_params())
        pars['g2_center'].set(value = peakCenter[1], min = max(peakCenter[1]*.8, peakCenter[0]), max = min(peakCenter[1]*1.2, peakCenter[2]))
        pars['g2_fwtm'].set(value = peakWidth[1], min = minWidth, max = 1.1*(peakCenter[2] - peakCenter[0]))
        # Uncertain Paramaters for Tidal Wave Depending on if We Found One
        if finalInd[2]:
            pars['g2_amplitude'].set(value = peakAmp[1], min = peakAmp[1]*.8, max = peakAmp[1]*1.05)
        else:
            pars['g2_amplitude'].set(value = peakAmp[1]/2, min = 0, max = peakAmp[0]*.95)
        
        # Dicrotic Peak Model
        gauss3 = Model(self.gaussModel, prefix = "g3_")
        pars.update(gauss3.make_params())
        pars['g3_center'].set(value = peakCenter[2], min = peakCenter[2]*.9, max =min(peakCenter[2]*1.1, peakCenter[3]))
        pars['g3_fwtm'].set(value = peakWidth[2], min = minWidth, max = 2*(peakCenter[2] - peakCenter[0]))
        pars['g3_amplitude'].set(value = peakAmp[2], min = peakAmp[2]*.9, max = peakAmp[2]*1.02)
        
        # Tail Wave Model
        gauss4 = Model(self.gaussModel, prefix = "g4_")
        pars.update(gauss4.make_params())
        pars['g4_center'].set(value = peakCenter[3], min = peakCenter[2] + 0.5*(peakCenter[2]- peakCenter[1]), max = min(peakCenter[3]*1.1, x[-1]))
        pars['g4_fwtm'].set(value = x[-1] - peakCenter[3], min = minWidth, max = x[-1] - peakCenter[1])
        pars['g4_amplitude'].set(value = peakAmp[3], min = peakAmp[3]*.8, max = peakAmp[3]*1.2)
        
        # Add Models Together
        mod = gauss1 + gauss2 + gauss3 + gauss4
        # Add Extra Gaussian to Tail if The Previous Fit Was Bad
        if addExtraGauss:
            gauss5 = Model(self.gaussModel, prefix = "g5_")
            pars.update(gauss5.make_params())
            pars['g5_amplitude'].set(value = peakAmp[3]/6, min = 0, max = peakAmp[3]/2)
            pars['g5_fwtm'].set(value = x[-1] - peakCenter[3], min = 0, max = x[-1] - peakCenter[2])
            pars['g5_center'].set(value = peakCenter[3]*1.01, min = peakCenter[2] + (peakCenter[2] - peakCenter[1]), max = x[-1])
            mod += gauss5
        
        # Get Fit Information
        finalFitInfo = mod.fit(y, pars, x=x, method='powell')
        #fitReport = finalFitInfo.fit_report(min_correl=0.6); print(fitReport)
        
        # Calcluate Different RSquared Methods
        rSquared1 = 1 - finalFitInfo.residual.var() / np.var(y)
        rSquared2 = 1 - finalFitInfo.redchi / np.var(y, ddof=1)
        coefficient_of_dermination = r2_score(y, finalFitInfo.best_fit)
        # Statistics for Fit
        errorSQ = finalFitInfo.residual[2:-2]**2  # Ignore First/Last 2 Points (Bad EndPoint Fit Given Smoothing)
        meanErrorSQ = np.mean(errorSQ)       
        
        # Only Take Pulses with a Good Fit
        if rSquared1 > 0.97 and rSquared2 > 0.97 and coefficient_of_dermination > 0.97 and meanErrorSQ < 0.0005:
            # Keep Track of Good Pules
            self.goodPulseNums.append(pulseNum)
            # Extract Data From Gaussian's in Fit to Save
            comps = finalFitInfo.eval_components(x=x)
            for peakInd in range(1,5):
                # Save the Gaussian Center's Index and Amplitude
                self.bloodPulse[pulseNum]["finalIndGauss"][peakInd] = comps['g'+str(peakInd)+'_'].argmax()
                self.bloodPulse[pulseNum]["finalAmpGauss"][peakInd-1] = max(comps['g'+str(peakInd)+'_'])
            # If We Previously Missed the Tidal Wave, Use the Gaussian's Tidal Wave Index
            if not self.bloodPulse[pulseNum]["finalInd"][2]:
                self.bloodPulse[pulseNum]["finalInd"][2] = self.bloodPulse[pulseNum]["finalIndGauss"][2]
        # If Bad, Try and Add an Extra Gaussian to the Tail
        elif not addExtraGauss:
            self.gausDecomp(pulseNum, addExtraGauss = True)
            return None
        # If Still Bad, Throw Out the Pulse
        else:
            return None
        
        # Plot the Pulse with its Fit 
        if plotFit:
            finalInd = self.bloodPulse[pulseNum]["finalInd"]
            dely = finalFitInfo.eval_uncertainty(sigma=3)
     #       x = np.array(x)
    #        y = np.array(y)
            plt.plot(x, y, linewidth = 2, color = "black")
            plt.plot(x[finalInd.astype(int)], y[finalInd.astype(int)], 'o')
            plt.plot(x, comps['g1_'], '--', color = "tab:red", alpha = 0.8, label='Systolic Pulse')
            plt.plot(x, comps['g2_'], '--', color = "tab:green", alpha = 0.8, label='Tidal Wave Pulse')
            plt.plot(x, comps['g3_'], '--', color = "tab:blue", alpha = 0.8, label='Dicrotic Pulse')
            plt.plot(x, comps['g4_'], '--', color = "tab:purple", alpha = 0.8, label='Tail Wave Pulse')
            
            if addExtraGauss:
                plt.plot(x, comps['g5_'], '--', color = "tab:orange", alpha = 0.5, label='Extra Tail Pulse')
            
            plt.fill_between(x, finalFitInfo.best_fit-dely, finalFitInfo.best_fit+dely, color="#ABABAB",
                 label='3-$\sigma$ uncertainty band')
            
            plt.legend(loc='best')
            plt.title("Gaussian Decomposition of Pulse Number " + str(pulseNum))
            plt.show()




    
    
    
    
    