
# Basic Modules
import math
import numpy as np
from scipy.stats.mstats import gmean
# Peak Detection
import scipy
import scipy.signal
# Baseline Subtraction
from BaselineRemoval import BaselineRemoval
# Gaussian Decomposition
from lmfit import Model
from sklearn.metrics import r2_score
# Matlab Plotting API
import matplotlib as mpl
import matplotlib.pyplot as plt
# Feature Extraction
from scipy.stats import skew
from scipy.stats import entropy
from scipy.stats import kurtosis

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
    
    
    def seeFilterFit(self, bloodPulse, pulseNum):
        # Get the Data
        dataProcessing = signalProcessing([], [])
        pulseData = dataProcessing.normalizePulseBaseline(bloodPulse[pulseNum]['pulseData'], polynomialDegree = 1)
        normalizedPulse = bloodPulse[pulseNum]["normalizedPulse"]
        firstDer = bloodPulse[pulseNum]["firstDer"]
        pulseTime = bloodPulse[pulseNum]['time']
        
        # Plot the Data
        plt.plot(pulseTime, pulseData, linewidth = 2, color = "black")
        plt.plot(pulseTime, normalizedPulse, linewidth = 1, color = "tab:orange")
        plt.plot(pulseTime, firstDer/max(firstDer), linewidth = 1, color = "tab:red")
        plt.show()
        
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
        # Storing Features
        self.featureList = []
        
        # Systolic and Diastolic References
        self.systolicPressure0 = None
        self.diastolicPressure0 = None
        
        # Program Flags
        self.plotGaussFit = plotGaussFit
        self.plotSeperation = plotSeperation
        self.alreadyFilteredData = alreadyFilteredData
        
        # Data Processing Parameters
        self.minGaussianWidth = 10E-4
        self.minPeakIndSep = 3
        
        self.pulseCounter = 0
        
    
    def sepPulseAnalyze(self, time, signalData, minBPM = 27, maxBPM = 480):
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
      
        # Estimate that Defines the Number of Points in a Pulse
        minPointsPerPulse = np.searchsorted(time, 60/maxBPM, side="left") + 1
        maxPointsPerPulse = np.searchsorted(time, 60/minBPM, side="left") + 1
        
        # Take First Derivative of Smoothened Data
        firstDer = [0]; peakStandard = 0
        risingPeaks = []; lastPeakRise = 0
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
                    if lastPeakRise + minPointsPerPulse < pointInd :
                        risingPeaks.append(pointInd)
                    # Else, Find the Max of the Peak
                    elif firstDer[risingPeaks[-1]] < firstDer[pointInd]:
                        risingPeaks[-1] = pointInd
                    lastPeakRise = pointInd
        
        # If Questioning: Plot to See How the Pulses Seperated
        if self.plotSeperation:
            risingPeaks = np.array(risingPeaks); firstDer = np.array(firstDer)
            scaledData = signalData*max(np.abs(firstDer))/(max(signalData) - min(signalData))
            plt.figure()
            plt.plot(time, scaledData - np.mean(scaledData), label = "Centered + Scaled Signal Data", zorder = 3)
            plt.plot(time, firstDer, label = "First Derivative of Signal Data", zorder = 2)
            plt.plot(time[risingPeaks], firstDer[risingPeaks], 'o', label = "Mid-Pulse Rise Identification")
            plt.legend(loc=9, bbox_to_anchor=(1.35, 1));
            plt.hlines(0,time[0], time[-1])
            #plt.xlim(0,20)
            plt.show()
        
        self.heartRateList = []
        self.heartRateList1 = []
        print("Analyzing Pulses")
        # Seperate Peaks Based on the Minimim Before the R-Peak Rise
        pulseStartInd = self.findNearbyMinimum(signalData, risingPeaks[0], binarySearchWindow=-1, maxPointsSearch=50)
        for pulseNum in range(1, len(risingPeaks) - 1):
            pulseEndInd = self.findNearbyMinimum(signalData, risingPeaks[pulseNum], binarySearchWindow=-1, maxPointsSearch=59)
            self.pulseCounter = pulseNum 
            
            # Calculate the Current Heart Rate
            minTime = max(0, time[pulseEndInd] - 60)
            maxTime = time[pulseEndInd]
            recentPulses = risingPeaks[time[risingPeaks] <= maxTime]
            recentPulses1 = recentPulses[time[recentPulses] - minTime >= 0]
            self.heartRate = 60*len(recentPulses1)/(maxTime-minTime)
            
            self.heartRateList.append(60*len(recentPulses1)/(maxTime-minTime))
            pulseTime = time[pulseStartInd:pulseEndInd+1] - time[pulseStartInd]
            self.heartRateList1.append(60/pulseTime[-1])
            
            self.timePoint = time[pulseEndInd]
            
            # ----------------------- Cull Bad Pulses ----------------------- #
            # Check if the Pulse is Too Big: Likely Double Pulse
            if pulseEndInd - pulseStartInd > maxPointsPerPulse:
                pulseStartInd = pulseEndInd; continue
            # Check if the Pulse is Too Small; Likely Not an R-Peak
            elif pulseEndInd - pulseStartInd < minPointsPerPulse:
                pulseStartInd = pulseEndInd; continue
            # --------------------------------------------------------------- #

            # -------------------- Seperate Out the Pulse ------------------- #
            # Extract Indivisual Pulse Data
            pulseData = signalData[pulseStartInd:pulseEndInd+1]
            pulseTime = time[pulseStartInd:pulseEndInd+1] - time[pulseStartInd]
            pulseVelocity = np.gradient(pulseData, pulseTime)
            pulseAcceleration = np.gradient(pulseVelocity, pulseTime)
            # Normalize the Pulse's Baseline to Zero
            if self.alreadyFilteredData:
                normalizedPulse = signalData[pulseStartInd:pulseEndInd+1]
            else:
                normalizedPulse = self.normalizePulseBaseline(signalData[pulseStartInd:pulseEndInd+1], polynomialDegree = 1)
            # Calculate Diastolic and Systolic Reference for the First Pulse
            if not self.diastolicPressure0:
                self.diastolicPressure0 = pulseData[0]
                self.systolicPressure0 = max(pulseData)
            #print(60/pulseTime[-1], self.heartRate)
            # --------------------------------------------------------------- #
            
            # -------------------- Extract Pulse Features ------------------- #
            # Label Systolic, Tidal Wave, Dicrotic, and Tail Wave Peaks Using Gaussian Decomposition   
           # import time as time1
           # t1 = time1.time()
            self.labelPulsePeakInds(pulseTime, normalizedPulse, pulseVelocity, pulseAcceleration)
           # t2 = time1.time()
           # print(t2-t1, "\n")
            # --------------------------------------------------------------- #
            #return pulseTime, normalizedPulse, pulseVelocity
            
            # Reste for Next Pulse
            pulseStartInd = pulseEndInd
    
    def labelPulsePeakInds(self, pulseTime, normalizedPulse, pulseVelocity, pulseAcceleration):
        # spl = UnivariateSpline(pulseTime, normalizedPulse, k=5, s=0.001)
        # spl.derivative(n=2)(np.arange(0,0.7,.00001))
        
        # -------------------- Detect Systolic Peak ------------------------ #
        # Perform Initial Peak Detection on Normalized Data      
        peakInds = scipy.signal.find_peaks(normalizedPulse, distance = self.minPeakIndSep)[0]
        
        # The Maximum Peak is the Systolic Peak
        #systolicPeakInd = self.findNearbyMaximum(normalizedPulse, 0, binarySearchWindow = 2, maxPointsSearch = int(len(pulseTime)/2))
        systolicPeakInd = np.argmax(normalizedPulse[0:int(len(normalizedPulse)/3)])
        # ------------------------------------------------------------------ #
                
        # ------------------- Detect Tidal Wave Peak ----------------------- #
        # Detect Peaks in the First Derivative
        velPeakInds = scipy.signal.find_peaks(pulseVelocity[systolicPeakInd + 2:], prominence = 10E-5, distance = self.minPeakIndSep)[0]
        accelPeakInds = scipy.signal.find_peaks(pulseAcceleration[systolicPeakInd + 2:], prominence = 10E-5, distance = self.minPeakIndSep)[0]
        
        # Initialize Possible Tidal Wave Peak
        tidalPeakInd = None
        # Check to See if Any Gradient Changes/Peaks are Found After the Systolic Peak
        if len(velPeakInds) > 0:
            # If So, Temporarily Label the First One Tidal Peak
            derivPeakMax = self.findRightMaximum(pulseAcceleration, systolicPeakInd+2, searchWindow = int(len(pulseTime)/2))
            tidalPeakInd = self.findNearbyMinimum(pulseAcceleration, derivPeakMax, binarySearchWindow=1, maxPointsSearch=10)
        # ------------------------------------------------------------------ #
        
        # --------  Detect/Distinguish the Tidal and Dicrotic Peak --------- #
        dicroticPeakInd = None
        # If There is Only One Recognized Top Peak After the Systolic Peak
        if len(peakInds) == 1:
            # Assumption: If Any peakInds is Found, it is Most Likely Dicrotic
            # If This Top Peak Was The Closest Peak to the Systolic Peak
            if tidalPeakInd and abs(peakInds[0] - tidalPeakInd) < self.minPeakIndSep:
                # Then There Was No Tidal Wave Peak; It Was Actually Dicrotic
                tidalPeakInd = None
            # The Top Peak is a Dicrotic Peak
            dicroticPeakInd = peakInds[0]
        
        # If There is More Than 1 peakInds
        elif len(peakInds) > 1:
            # Find the peakInds After the Potential Tidal Wave
            if tidalPeakInd:
                peakIndsHold = peakInds[peakInds > tidalPeakInd + self.minPeakIndSep]
            # Label the Dicrotic Peak the Next Highest Peak
            if len(peakIndsHold) != 0:
                peakInds = peakIndsHold
                dicroticPeakInd = max(peakInds, key = lambda dicroticInd: normalizedPulse[dicroticInd])
            # If No Next highest Peak After the Tidal ... It was Dicrotic NOT Tidal
            else:
                dicroticPeakInd = max(peakInds)
                tidalPeakInd = None
                peakInds = np.array([])
        # ------------------------------------------------------------------ #
        
        # -------------------- Detect Tail Wave Peak ----------------------- #
        tailPeak1 = None
        tailPeak2 = None
        # If There is a Dicrotic Peak, Look for Tail Peak (Else, It is Worthless to Check; Bad Pulse)
        if dicroticPeakInd:
            # Get Possible Tail Wave Peaks AFTER First Gradient Hump Thats AFTER Dicrotic
            derivTail = velPeakInds[velPeakInds > dicroticPeakInd + self.minPeakIndSep]
            if len(derivTail) > 0:
                tailPeaks = peakInds[peakInds > derivTail[0]]
            else:
                tailPeaks = peakInds[peakInds > dicroticPeakInd + self.minPeakIndSep]
            # Get the Maximum Tail Peak (Small Weight to Take One Further Out) and Label it Tail Peak
            tailPeak1 = max(tailPeaks, key = lambda tailInd: normalizedPulse[tailInd] + tailInd/(6*peakInds[-1]), default=None)
            # If No peakInds After the Dicrotic Peak, Use the Gradient to Find Tail Peak Hump
            if not tailPeak1 and len(peakInds) != 0:
                tailPeak1 = max(derivTail, key = lambda tailInd: normalizedPulse[tailInd] + tailInd/(6*peakInds[-1]), default=None) 
            elif tailPeak1:
                tailPeaks = tailPeaks[tailPeak1 < tailPeaks]
                if len(tailPeaks) != 0:
                    tailPeak2 = tailPeaks[0]
        # ------------------------------------------------------------------ #
        
        pulsePeakInds = [0,systolicPeakInd, tidalPeakInd, dicroticPeakInd, tailPeak1, len(pulseTime)-1]
        # If the Systolic, Dicrotic, and Tail Peaks were Found (Don't Need Tidal)
        if None not in pulsePeakInds:
            # Perform Gaussian Decomposition on the Data
            pulsePeakInds, gaussPeakInds, gaussPeakAmps = self.gausDecomp(pulseTime, normalizedPulse, pulsePeakInds, addExtraGauss = tailPeak2 != None)
            if len(pulsePeakInds) != 0:
                self.extractFeatures(normalizedPulse, pulseTime, pulseVelocity, pulseAcceleration, pulsePeakInds, gaussPeakInds, gaussPeakAmps)

    
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
        pars['g1_center'].set(value = peakCenter[0], min = peakCenter[0]*.95, max = min(peakCenter[0]*1.05, peakCenter[1]))
        pars['g1_fwtm'].set(value = peakWidth[0], min = self.minGaussianWidth, max = peakCenter[3])
        pars['g1_amplitude'].set(value = peakAmp[0], min = peakAmp[0]*.95, max = peakAmp[0])
        
        # Tidal Wave Model
        gauss2 = Model(self.gaussModel, prefix = "g2_")
        pars.update(gauss2.make_params())
        pars['g2_center'].set(value = peakCenter[1], min = max(peakCenter[1]*.8, peakCenter[0]), max = min(peakCenter[1]*1.2, peakCenter[2]))
        pars['g2_fwtm'].set(value = peakWidth[1], min = self.minGaussianWidth, max = 1.1*(peakCenter[2] - peakCenter[0]))
        # Uncertain Paramaters for Tidal Wave Depending on if We Found One
        if pulsePeakInds[2]:
            pars['g2_amplitude'].set(value = peakAmp[1], min = peakAmp[1]*.8, max = peakAmp[1]*1.05)
        else:
            pars['g2_amplitude'].set(value = peakAmp[1]/2, min = 0, max = peakAmp[0]*.95)
        
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
            plt.title("Gaussian Decomposition of Pulse Number " + str(self.pulseCounter))
            plt.show()
        
        #comps = finalFitInfo.eval_components(xData=xData)
        #plotGaussianFit(xData, yData, pulsePeakInds)
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
            return pulsePeakInds, gaussPeakInds, gaussPeakAmps
        # If Bad, Try and Add an Extra Gaussian to the Tail
        elif not addExtraGauss:
            return self.gausDecomp(xData, yData, pulsePeakInds, addExtraGauss = True)
        # If Still Bad, Throw Out the Pulse
        return [], [], []

    def extractFeatures(self, normalizedPulse, pulseTime, pulseVelocity, pulseAcceleration, pulsePeakInds, gaussPeakInds, gaussPeakAmps):
        
        # Unpack Pulse Peaks
        _, systolicPeakInd, tidalPeakInd, dicroticPeakInd, tailPeakInd, _ = pulsePeakInds
        _, systolicPeakTime, tidalPeakTime, dicroticPeakTime, tailPeakTime, _ = pulseTime[pulsePeakInds]
        _, systolicPeakAmp, tidalPeakAmp, dicroticPeakAmp, tailPeakAmp, _ = normalizedPulse[pulsePeakInds]
        _, systolicPeakVel, tidalPeakVel, dicroticPeakVel, tailPeakVel, _ = pulseVelocity[pulsePeakInds]
        _, systolicPeakAccel, tidalPeakAccel, dicroticPeakAccel, tailPeakAccel, _ = pulseAcceleration[pulsePeakInds]
        # Unpack Gaussian Peaks
        systolicGaussInd, tidalGaussInd, dicroticGaussInd, tailGaussInd = gaussPeakInds
        systolicGaussAmp, tidalGaussAmp, dicroticGaussAmp, tailGaussAmp = gaussPeakAmps
        # Get the Dicrotic Notch
        dicroticNotchInd = self.findNearbyMinimum(normalizedPulse, dicroticPeakInd, binarySearchWindow = -1, maxPointsSearch = dicroticPeakInd - tidalPeakInd)
        dicroticNotchTime = pulseTime[dicroticNotchInd]
        dicroticNotchAmp = normalizedPulse[dicroticNotchInd]
        dicroticNotchVel = pulseVelocity[dicroticNotchInd]
        dicroticNotchAccel = pulseAcceleration[dicroticNotchInd]
        
        # Diastole and Systole Parameters
        pulseDuration = pulseTime[-1]
        systolicTime = pulseTime[dicroticNotchInd]
        DiastolicTime = pulseDuration - systolicTime
        leftVentricularPerformance = systolicTime/DiastolicTime
        
        # Calculate the Area Under the Curve
        pulseArea = scipy.integrate.simpson(normalizedPulse, pulseTime)
        pulseAreaSquared = scipy.integrate.simpson(normalizedPulse**2, pulseTime)
        leftVentricleLoad = scipy.integrate.simpson(normalizedPulse[0:dicroticNotchInd], pulseTime[0:dicroticNotchInd])
        diastolicArea = pulseArea - leftVentricleLoad
        # Average of the Pulse
        geometricMean = gmean(normalizedPulse)
        pulseAverage = np.mean(normalizedPulse)
        
        # Systole and Diastole Ratios
        areaRatio = leftVentricleLoad/diastolicArea
        systolicDicroticNotchAmpRatio = dicroticNotchAmp/systolicPeakAmp
        systolicDicroticNotchVelRatio = dicroticNotchVel/systolicPeakVel
        systolicDicroticNotchAccelRatio = dicroticNotchAccel/systolicPeakAccel
        
        # Other Systolic Amplitude Ratios
        systolicTidalAmpRatio = tidalPeakAmp/systolicPeakAmp
        systolicDicroticAmpRatio = dicroticPeakAmp/systolicPeakAmp
        systolicTailAmpRatio = tailPeakAmp/systolicPeakAmp
        # Other Diastole Ratios
        dicroticNotchTidalAmpRatio = tidalPeakAmp/dicroticNotchAmp
        dicroticNotchDicroticAmpRatio = dicroticPeakAmp/dicroticNotchAmp
        dicroticNotchTailAmpRatio = tailPeakAmp/dicroticNotchAmp
        
        # Systolic Velocty Ratios
        systolicTidalVelRatio = tidalPeakVel/systolicPeakVel
        systolicDicroticVelRatio = dicroticPeakVel/systolicPeakVel
        systolicTailVelRatio = tailPeakVel/systolicPeakVel
        # Diastole Velocity Ratios
        dicroticNotchTidalVelRatio = tidalPeakVel/dicroticNotchVel
        dicroticNotchDicroticVelRatio = dicroticPeakVel/dicroticNotchVel
        dicroticNotchTailVelRatio = tailPeakVel/dicroticNotchVel        
        
        # Systolic Acceleration Ratios
        systolicTidalAccelRatio = tidalPeakAccel/systolicPeakAccel
        systolicDicroticAccelRatio = dicroticPeakAccel/systolicPeakAccel
        systolicTailAccelRatio = tailPeakAccel/systolicPeakAccel
        # Diastole Acceleration Ratios
        dicroticNotchTidalAccelRatio = tidalPeakAccel/dicroticNotchAccel
        dicroticNotchDicroticAccelRatio = dicroticPeakAccel/dicroticNotchAccel
        dicroticNotchTailAccelRatio = tailPeakAccel/dicroticNotchAccel  
        
        # Time from Systolic Peak
        systolicToTidal = tidalPeakTime - systolicPeakTime
        systolicToDicroticNotch = dicroticNotchTime - systolicPeakTime
        systolicToDicrotic = dicroticPeakTime - systolicPeakTime
        systolicToTail = tailPeakTime - systolicPeakTime
        # Time from Dicrotic Notch
        dicroticNotchToTidal = tidalPeakTime - dicroticNotchTime
        dicroticNotchToDicrotic = dicroticPeakTime - dicroticNotchTime
        dicroticNotchToTail = tailPeakTime - dicroticNotchTime

        # Pulse Entropy
        pulseEntropy = entropy(normalizedPulse + 10E-50)
        systoleEntropy = entropy(normalizedPulse[0:dicroticNotchInd]+10E-50)
        diastoleEntropy = entropy(normalizedPulse[dicroticNotchInd:]+10E-50)
        # Velocity Entropy
        pulseVelEntropy = entropy(pulseVelocity + 10E-50)
        systoleVelEntropy = entropy(pulseVelocity[0:dicroticNotchInd]+10E-50)
        diastoleVelEntropy = entropy(pulseVelocity[dicroticNotchInd:]+10E-50)
        # Acceleration Entropy
        pulseAccelEntropy = entropy(pulseAcceleration + 10E-50)
        systoleAccelEntropy = entropy(pulseAcceleration[0:dicroticNotchInd]+10E-50)
        diastoleAccelEntropy = entropy(pulseAcceleration[0:dicroticNotchInd:]+10E-50)
        
        systolicPeakInd, tidalPeakInd, dicroticPeakInd, tailPeakInd
        # Peak Slopes
        systolicSlopeUp = np.polyfit(pulseTime[1: systolicPeakInd-1], normalizedPulse[1: systolicPeakInd-1], 1)[0]
        SystolicSlopeDown = np.polyfit(pulseTime[systolicPeakInd+1:tidalPeakInd-1], normalizedPulse[systolicPeakInd+1:tidalPeakInd-1], 1)[0]
        tidalSlope = np.polyfit(pulseTime[tidalPeakInd-1:tidalPeakInd+1], normalizedPulse[tidalPeakInd-1:tidalPeakInd+1], 1)[0]
        DicroticNotchSlopeLeft = np.polyfit(pulseTime[tidalPeakInd:dicroticNotchInd], normalizedPulse[tidalPeakInd:dicroticNotchInd], 1)[0]
        DicroticNotchSlopeRight = np.polyfit(pulseTime[dicroticNotchInd:dicroticPeakInd], normalizedPulse[dicroticNotchInd:dicroticPeakInd], 1)[0]
        tailSlope =  np.polyfit(pulseTime[dicroticPeakInd:], normalizedPulse[dicroticPeakInd:], 1)[0]
        tailSlope2 =  np.polyfit(pulseTime[tailPeakInd:], normalizedPulse[tailPeakInd:], 1)[0]
        
        # Find the Diastolic and Systolic Pressure
        diastolicPressure = self.diastolicPressure0
        systolicPressure = self.diastolicPressure0 + (systolicPeakAmp - normalizedPulse[0])
        pressureRatio = systolicPressure/diastolicPressure
        pulsePressure = systolicPressure - diastolicPressure

        momentumDensity = 2*pulseTime[-1]*pulseArea
        meanArterialBloodPressure = diastolicPressure + pulsePressure/3
        pseudoCardiacOutput = pulseArea/pulseTime[-1]
        pseudoCardiacOutput2 = 1/pulseArea
        pseudoSystemicVascularResistance = meanArterialBloodPressure/pulseTime[-1]
        pseudoStrokeVolume = pseudoCardiacOutput/pulseTime[-1]
        
        maxSystolicVelocity = max(pulseVelocity)
        pseudoVelocityAv = 1/pulseTime[-1]
        valveCrossSectionalArea = pseudoCardiacOutput/pseudoVelocityAv
        velocityTimeIntegral = scipy.integrate.simpson(pulseVelocity, pulseTime)
        velocityTimeIntegralABS = scipy.integrate.simpson(abs(pulseVelocity), pulseTime)
        velocityTimeIntegral_ALT = pseudoStrokeVolume/valveCrossSectionalArea
        leafletBaseLength = np.sqrt(valveCrossSectionalArea/0.433)
        valveRadius = np.sqrt(valveCrossSectionalArea/math.pi)
        
        # Time to Max Deriv
        # Time from Max Deriv to Systolic Peak
        
        # Add Index Parameters: https://www.vitalscan.com/dtr_pwv_parameters.htm
        centralAugmentationIndex = normalizedPulse[np.argmax(pulseVelocity)]/systolicPeakAmp  # Tidal Peak / Systolic Max Vel yData
        centralAugmentationIndex_EST = tidalPeakAmp/systolicPeakAmp  # Tidal Peak / Systolic Max Vel yData
        reflectionIndex = dicroticPeakAmp/systolicPeakAmp  # Dicrotic Peak / Systolic Peak
        stiffensIndex = 1/(dicroticPeakTime - systolicPeakTime)  # 1/ Time from the Systolic to Dicrotic Peaks
        
        # Save Diastolic and Systolic Pressure
        pulseFeatures = [self.timePoint]
        pulseFeatures.extend([systolicPeakTime, tidalPeakTime, dicroticPeakTime, tailPeakTime])
        pulseFeatures.extend([systolicPeakAmp, tidalPeakAmp, dicroticPeakAmp, tailPeakAmp])
        pulseFeatures.extend([systolicPeakVel, tidalPeakVel, dicroticPeakVel, tailPeakVel])
        pulseFeatures.extend([systolicPeakAccel, tidalPeakAccel, dicroticPeakAccel, tailPeakAccel])
        pulseFeatures.extend([systolicGaussAmp, tidalGaussAmp, dicroticGaussAmp, tailGaussAmp])
        pulseFeatures.extend([dicroticNotchInd, dicroticNotchTime, dicroticNotchAmp, dicroticNotchVel, dicroticNotchAccel])
        pulseFeatures.extend([pulseDuration, systolicTime, DiastolicTime, leftVentricularPerformance])
        pulseFeatures.extend([pulseArea, pulseAreaSquared, leftVentricleLoad, diastolicArea, geometricMean, pulseAverage])
        pulseFeatures.extend([areaRatio, systolicDicroticNotchAmpRatio, systolicDicroticNotchVelRatio, systolicDicroticNotchAccelRatio])
        pulseFeatures.extend([systolicTidalAmpRatio, systolicDicroticAmpRatio, systolicTailAmpRatio, dicroticNotchTidalAmpRatio, dicroticNotchDicroticAmpRatio, dicroticNotchTailAmpRatio])
        pulseFeatures.extend([systolicTidalVelRatio, systolicDicroticVelRatio, systolicTailVelRatio, dicroticNotchTidalVelRatio, dicroticNotchDicroticVelRatio, dicroticNotchTailVelRatio])
        pulseFeatures.extend([systolicTidalAccelRatio, systolicDicroticAccelRatio, systolicTailAccelRatio, dicroticNotchTidalAccelRatio, dicroticNotchDicroticAccelRatio, dicroticNotchTailAccelRatio])
        pulseFeatures.extend([systolicToTidal, systolicToDicroticNotch, systolicToDicrotic, systolicToTail, dicroticNotchToTidal, dicroticNotchToDicrotic, dicroticNotchToTail])
        pulseFeatures.extend([pulseEntropy, systoleEntropy, diastoleEntropy, pulseVelEntropy, systoleVelEntropy, diastoleVelEntropy, pulseAccelEntropy, systoleAccelEntropy, diastoleAccelEntropy])
        pulseFeatures.extend([diastolicPressure, systolicPressure, pressureRatio, pulsePressure])
        pulseFeatures.extend([momentumDensity, meanArterialBloodPressure, pseudoCardiacOutput, pseudoCardiacOutput2, pseudoSystemicVascularResistance, pseudoStrokeVolume])
        pulseFeatures.extend([maxSystolicVelocity, pseudoVelocityAv, valveCrossSectionalArea, velocityTimeIntegral, velocityTimeIntegralABS, velocityTimeIntegral_ALT, leafletBaseLength, valveRadius])
        pulseFeatures.extend([centralAugmentationIndex, centralAugmentationIndex_EST, reflectionIndex, stiffensIndex])
        pulseFeatures.extend([systolicSlopeUp, SystolicSlopeDown, tidalSlope, DicroticNotchSlopeLeft, DicroticNotchSlopeRight, tailSlope, tailSlope2])

        # Save the Pulse Features
        self.featureList.append(pulseFeatures)

    def findNearbyMinimum(self, data, xPointer, binarySearchWindow = 5, maxPointsSearch = 500):
        """
        Search Right: binarySearchWindow > 0
        Search Left: binarySearchWindow < 0
        """
        # Base Case
        if abs(binarySearchWindow) < 1 or maxPointsSearch == 0:
            return xPointer
        
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
        if abs(binarySearchWindow) < 1 or maxPointsSearch == 0:
            return xPointer
        
        minHeight = data[xPointer]; searchDirection = binarySearchWindow//abs(binarySearchWindow)
        # Binary Search Data to Find the Minimum (Skip Over Minor Fluctuations)
        for dataPointer in range(max(xPointer, 0), max(0, min(xPointer + searchDirection*maxPointsSearch, len(data))), binarySearchWindow):
            # If the Next Point is Greater Than the Previous, Take a Step Back
            if data[dataPointer] < minHeight:
                return self.findNearbyMaximum(data, dataPointer - binarySearchWindow, round(binarySearchWindow/2), maxPointsSearch - searchDirection*(abs(dataPointer - binarySearchWindow)) - xPointer)
            # Else, Continue Searching
            else:
                minHeight = data[dataPointer]
    
        # If Your Binary Search is Too Large, Reduce it
        return self.findNearbyMaximum(data, xPointer, round(binarySearchWindow/2), maxPointsSearch-1)
    
    
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
        #possibleRisingPeaks = scipy.signal.find_peaks(firstDer, prominence = .9, distance = minPointsPerPulse, width=2)[0]
        # Cull Bad Pulse Indices
        # lastGoodInd = 1; risingPeaks = []; gotGoodPeak = False;
        # for currentPlace, peakCompareInd in enumerate(possibleRisingPeaks):
        #     peakStandard = firstDer[possibleRisingPeaks[lastGoodInd]]
        #     peakCompare = firstDer[peakCompareInd]
        #     # Good Peak
        #     if peakCompare > peakStandard * 0.5:
        #         risingPeaks.append(peakCompareInd)
        #         lastGoodInd = currentPlace
        #         gotGoodPeak = True
        #     elif not gotGoodPeak:
        #         print("Here")
        #         lastGoodInd += 1
"""
    
    
    
    
    