
# Basic Modules
import sys
import numpy as np
# Matlab Plotting API
import matplotlib as mpl
import matplotlib.pyplot as plt

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
            




    
    
    
    
    