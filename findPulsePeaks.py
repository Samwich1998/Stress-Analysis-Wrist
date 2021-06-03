"""
    Written by Samuel Solomon
    
    --------------------------------------------------------------------------
    Program Description:
    
    Perform signal processing to filter blood pulse peaks. After filtering the data,
    perform peak detection and seperate the pulss from each other. Extract peaks
    from each pulse and record the data in an excel spreadsheet. 
    
    --------------------------------------------------------------------------
    
    Modules to Import Before Running the Program (Some May be Missing):
        %conda install matplotlib
        %conda install openpyxl
        %conda install numpy
        %pip install heartpy
        %pip install neurokit2
        %pip install pyexcel
        %pip install pyexcel-xls
        %pip install pyexcel-xlsx;
        %pip install BaselineRemoval
        %pip install peakutils
        %pip install lmfit
        %pip install findpeaks
        
    --------------------------------------------------------------------------
"""


# Basic Modules
import os
import sys
# Extra Options to Extract Signal Information
import heartpy as hp
import neurokit2 as nk
# Import Python Helper Files (And Their Location)
sys.path.append('./Helper Files/')  # Folder with All the Helper Files
import excelProcessing
import peakAnalysis



if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #
    
    # Specify the Location of the Input Data (Excel File: .XLSX or .XLS)
    testDataExcelFile = "./Input Data/Changhao 0528/1-Rest_Changhao_20210528.xls" # Path to the Excel Data ('.xls' or '.xlsx')
    testSheetNum = 0 # The Sheet/Tab Order (Zeroth/First/Second/Third) on the Bottom of the Excel Document
    
    # Optional Parameters
    plotSeperation = False
    plotGaussFit = False
    
    # Take the Average of Pulse Features in a Certain Time Frame
    combinePulses = False # Reduce Signal Features to One Feature Per pulsePerInterval
    pulsePerInterval = 3 # The Number of Seconds for Each Signal. Ex: [0, 4.99999] for pulsePerInterval = 5
    
    # Saves the Data Analysis: Peak Features for Each Well-Shaped Pulse
    saveInputData = True   
    if saveInputData:
        saveDataFolder = "./Output Data/20210510/"      # Data Folder to Save the Data; MUST END IN '/'
        sheetName = "Blood Pulse Data"                   # If SheetName Already Exists, Excel Will Add 1 to the end (The Copy Number) 
      
    # ---------------------------------------------------------------------- #
    #                   Extract Pulse Peak Data from Signals                 #
    # ---------------------------------------------------------------------- #

    # Read Data from Excel
    excelData = excelProcessing.excelProcessing()
    time, signalData = excelData.getData(testDataExcelFile, testSheetNum)
    signalData = signalData*10**12 # Get Data into pico-Farad
    
    # Plot the Initial Input Data
    plot = peakAnalysis.plot()
    plot.plotData(time, signalData, title = "Input ECG Data")
    
    # Seperate Pulses and Perform Indivisual Analysis
    dataProcessing = peakAnalysis.signalProcessing()
    bloodPulse = dataProcessing.sepPulseAnalyze(time, signalData, minBPM = 30, maxBPM = 220, 
                                plotSeperation = plotSeperation, plotGaussFit = plotGaussFit)
    
    if combinePulses:
        savingDict, savingPulseInd = dataProcessing.combinePulses(pulsePerInterval)
    else:
        savingDict = bloodPulse
        savingPulseInd = dataProcessing.goodPulseNums
    
    # Plot a Specific Pulse
    plotPulse = False
    if plotPulse:
        # Specify Number of Plots and Figure Style
        maxPulsesPlot = 9; numSubPlotsX = 3;
        figWidth = 25; figHeight = 13;
        # Create One Plot with Up to First 'maxPulsesPlot' Pulse Curves
        firstPeakPlotting = 1
        plot.plotPulses(bloodPulse, numSubPlotsX, firstPeakPlotting, maxPulsesPlot, figWidth, figHeight, finalPlot = True)
        
        # Plot a Specific Pulse
        pulseNum = 4
        plot.plotPulseNum(bloodPulse, pulseNum = pulseNum, finalPlot = True)
        
    # Save Pulse Labels (if Desired)
    if saveInputData:
        saveExcelName = os.path.basename(testDataExcelFile).split(".")[0] + ".xlsx"
        excelData.saveResults(savingDict, savingPulseInd, saveDataFolder, saveExcelName, sheetName)
    
    
    # ---------------------------------------------------------------------- #
    #                    Extract Body Parameters from Signals                #
    # ---------------------------------------------------------------------- #
    # Not Fully Developed Yet (Accuracy NOT Tested), But Can be Useful in the Future
    getBodyParam = False
    if getBodyParam:
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting: DONT REMOVE DESPITE WARNING
        
        # Get Corrected Baseline Data
        filterData = []
        for pulseNum in bloodPulse:
            filterData.extend(list(bloodPulse[pulseNum]["smoothData"][:-1]))
        filterData = np.array(filterData)
        # Get Sampling Rate
        samplingRate = hp.get_samplerate_mstimer(time)/1000 # Calculate the Sampling Rate
        
        # Data Analysis: User Parameters from the Data
        working_data, measures = hp.process(filterData, sample_rate = samplingRate, high_precision=True, windowsize=2)
        hp.plotter(working_data, measures, show=True, title='Heart Beat Detection on Noisy Signal', moving_average=True)
        #display measures computed
        for measure in measures.keys():
            print('%s: %f' %(measure, measures[measure]))
        
        # Show Complexity of the Signal
        parameters = nk.complexity_optimize(filterData, show=True)

