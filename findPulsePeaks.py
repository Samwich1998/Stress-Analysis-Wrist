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
        %pip install scikit-image
        
    --------------------------------------------------------------------------
"""


# Basic Modules
import os
import sys
import numpy as np
# Machine Learning Modules
import collections
from sklearn.model_selection import train_test_split
# Import Python Helper Files (And Their Location)
sys.path.append('./Helper Files/Data Aquisition and Analysis/')  # Folder with All the Helper Files
import excelProcessing
import pulseAnalysis
import gsrAnalysis
# Import Machine Learning Files (And They Location)
sys.path.append("Helper Files/Machine Learning Modules/")
import ANN
import KNN                  # Functions for K-Nearest Neighbors' Algorithm
import SVM                  # Functions for Support Vector Machines Algorithm
import randomForest
import LogisticRegression   # Functions for Logistic Regression's Algorithm


if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #

    # Specify Which Program to Run; All Can be Run in One Scirpt (NOT Simutaneously Yet)
    analyzePulse = True
    analyzeGSR = False
    trainModel = False
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # Pulse Parameters
    if analyzePulse:
        # Specify the Location of the Input Data
        pulseExcelFile = "./Input Data/Pulse Data/stroop_01.xlsx" # Path to the Excel Data ('.xls' or '.xlsx')
        # Required Parameters
        diastolicCapacitance = 112  # This Represents the Diastolic Pressure; We Are Assuming it is Constant Throughout the Experiment
        systolicCapacitance = 65    # This Represents the Systolic Pressure; We Will Use This Value as a Baseline for Other Systolic Pressures
        # OPTIONAL Parameters to Visualize the Pulse Data
        plotSeperation = False
        plotGaussFit = False
        # Average Nearby Pulses (After the Data is Collected)
        combinePulses = False # Reduce Signal Features to One Feature Per pulsePerInterval
        if combinePulses:
            pulsePerInterval = 3  # The Number of Seconds for Each Signal. Ex: [0, 4.99999] for pulsePerInterval = 5
        # Saves the Data Analysis: Peak Features for Each Well-Shaped Pulse
        saveInputData = True   
        if saveInputData:
            saveDataFolder = "./Output Data/Pulse Data/"      # Data Folder to Save the Data; MUST END IN '/'
            sheetName = "Blood Pulse Data"                   # If SheetName Already Exists, Excel Will Add 1 to the end (The Copy Number) 
    
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # GSR Parameters
    if analyzeGSR:
        # Specify the Location of the Input Data
        gsrFile = "./Input Data/Galvanic Skin Response Data/20210525 cold gsr_ehsan.txt"  # Path to the GSR Data ('.txt', '.csv', 'xlsx')
        # GSR Parameters
        windowGSR = 5     # Number of Points to Average Together
        # Save GSR Data
        saveGSRData = False
        if saveGSRData:
            saveDataFolderGSR = "./Output Data/GSR Data/20210510/"  # Data Folder to Save the Data; MUST END IN '/'
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # Machien Learning Parameters
    if trainModel:
        # Machine Learning File/Model Paths + Titles
        trainingDataExcelFolder = "./Input Data/Compiled Data/Stress Test Changhao/Rest&CPT data_Changhao only/Training Data/"
        validationDataExcelFolder = "./Input Data/Compiled Data/Stress Test Changhao/Rest&CPT data_Changhao only/Validation Data/"
        modelPath = "./Helper Files/Machine Learning Modules/Models/myPulseModel.pkl"
        signalLabelsTitles = ["Not Stressed", "Stressed"]
        # Specify the ML Model Type
        applyKNN = False
        applyNN = False
        applySVM = False
        applyRF = True
        applyLR = False
        # Specify if We Are Saving the Model
        saveModel = False
    # ---------------------------------------------------------------------- #
    
    # ---------------------------------------------------------------------- #
    #                   Extract Pulse Peak Data from Signals                 #
    # ---------------------------------------------------------------------- #
    
    if analyzePulse:
        # Read Data from Excel
        excelDataPulse = excelProcessing.processPulseData()
        time, signalData = excelDataPulse.getData(pulseExcelFile, testSheetNum = 0)
        signalData = signalData*10**12 # Get Data into pico-Farad
        
        # Plot the Initial Input Data
        plot = pulseAnalysis.plot()
        plot.plotData(time, signalData, title = "Input Pulse Data")
        
        # Seperate Pulses and Perform Indivisual Analysis
        dataProcessing = pulseAnalysis.signalProcessing()
        bloodPulse = dataProcessing.sepPulseAnalyze(time, signalData, diastolicCapacitance, systolicCapacitance,
                        minBPM = 30, maxBPM = 220, plotSeperation = plotSeperation, plotGaussFit = plotGaussFit)
        
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
            saveExcelName = os.path.basename(pulseExcelFile).split(".")[0] + ".xlsx"
            excelDataPulse.saveResults(savingDict, savingPulseInd, saveDataFolder, saveExcelName, sheetName)
    
    
    # ---------------------------------------------------------------------- #
    #                         Analyze GSR Data                               #
    # ---------------------------------------------------------------------- #
    
    if analyzeGSR:
        # Read Data from Excel
        excelDataGSR = excelProcessing.processGSRData()
        timeGSR, currentGSR = excelDataGSR.getData(gsrFile, testSheetNum = 0)
        currentGSR = currentGSR*10**6 # Get Data into micro-Ampes
        
        # Plot the Initial Input Data
        plot = gsrAnalysis.plot()
        plot.plotData(timeGSR, currentGSR, title = "Input GSR Data")
        
        gsrProcessing = gsrAnalysis.signalProcessing()
        timeGSR, currentGSR = gsrProcessing.downsizeDataTime(timeGSR, currentGSR, downsizeWindow = windowGSR)
        plot.plotData(timeGSR, currentGSR, title = "Downsized GSR Data")
        
        if saveGSRData:
            saveExcelNameGSR = os.path.basename(gsrFile).split(".")[0] + ".xlsx"
            excelDataGSR.saveResults(timeGSR, currentGSR, saveDataFolderGSR, saveExcelNameGSR)
        
        
    # ---------------------------------------------------------------------- #
    #                         Analyze GSR Data                               #
    # ---------------------------------------------------------------------- #
    
    if trainModel:
        excelDataML = excelProcessing.processMLData()
        # Read in Training Data/Labels
        signalData = []; signalLabels = []; headerTitles = []
        for MLFile in os.listdir(trainingDataExcelFolder):
            MLFile = trainingDataExcelFolder + MLFile
            signalData, signalLabels, headerTitles = excelDataML.getData(MLFile, signalData = signalData, signalLabels = signalLabels, testSheetNum = 0)
        signalData = np.array(signalData); signalLabels = np.array(signalLabels)
        # Read in Validation Data/Labels
        Validation_Data = []; Validation_Labels = [];
        for MLFile in os.listdir(validationDataExcelFolder):
            MLFile = validationDataExcelFolder + MLFile
            Validation_Data, Validation_Labels, headerTitles = excelDataML.getData(MLFile, signalData = Validation_Data, signalLabels = Validation_Labels, testSheetNum = 0)
        Validation_Data = np.array(Validation_Data); Validation_Labels = np.array(Validation_Labels)
        print("\nCollected Signal Data")
        
        Validation_Data = Validation_Data[:][:,0:6]
        signalData = signalData[:][:,0:6]
        headerTitles = headerTitles[0:6]
        
        # Split the Data Randomly into Training and Testing Data Sets
        Training_Data, Testing_Data, Training_Labels, Testing_Labels = train_test_split(signalData, signalLabels, test_size=0.5, shuffle= True, stratify=signalLabels)
        
        # Create the Machine Learning Module
        if applyNN:
            MLModel = ANN.NeuralNet(modelPath = modelPath, dataDim = len(signalData[0]))

            # If You Want to Loop Through All Options: DONT DO THIS (WORK IN PROGRESS)
            if False:
                ANNHelp = ANN.Helpers(modelPath, dataDimension = len(signalData[0]), numClasses = 2)
                neuralOptimizerList = ANNHelp.neuralPermutations()
                
                validationScores = []
                for model in neuralOptimizerList:
                    model.trainModel(Training_Data, Training_Labels, Testing_Data, Testing_Labels)
                    validationScores.append(model.scoreModel(Validation_Data, Validation_Labels))

        if applyKNN:
            MLModel = KNN.KNN(modelPath = modelPath, numClasses = 2, weight = 'distance')
        elif applySVM:
            MLModel = SVM.SVM(modelPath = modelPath, modelType = "linear", polynomialDegree = 5)
        elif applyRF:
            MLModel = randomForest.randomForest(modelPath = modelPath)
        elif applyLR:
            MLModel = LogisticRegression.logisticRegression(modelPath = modelPath)
        
        

        # Train the Classifier (the Model) with the Training Data
        MLModel.trainModel(Training_Data, Training_Labels, Testing_Data, Testing_Labels, scoreType = "Testing Score:")
        MLModel.scoreModel(Validation_Data, Validation_Labels, scoreType = "Validation Score:")
        MLModel.featureImportance(signalData, signalLabels, headerTitles)
        
        # Plot the Machine Learning Results
        MLModel.accuracyDistributionPlot(signalData, signalLabels, MLModel.predictData(signalData), signalLabelsTitles)
        #map2D = MLModel.mapTo2DPlot(signalData, signalLabels) # Plot the Classification in 2D Map
        
        # Find the Class Data Distribution in the Total Training/Testing Set
        classDistribution = collections.Counter(signalLabels)
        print("Class Distribution:", classDistribution)
        print("Number of Classes Found = ", len(classDistribution))

        # Save the Classifier: if Desired
        if saveModel:
            MLModel.saveModel(modelPath)
        
        

    
"""
# Extra Options to Extract Signal Information
import heartpy as hp
import neurokit2 as nk

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
        
pulseNum = 17; window = 10; x = bloodPulse[pulseNum]['time']; y = bloodPulse[pulseNum]['pulseData']; smoothY = bloodPulse[pulseNum]['smoothData']; fftY = scipy.fft(smoothY); plt.plot(x[1:],smoothY[0]+fftY[1:]-fftY[1]); plt.plot(x,smoothY)
"""
