"""
    Written by Samuel Solomon
    
    --------------------------------------------------------------------------
    Program Description:
    
    Perform signal processing to filter wrist pulse peaks. Seperate the peaks,
    and extract key features from each pulse. 
    
    --------------------------------------------------------------------------
    
    Modules to Import Before Running the Program (Some May be Missing):
        %conda install matplotlib
        %conda install openpyxl
        %conda install numpy
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

# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic Modules
import os
import sys
import numpy as np
from natsort import natsorted

# Import Data Extraction Files (And Their Location)
sys.path.append('./Data Aquisition and Analysis/')  # Folder with All the Helper Files
import excelProcessing

# Import Analysis Files (And Their Locations)
sys.path.append('./Data Aquisition and Analysis/_Analysis Protocols')  # Folder with All the Helper Files
import pulseAnalysis

# Import Machine Learning Files (And They Location)
sys.path.append("./Machine Learning/")
import machineLearningMain   # Class Header for All Machine Learning
import featureAnalysis       # Functions for Feature Analysis

# -------------------------------------------------------------------------- #
# --------------------------- Program Starts Here -------------------------- #

if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #    

    # Program Flags
    trainModel = False              # Train a Model with the Collected Features.
    
    # Data Collection Parameters
    useAllFiles = True              # If True, Read in All the Tabulated Data in the Input Folder.
    # Analysis Flags
    alreadyFilteredData = False     # Skip the Filtering and Baseline Subtraction.
    plotSeperation = False          # Plot the Results of Seperating the Pulse Data.
    plotFeatures = True             # Plot the Collected Features.
    saveAnalysis = True             # Save the Analyzed Data: The Peak Features for Each Well-Shaped Pulse
    plotPulses = False              # Plot Each Successful Pulse Analyzed.
    
    # Analysis Parameters
    stimulusTimes = [None, 1000 + 60*4]    # The [Beginning, End] of the Stimulus in Seconds; Type List.
    unitOfData = "pF"               # Specify the Unit the Data is Represented as: ['F', 'mF', 'uF', 'nF', 'pF', 'fF']
    
    if useAllFiles:
        # Specify the Location of the Input Data
        inputFolder = '../Input Data/All Data/Cleaned data for ML/Subject 1/CPT pulse/' # Path to ALL the Excel Data. The Path Must End with '/'
    else:
        pulseExcelFiles = ["../Input Data/Pulse Data/20220112 CPT/62.xls"]          # A List of the Excel Data ('.xls' or '.xlsx') to Analyze.

    # ---------------------------------------------------------------------- #
    # ------------------------- Preparation Steps -------------------------- #

    # Create all the Pulse Analysis Instances
    excelProcessingPulse = excelProcessing.processPulseData()
    pulseAnalysisProtocol = pulseAnalysis.signalProcessing(alreadyFilteredData, plotPulses, plotSeperation)
    
    # Find the Scale Factor for the Data
    scaleFactorMap = {'F': 1, 'mF': 10**-3, 'uF': 10**-6, 'nF': 10**-9, 'pF': 10**-12, 'fF': 10**-15}
    scaleFactor = 1/scaleFactorMap[unitOfData]
    
    # Collect the File Names, If Using Every File in the Folder
    if useAllFiles:
        pulseExcelFiles = []
        for file in os.listdir(inputFolder):
            if file.endswith(("xlsx", "xls")) and not file.startswith(("$", '~')):
                pulseExcelFiles.append(inputFolder + file)
        pulseExcelFiles = natsorted(pulseExcelFiles)
    
    if trainModel or plotFeatures:
        # Extract the Pulse Features we are Using
        pulseFeaturesFile = "./Machine Learning/Compiled Feature Names/pulseFeatureLabels.txt"
        pulseFeatureLabels = excelProcessingPulse.extractFeatures(pulseFeaturesFile, prependedString = "pulseFeatures.extend([")
    
    # Saves the Analyzed Data
    if saveAnalysis:
        saveDataFolder = inputFolder + "Pulse Analysis/"    # Data Folder to Save the Data; MUST END IN '/'
        sheetName = "Blood Pulse Data"                      # If SheetName Already Exists, Excel Will Add 1 to the end (The Copy Number) 

    # ---------------------------------------------------------------------- #
    # ---------------------- Machine Learning Protocol --------------------- #
    
    # Machien Learning Parameters
    if trainModel:
        # Machine Learning File/Model Paths + Titles
        trainingDataExcelFolder = "../Input Data/Compiled Data/Stress Test Changhao/Rest&CPT data_Changhao only/Training Data/"
        validationDataExcelFolder = "../Input Data/Compiled Data/Stress Test Changhao/Rest&CPT data_Changhao only/Validation Data/"
        modelPath = "./Machine Learning Modules/Models/myPulseModel.pkl"
        modelType = "RF"  # Machine Learning Options: RF, LR, KNN, SVM
        machineLearningClasses = ["Not Stressed", "Stressed"]
        # Specify if We Are Saving the Model
        saveModel = False
        saveDataFolder = trainingDataExcelFolder + "Data Analysis/" + modelType + "/"
        # Get the Machine Learning Module
        performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = len(pulseFeatureLabels), gestureClasses = machineLearningClasses, saveDataFolder = saveDataFolder)
        predictionModel = performMachineLearning.predictionModel

    # ---------------------------------------------------------------------- #
    # ------------------- Extract and Analyze Pulse Data ------------------- #
    
    # For Each PulseFile, Collect the Data in the Same Instance
    for pulseExcelFile in pulseExcelFiles:
        # Read Data from Excel
        time, signalData = excelProcessingPulse.getData(pulseExcelFile, testSheetNum = 0)
        # Get Data into Farads
        if not alreadyFilteredData:
            signalData = signalData*scaleFactor
        
        # Calibrate Systolic and Diastolic Pressure
        fileBasename = os.path.basename(pulseExcelFile)
        pressureInfo = fileBasename.split("SYS")
        if len(pressureInfo) > 1 and pulseAnalysisProtocol.systolicPressure0 == None:
            pressureInfo = pressureInfo[-1].split(".")[0]
            systolicPressure0, diastolicPressure0 = pressureInfo.split("_DIA")
            pulseAnalysisProtocol.setPressureCalibration(float(systolicPressure0), float(diastolicPressure0))
        
        # Check Whether the StartTime is Specified in the File
        if fileBasename.lower() in ["cpt", "exercise", "vr", "start"] and stimulusTimes[0] != None:
            stimulusTimes[0] = pulseAnalysisProtocol.timeOffset
        
        # Seperate Pulses, Perform Indivisual Analysis, and Extract Features
        pulseData = pulseAnalysisProtocol.analyzePulse(time, signalData, minBPM = 30, maxBPM = 180)
    
    # Plot the Features Collected from the Pulses
    if plotFeatures:
        pulseAnalysisProtocol.featureListExact = np.array(pulseAnalysisProtocol.featureListExact)
        plotFeatures = featureAnalysis.featureAnalysis(pulseAnalysisProtocol.featureListExact[:,0], pulseAnalysisProtocol.featureListExact[:,1:], pulseFeatureLabels[1:], stimulusTimes, saveDataFolder)
        plotFeatures.singleFeatureAnalysis()
    
    # Save Pulse Data
    if saveAnalysis:
        saveCompiledData = saveDataFolder + "Compiled Data in Excel/"
        # Save the Features and Filtered Data
        excelProcessingPulse.saveResults(pulseAnalysisProtocol.featureListExact, pulseFeatureLabels, saveCompiledData, "Feature List.xlsx", sheetName)
        excelProcessingPulse.saveFilteredData(pulseAnalysisProtocol.time, pulseAnalysisProtocol.signalData, pulseAnalysisProtocol.filteredData, saveCompiledData, "Filtered Data.xlsx", "Filtered Data")
        
    # ---------------------------------------------------------------------- #
    #                          Train the Model                               #
    # ---------------------------------------------------------------------- #
    
    if trainModel:
        excelDataML = excelProcessing.processMLData()
        # Read in Training Data/Labels
        signalData = []; signalLabels = []; featureLabels = []
        for MLFile in os.listdir(trainingDataExcelFolder):
            MLFile = trainingDataExcelFolder + MLFile
            signalData, signalLabels, featureLabels = excelDataML.getData(MLFile, signalData = signalData, signalLabels = signalLabels, testSheetNum = 0)
        signalData = np.array(signalData); signalLabels = np.array(signalLabels)
        # Read in Validation Data/Labels
        Validation_Data = []; Validation_Labels = [];
        for MLFile in os.listdir(validationDataExcelFolder):
            MLFile = validationDataExcelFolder + MLFile
            Validation_Data, Validation_Labels, featureLabels = excelDataML.getData(MLFile, signalData = Validation_Data, signalLabels = Validation_Labels, testSheetNum = 0)
        Validation_Data = np.array(Validation_Data); Validation_Labels = np.array(Validation_Labels)
        print("\nCollected Signal Data")
        
        Validation_Data = Validation_Data[:][:,0:6]
        signalData = signalData[:][:,0:6]
        featureLabels = featureLabels[0:6]
                    
        # Train the Data on the Gestures
        performMachineLearning.trainModel(signalData, signalLabels, pulseFeatureLabels)
        # Save Signals and Labels
        if False and performMachineLearning.map2D:
            saveInputs = excelProcessing.saveExcel()
            saveExcelNameMap = "mapedData.xlsx" #"Signal Features with Predicted and True Labels New.xlsx"
            saveInputs.saveLabeledPoints(performMachineLearning.map2D, signalLabels,  performMachineLearning.predictionModel.predictData(signalData), saveDataFolder, saveExcelNameMap, sheetName = "Signal Data and Labels")
        # Save the Neural Network (The Weights of Each Edge)
        if saveModel:
            modelPathFolder = os.path.dirname(modelPath)
            os.makedirs(modelPathFolder, exist_ok=True)
            performMachineLearning.predictionModel.saveModel(modelPath)
    
        

