"""
    Written by Samuel Solomon
    
    --------------------------------------------------------------------------
    Program Description:
    
    Perform signal processing to filter blood pulse peaks. Seperate the peaks,
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


# Basic Modules
import os
import sys
import numpy as np
from pathlib import Path
from natsort import natsorted
# Import Python Helper Files (And Their Location)
sys.path.append('./Helper Files/Data Aquisition and Analysis/_Analysis Protocols')  # Folder with All the Helper Files
sys.path.append('./Helper Files/Data Aquisition and Analysis/')  # Folder with All the Helper Files
import excelProcessing
import pulseAnalysis
import gsrAnalysis
# Import Machine Learning Files (And They Location)
sys.path.append("Helper Files/Machine Learning/")
# Import Files for Machine Learning
import machineLearningMain  # Class Header for All Machine Learning
import featureAnalysis  # Functions for Feature Analysis


if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #
    sys.exit()

    # Specify Which Program to Run; All Can be Run in One Scirpt (NOT Simutaneously Yet)
    analyzePulse = True
    analyzeGSR = False
    trainModel = False
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # Pulse Parameters
    if analyzePulse:
        multipleFiles = True
        # Specify the Location of the Input Data
        if multipleFiles:
            pulseExcelFiles = []
            inputFolder = './Input Data/Pulse Data/20220112 CPT/'
            for file in os.listdir(inputFolder):
                if file.endswith(("xlsx", "xls")):
                    pulseExcelFiles.append(inputFolder + file)
            pulseExcelFiles = natsorted(pulseExcelFiles)
        else:
            pulseExcelFiles = ["./Input Data/Pulse Data/20220112 CPT/62.xls"] # Path to the Excel Data ('.xls' or '.xlsx')
        # Parameters to Visualize the Pulse Data
        plotSeperation = False
        plotGaussFit = False
        # If Filtering Twice
        alreadyFilteredData = False
        # Saves the Data Analysis: Peak Features for Each Well-Shaped Pulse
        saveInputData = True   
        if saveInputData:
            saveDataFolder = "./Output Data/Pulse Data/20220112 CPT/"      # Data Folder to Save the Data; MUST END IN '/'
            sheetName = "Blood Pulse Data"                   # If SheetName Already Exists, Excel Will Add 1 to the end (The Copy Number) 
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    # GSR Parameters
    if analyzeGSR:
        # Specify the Location of the Input Data
        gsrFile = "./Input Data/Galvanic Skin Response Data/20220112 CPT/1.xlsx"  # Path to the GSR Data ('.txt', '.csv', 'xlsx')
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
        modelType = "RF"  # Machine Learning Options: NN, RF, LR, KNN, SVM
        machineLearningClasses = ["Not Stressed", "Stressed"]
        # Specify if We Are Saving the Model
        saveModel = False
        saveDataFolder = trainingDataExcelFolder + "Data Analysis/" + modelType + "/"
        # Get the Machine Learning Module
        pulseFeatures = []
        performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = len(pulseFeatures), gestureClasses = machineLearningClasses, saveDataFolder = saveDataFolder)
        predictionModel = performMachineLearning.predictionModel

    # ---------------------------------------------------------------------- #
    
    # ---------------------------------------------------------------------- #
    #                   Extract Pulse Peak Data from Signals                 #
    # ---------------------------------------------------------------------- #

    if analyzePulse:
        # Create all the Pulse Analysis Instances
        plot = pulseAnalysis.plot()
        excelDataPulse = excelProcessing.processPulseData()
        dataProcessing = pulseAnalysis.signalProcessing(alreadyFilteredData, plotGaussFit, plotSeperation)
        
        # For Each PulseFile, Collect the Data in the Same Instance
        for pulseExcelFile in pulseExcelFiles:
            # Read Data from Excel
            time, signalData = excelDataPulse.getData(pulseExcelFile, testSheetNum = 0)
            if not alreadyFilteredData:
                signalData = signalData*10**12 # Get Data into pico-Farad
            
            # Plot the Initial Input Data
            # plot.plotData(time, signalData, title = "Input Pulse Data")
            
            # Seperate Pulses, Perform Indivisual Analysis, nd Extract Features
            dataProcessing.analyzePulse(time, signalData, minBPM = 30, maxBPM = 180)
            
        # Input Feature Labels
        pulseFeatures = ["timePoint"]
        # Saving Features from Section: Extract Data from Peak Inds
        pulseFeatures.extend(['systolicUpstrokeAccelMaxTime', 'systolicUpstrokeVelTime', 'systolicUpstrokeAccelMinTime', 'systolicPeakTime'])
        pulseFeatures.extend(['beforeTidalVelMinTime', 'tidalStartTime', 'tidalPeakTime', 'tidalEndTime', 'afterTidalVelMinTime'])
        pulseFeatures.extend(['dicroticNotchTime', 'dicroticRiseVelMaxTime', 'dicroticPeakTime', 'dicroticFallVelMinTime'])
        pulseFeatures.extend(['systolicUpstrokeAccelMaxAmp', 'systolicUpstrokeVelAmp', 'systolicUpstrokeAccelMinAmp', 'systolicPeakAmp'])
        pulseFeatures.extend(['beforeTidalVelMinAmp', 'tidalStartAmp', 'tidalPeakAmp', 'tidalEndAmp', 'afterTidalVelMinAmp'])
        pulseFeatures.extend(['dicroticNotchAmp', 'dicroticRiseVelMaxAmp', 'dicroticPeakAmp', 'dicroticFallVelMinAmp'])
        pulseFeatures.extend(['systolicUpstrokeAccelMaxVel', 'systolicUpstrokeVelVel', 'systolicUpstrokeAccelMinVel', 'systolicPeakVel'])
        pulseFeatures.extend(['beforeTidalVelMinVel', 'tidalStartVel', 'tidalPeakVel', 'tidalEndVel', 'afterTidalVelMinVel'])
        pulseFeatures.extend(['dicroticNotchVel', 'dicroticRiseVelMaxVel', 'dicroticPeakVel', 'dicroticFallVelMinVel'])
        pulseFeatures.extend(['systolicUpstrokeAccelMaxAccel', 'systolicUpstrokeVelAccel', 'systolicUpstrokeAccelMinAccel', 'systolicPeakAccel'])
        pulseFeatures.extend(['beforeTidalVelMinAccel', 'tidalStartAccel', 'tidalPeakAccel', 'tidalEndAccel', 'afterTidalVelMinAccel'])
        pulseFeatures.extend(['dicroticNotchAccel', 'dicroticRiseVelMaxAccel', 'dicroticPeakAccel', 'dicroticFallVelMinAccel'])
        
        # Saving Features from Section: Time Features
        pulseFeatures.extend(['pulseDuration', 'systolicTime', 'DiastolicTime', 'leftVentricularPerformance'])
        pulseFeatures.extend(['maxDerivToSystolic', 'systolicToTidal', 'systolicToDicroticNotch', 'systolicToDicrotic', 'dicroticNotchToTidal', 'dicroticNotchToDicrotic'])
        pulseFeatures.extend(['systolicUpSlopeTime', 'tidalPeakInterval', 'outerTidalInterval', 'midToEndTidal', 'tidalToDicroticVelPeakInterval'])

        # Saving Features from Section: Under the Curve Features
        pulseFeatures.extend(['pulseArea', 'pulseAreaSquared', 'leftVentricleLoad', 'diastolicArea'])
        pulseFeatures.extend(['systolicUpSlopeArea', 'velToTidalArea', 'geometricMean', 'pulseAverage'])
        
        # Saving Features from Section: Ratio Features
        pulseFeatures.extend(['areaRatio', 'systolicDicroticNotchAmpRatio', 'systolicDicroticNotchVelRatio', 'systolicDicroticNotchAccelRatio'])
        pulseFeatures.extend(['systolicTidalAmpRatio', 'systolicDicroticAmpRatio', 'dicroticNotchTidalAmpRatio', 'dicroticNotchDicroticAmpRatio'])
        pulseFeatures.extend(['systolicTidalVelRatio', 'systolicDicroticVelRatio', 'dicroticNotchTidalVelRatio', 'dicroticNotchDicroticVelRatio'])
        pulseFeatures.extend(['systolicTidalAccelRatio', 'systolicDicroticAccelRatio', 'dicroticNotchTidalAccelRatio', 'dicroticNotchDicroticAccelRatio'])

        # Saving Features from Section: Slope Features
        pulseFeatures.extend(['systolicSlopeUp', 'SystolicSlopeDown', 'SystolicSlopeDown2', 'tidalSlope', 'DicroticSlopeUp', 'endSlope'])

        # Saving Features from Section: Biological Features
        pulseFeatures.extend(['momentumDensity', 'pseudoCardiacOutput', 'pseudoStrokeVolume'])
        pulseFeatures.extend(['maxSystolicVelocity', 'valveCrossSectionalArea', 'velocityTimeIntegral', 'velocityTimeIntegralABS', 'velocityTimeIntegral_ALT'])
        pulseFeatures.extend(['centralAugmentationIndex', 'centralAugmentationIndex_EST', 'reflectionIndex', 'stiffensIndex'])
        # sys.exit()
             
        dataProcessing.featureList = np.array(dataProcessing.featureList)
        analyzeFeatures = featureAnalysis.featureAnalysis(dataProcessing.featureList[:,0], dataProcessing.featureList[:,1:], pulseFeatures[1:], [1110, 1110+60*3], saveDataFolder)
        analyzeFeatures.singleFeatureAnalysis()
        
        sys.exit()
        # Save Pulse Labels (if Desired)
        if saveInputData:
            saveExcelName = os.path.basename(pulseExcelFile).split(".")[0] + ".xlsx"
            excelDataPulse.saveResults(dataProcessing.featureList, pulseFeatures, saveDataFolder, saveExcelName, sheetName)
            # NOT IMPLEMENTED BELOW
            # saveExcelName = os.path.basename(pulseExcelFile).split(".")[0] + "_Data.xlsx"
            # excelDataPulse.saveFilteredData(savingDict, savingPulseInd, saveDataFolder + "Analyzed Data/", saveExcelName, "Filtered Data")

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
    #                          Train the Model                               #
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
                    
        # Train the Data on the Gestures
        performMachineLearning.trainModel(signalData, signalLabels, pulseFeatures)
        # Save Signals and Labels
        if False and performMachineLearning.map2D:
            saveInputs = excelProcessing.saveExcel()
            saveExcelNameMap = Path(saveExcelName).stem + "_mapedData.xlsx" #"Signal Features with Predicted and True Labels New.xlsx"
            saveInputs.saveLabeledPoints(performMachineLearning.map2D, signalLabels,  performMachineLearning.predictionModel.predictData(signalData), saveDataFolder, saveExcelNameMap, sheetName = "Signal Data and Labels")
        # Save the Neural Network (The Weights of Each Edge)
        if saveModel:
            modelPathFolder = os.path.dirname(modelPath)
            os.makedirs(modelPathFolder, exist_ok=True)
            performMachineLearning.predictionModel.saveModel(modelPath)
    
        

"""
# --------- Analyze the Effect of Multiple Runs

from scipy import stats
import matplotlib.pyplot as plt
signalData = np.array(signalData); signalLabels = np.array(signalLabels)

#featureList1 = np.array(dataProcessing.featureList  # Rest
#featureList2 = np.array(dataProcessing.featureList  # Cold/IP
#featureList3 = np.array(dataProcessing.featureList  # Recover

time1 = featureList1[:, 0]
time2 = featureList2[:, 0]
time3 = featureList3[:, 0]

for featureInd in range(len(pulseFeatures)):
    fig = plt.figure()

    allFeatures1 = featureList1[:,featureInd]
    allFeatures2 = featureList2[:,featureInd]
    allFeatures3 = featureList3[:,featureInd]
    allFeatures = [allFeatures1, allFeatures2, allFeatures3]
    
    time = [time1, time2, time3]
    
    colors = ['ko', 'ro', 'bo', 'go', 'mo']
    for ind, averageTogether in enumerate([60*0.5]):
        features = [[] for _ in range(len(allFeatures))]
        for fileInd, allFeaturesI in enumerate(allFeatures):
            for pointInd in range(len(allFeaturesI)):
                featureInterval = allFeaturesI[time[fileInd] > time[fileInd][pointInd] - averageTogether]
                timeMask = time[fileInd][time[fileInd] > time[fileInd][pointInd] - averageTogether]
                featureInterval = featureInterval[timeMask <= time[fileInd][pointInd]]
    
                featute = stats.trim_mean(featureInterval, 0.3)
                features[fileInd].append(featute)

        plt.plot(time1, features[0], colors[0], markersize=5)
        plt.plot(time2, features[1], colors[1], markersize=5)
        plt.plot(time3, features[2], colors[2], markersize=5)


    plt.xlabel("Time (Seconds)")
    plt.ylabel(pulseFeatures[featureInd])
    #plt.vlines(np.array([1, 2, 3, 4, 5])*60*6, min(min(features))*0.8, max(max(features))*1.2, 'g', zorder=100)
    plt.legend(['Rest', 'Cold', "Recover"])
    plt.title("Averaged Together: " + str(averageTogether) + " Seconds")
    fig.savefig('./Heather IP/Compare/' + pulseFeatures[featureInd] + ".png", dpi=300, bbox_inches='tight')
    plt.show()
# -----------

for i in range(len(dataProcessing.featureList[:,0])):
    plt.plot(dataProcessing.featureList[:,0], dataProcessing.featureList[:,i])
    plt.title(pulseFeatures[i])
    plt.show()
"""
