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
# Import Data Extraction Files (And Their Location)
sys.path.append('./Helper Files/Data Aquisition and Analysis/')  # Folder with All the Helper Files
import excelProcessing
# Import Analysis Files (And Their Locations)
sys.path.append('./Helper Files/Data Aquisition and Analysis/_Analysis Protocols')  # Folder with All the Helper Files
import chemicalAnalysis
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

    # Specify Which Program to Run; All Can be Run in One Scirpt (NOT Simutaneously Yet)
    analyzePulse = True
    analyzeChemical = False
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
            inputFolder = './Input Data/Pulse Data/20220119 changhao VR stress pulse/'
            for file in os.listdir(inputFolder):
                if file.endswith(("xlsx", "xls")) and not file.startswith(("$", '~')):
                    pulseExcelFiles.append(inputFolder + file)
            pulseExcelFiles = natsorted(pulseExcelFiles)
        else:
            pulseExcelFiles = ["./Input Data/Pulse Data/20220112 CPT/62.xls"] # Path to the Excel Data ('.xls' or '.xlsx')
        
        # Parameters to Visualize the Pulse Data
        plotSeperation = False
        plotGaussFit = False
        
        # If Filtering Twice
        alreadyFilteredData = False
        
        # Plot the Features in Time
        analyzeFeatures = True
        
        # Saves the Data Analysis: Peak Features for Each Well-Shaped Pulse
        saveInputData = True   
        if saveInputData:
            saveDataFolder = inputFolder + "Pulse Analysis/"     # Data Folder to Save the Data; MUST END IN '/'
            sheetName = "Blood Pulse Data"                   # If SheetName Already Exists, Excel Will Add 1 to the end (The Copy Number) 
    # ---------------------------------------------------------------------- #
    # ---------------------------------------------------------------------- #
    if analyzeChemical:
        multipleFiles = True
        # Specify the Location of the Input Data
        if multipleFiles:
            chemicalFiles = []
            inputFolder = './Input Data/Chemical Data/'
            for file in os.listdir(inputFolder):
                if file.endswith(("xlsx", "xls")) and not file.startswith(("$", '~')):
                    chemicalFiles.append(inputFolder + file)
            chemicalFiles = natsorted(chemicalFiles)
        else:
            chemicalFiles = ["./Input Data/Chemical Data/20211022 cold enzymatic_jose - aligned.xlsx"] # Path to the Excel Data ('.xls' or '.xlsx')
        
        # Save Data
        saveChemicalData = True
        analyzeFeatures = True
        if saveChemicalData:
            saveDataFolderChemical = inputFolder + "Chemical Analysis/"     # Data Folder to Save the Data; MUST END IN '/'
    
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
            saveDataFolderGSR = inputFolder + "GSR Analysis/"     # Data Folder to Save the Data; MUST END IN '/'
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
        startStimulus = 0
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
            
            # Calibrate Systolic and Diastolic Pressure
            fileBasename = os.path.basename(pulseExcelFile)
            pressureInfo = fileBasename.split("SYS")
            if len(pressureInfo) > 1 and dataProcessing.systolicPressure0 == None:
                pressureInfo = pressureInfo[-1].split(".")[0]
                systolicPressure0, diastolicPressure0 = pressureInfo.split("_DIA")
                dataProcessing.setPressureCalibration(float(systolicPressure0), float(diastolicPressure0))
            
            if fileBasename.lower() in ["cpt", "exercise start"]:
                startStimulus = dataProcessing.timeOffset
            
            # Seperate Pulses, Perform Indivisual Analysis, nd Extract Features
            pulseData = dataProcessing.analyzePulse(time, signalData, minBPM = 30, maxBPM = 180)
            
        # Input Feature Labels
        pulseFeaturesFile = "./Helper Files/Machine Learning/pulseFeatures.txt"
        pulseFeatures = excelDataPulse.extractFeatures(pulseFeaturesFile, prependedString = "pulseFeatures.extend([")
        
        if analyzeFeatures:
            stimulusTimes = [startStimulus, startStimulus+60*3]
            dataProcessing.featureList = np.array(dataProcessing.featureList)
            analyzeFeatures = featureAnalysis.featureAnalysis(dataProcessing.featureList[:,0], dataProcessing.featureList[:,1:], pulseFeatures[1:], stimulusTimes, saveDataFolder)
            analyzeFeatures.singleFeatureAnalysis()
        
        # Save Pulse Labels (if Desired)
        if saveInputData:
            saveCompiledData = saveDataFolder + "Compiled Data in Excel/"
            # Save the Features and Filtered Data
            excelDataPulse.saveResults(dataProcessing.featureList, pulseFeatures, saveCompiledData, "Feature List.xlsx", sheetName)
            excelDataPulse.saveFilteredData(dataProcessing.time, dataProcessing.signalData, dataProcessing.filteredData, saveCompiledData, "Filtered Data.xlsx", "Filtered Data")
        
    # ---------------------------------------------------------------------- #
    #                       Analyze Chemical Data                            #
    # ---------------------------------------------------------------------- #
    
    if analyzeChemical:
        # Create all the Analysis Instances
        chemicalProcessing = chemicalAnalysis.signalProcessing(startStimulus = 1000, stimulusDuration = 3*60, stimulusBuffer = 500, plotData = True)
        excelDataChemical = excelProcessing.processChemicalData()
        
        analyzeTogether = True
        
        featureLabels = []
        for chemicalFile in chemicalFiles:
            # Read in the Chemical Data from Excel
            timePoints, chemicalData = excelDataChemical.getData(chemicalFile, testSheetNum = 0)
            glucose, lactate, uricAcid = chemicalData # Extract the Specific Chemicals
            
            fileName = Path(chemicalFile).stem
            featureLabel = fileName.split(" ")[1]
            if 'cold' == featureLabel.lower():
                featureLabel = 0
            elif 'exercise' == featureLabel.lower():
                featureLabel = 1
            elif 'vr' == featureLabel.lower():
                featureLabel = 2
            else:
                print("UNSURE OF THE LABEL. STOP CHANGING THE FORMAT ON ME"); sys.exit()
                            
            # Process the Data
            if not analyzeTogether or (len(glucose) > 0 and len(lactate) > 0 and len(uricAcid) > 0):
                chemicalProcessing.analyzeChemicals(timePoints, glucose, lactate, uricAcid, featureLabel, analyzeTogether)
            
            # if chemicalProcessing.continueAnalysis:
            #     fileName = Path(chemicalFile).stem
            #     featureLabel = fileName.split(" ")[1]
                
            #     if 'cold' == featureLabel.lower():
            #         featureLabels.append(0)
            #     elif 'exercise' == featureLabel.lower():
            #         featureLabels.append(1)
            #     elif 'vr' == featureLabel.lower():
            #         featureLabels.append(2)
            #     else:
            #         print("UNSURE OF THE LABEL. STOP CHANGING THE FORMAT ON ME"); sys.exit()
        
        glucoseFeatures = np.array(chemicalProcessing.glucoseFeatures)
        lactateFeatures = np.array(chemicalProcessing.lactateFeatures)
        uricAcidFeatures = np.array(chemicalProcessing.uricAcidFeatures)
        
        featureLabelsGlucose = np.array(chemicalProcessing.featureLabelsGlucose)
        featureLabelsLactate = np.array(chemicalProcessing.featureLabelsLactate)
        featureLabelsUricAcid = np.array(chemicalProcessing.featureLabelsUricAcid)
    
        # featureNames = []
        # featureNames.extend(['baselineData', 'velocity', 'acceleration', 'thirdDeriv', 'forthDeriv'])
        
        # saveDataFolderChemical = "./Output Data/Chemical Data/Pointwise Analysis - all chemicals/"  # Data Folder to Save the Data; MUST END IN '/'
        # analyzeFeatures = featureAnalysis.featureAnalysis([], [], featureNames, [1110, 1110+60*3], saveDataFolderChemical)
        # analyzeFeatures.singleFeatureComparison([glucoseFeatures, lactateFeatures, uricAcidFeatures], [featureLabelsGlucose,featureLabelsLactate,featureLabelsUricAcid], ["Glucose", "Lactate", "UricAcid"], featureNames)
        # if analyzeTogether:
        #     analyzeFeatures.featureComparison(lactateFeatures, lactateFeatures, featureLabelsGlucose, featureNames, 'Lactate', 'Lactate')
        #     analyzeFeatures.featureComparison(uricAcidFeatures, uricAcidFeatures, featureLabelsGlucose, featureNames, 'Glucose', 'Glucose')
        #     analyzeFeatures.featureComparison(glucoseFeatures, glucoseFeatures, featureLabelsGlucose, featureNames, 'Uric Acid', 'Uric Acid')
        #     analyzeFeatures.featureComparison(lactateFeatures, uricAcidFeatures, featureLabelsGlucose, featureNames, 'Lactate', 'Uric Acid')
        #     analyzeFeatures.featureComparison(glucoseFeatures, uricAcidFeatures, featureLabelsGlucose, featureNames, 'Glucose', 'Uric Acid')
        #     analyzeFeatures.featureComparison(lactateFeatures, glucoseFeatures, featureLabelsGlucose, featureNames, 'Lactate', 'Glucose')
            
        # sys.exit()
        
        
        glucoseNames = []
        # Saving Features from Section: Time
        glucoseNames.extend(['peakHeight_Glucose', 'peakDiffLeft_Glucose', 'peakDiffRight_Glucose'])
        glucoseNames.extend(['velInterval_Glucose', 'velIntervalLeft_Glucose', 'velIntervalRight_Glucose'])
        glucoseNames.extend(['accelIntervalLeft_Glucose', 'accelIntervalRight_Glucose', 'accelInterval_Glucose', 'thirdDerivInterval_Glucose'])
        # Saving Features from Section: Amplitude Features
        glucoseNames.extend(['maxUpSlopeConc_Glucose', 'maxDownSlopeConc_Glucose'])
        glucoseNames.extend(['maxUpSlopeVel_Glucose', 'maxDownSlopeVel_Glucose'])
        glucoseNames.extend(['maxAccelLeftIndConc_Glucose', 'minAccelCenterIndConc_Glucose', 'maxAccelRightIndConc_Glucose'])
        glucoseNames.extend(['maxAccelLeftIndAccel_Glucose', 'minAccelCenterIndAccel_Glucose', 'maxAccelRightIndAccel_Glucose'])
        glucoseNames.extend(['velDiffConc_Glucose', 'accelDiffMaxConc_Glucose', 'accelDiffRightConc_Glucose', 'accelDiffLeftConc_Glucose'])
        glucoseNames.extend(['velDiff_Glucose', 'accelDiffMax_Glucose', 'accelDiffRight_Glucose', 'accelDiffLeft_Glucose'])
        # Saving Features from Section: Slope Features
        glucoseNames.extend(['upSlope_Glucose', 'downSlope_Glucose'])
        # Saving Features from Section: Under the Curve Features
        glucoseNames.extend(['velToVelArea_Glucose'])
        # Saving Features from Section: Peak Shape Features
        glucoseNames.extend(['peakTentX_Glucose', 'peakTentY_Glucose', 'tentDeviationX_Glucose', 'tentDeviationY_Glucose', 'tentDeviationRatio_Glucose', 'blinkDuration_Final_Glucose'])
        glucoseNames.extend(['peakAverage_Glucose', 'peakEntropy_Glucose', 'peakSkew_Glucose', 'peakKurtosis_Glucose'])
        glucoseNames.extend(['peakHeightFFT_Glucose', 'leftVelHeightFFT_Glucose','rightVelHeightFFT_Glucose', 'peakSTD_FFT_Glucose', 'peakEntropyFFT_Glucose'])
        glucoseNames.extend(['peakCurvature_Glucose', 'leftVelCurvature_Glucose', 'rightVelCurvature_Glucose'])
        
        lactateNames = []
        # Saving Features from Section: Time Features
        lactateNames.extend(['peakHeight_Lactate', 'peakDiffLeft_Lactate', 'peakDiffRight_Lactate'])
        lactateNames.extend(['velInterval_Lactate', 'velIntervalLeft_Lactate', 'velIntervalRight_Lactate'])
        lactateNames.extend(['accelIntervalLeft_Lactate', 'accelIntervalRight_Lactate', 'accelInterval_Lactate', 'thirdDerivInterval_Lactate'])
        # Saving Features from Section: Amplitude Features
        lactateNames.extend(['maxUpSlopeConc_Lactate', 'maxDownSlopeConc_Lactate'])
        lactateNames.extend(['maxUpSlopeVel_Lactate', 'maxDownSlopeVel_Lactate'])
        lactateNames.extend(['maxAccelLeftIndConc_Lactate', 'minAccelCenterIndConc_Lactate', 'maxAccelRightIndConc_Lactate'])
        lactateNames.extend(['maxAccelLeftIndAccel_Lactate', 'minAccelCenterIndAccel_Lactate', 'maxAccelRightIndAccel_Lactate'])
        lactateNames.extend(['velDiffConc_Lactate', 'accelDiffMaxConc_Lactate', 'accelDiffRightConc_Lactate', 'accelDiffLeftConc_Lactate'])
        lactateNames.extend(['velDiff_Lactate', 'accelDiffMax_Lactate', 'accelDiffRight_Lactate', 'accelDiffLeft_Lactate'])
        lactateNames.extend(['leftDiffAmp_Lactate', 'rightDiffAmp_Lactate'])
        # Saving Features from Section: Slope Features
        lactateNames.extend(['upSlope_Lactate', 'downSlope_Lactate'])
        # Saving Features from Section: Under the Curve Features
        lactateNames.extend(['velToVelArea_Lactate'])
        # Saving Features from Section: Peak Shape Features
        lactateNames.extend(['peakTentX_Lactate', 'peakTentY_Lactate', 'tentDeviationX_Lactate', 'tentDeviationY_Lactate', 'tentDeviationRatio_Lactate', 'blinkDuration_Final_Lactate'])
        lactateNames.extend(['peakAverage_Lactate', 'peakEntropy_Lactate', 'peakSkew_Lactate', 'peakKurtosis_Lactate'])
        lactateNames.extend(['peakHeightFFT_Lactate', 'leftVelHeightFFT_Lactate','rightVelHeightFFT_Lactate', 'peakSTD_FFT_Lactate', 'peakEntropyFFT_Lactate'])
        lactateNames.extend(['peakCurvature_Lactate', 'leftVelCurvature_Lactate', 'rightVelCurvature_Lactate'])
        
        uricAcidNames = []
        # Saving Features from Section: Amplitude Features
        uricAcidNames.extend(['maxUpSlopeConc_UricAcid', 'maxDownSlopeConc_UricAcid'])
        uricAcidNames.extend(['maxUpSlopeVel_UricAcid', 'maxDownSlopeVel_UricAcid'])
        uricAcidNames.extend(['maxAccelLeftIndConc_UricAcid', 'minAccelCenterIndConc_UricAcid', 'maxAccelRightIndConc_UricAcid'])
        uricAcidNames.extend(['velDiffConc_UricAcid', 'accelDiffMaxConc_UricAcid', 'accelDiffRightConc_UricAcid', 'accelDiffLeftConc_UricAcid'])
        uricAcidNames.extend(['leftDiffAmp_UricAcid', 'rightDiffAmp_UricAcid'])
        # Saving Features from Section: Slope Features
        uricAcidNames.extend(['upSlope_UricAcid', 'downSlope_UricAcid'])
        # Saving Features from Section: Under the Curve Features
        uricAcidNames.extend(['velToVelArea_UricAcid'])
        # Saving Features from Section: Peak Shape Features
        uricAcidNames.extend(['peakTentX_UricAcid', 'peakTentY_UricAcid', 'tentDeviationX_UricAcid', 'tentDeviationY_UricAcid', 'tentDeviationRatio_UricAcid', 'blinkDuration_Final_UricAcid'])
        uricAcidNames.extend(['peakSkew_UricAcid', 'peakKurtosis_UricAcid'])
        uricAcidNames.extend(['peakHeightFFT_UricAcid', 'leftVelHeightFFT_UricAcid','rightVelHeightFFT_UricAcid'])
        
        featureNames = []
        featureNames.extend(glucoseNames)
        featureNames.extend(lactateNames)
        featureNames.extend(uricAcidNames)

        
        analyzeFeatures = featureAnalysis.featureAnalysis([], [], featureNames, [1110, 1110+60*3], saveDataFolderChemical)
        analyzeFeatures.singleFeatureComparison([glucoseFeatures, lactateFeatures, uricAcidFeatures], [featureLabelsGlucose, featureLabelsLactate, featureLabelsUricAcid], ["Glucose", "Lactate", "UricAcid"], featureNames)
            
        
        # import matplotlib.pyplot as plt
        # colorList = ['k', 'r', 'b']
        # labels = [featureLabelsGlucose, featureLabelsLactate, featureLabelsUricAcid]
        # for i, key in enumerate(["Glucose", "Lactate", "Uric Acid"]):
        #     for dataInd, data in enumerate(chemicalProcessing.peakData[key]):
        #         scaleData1 = 1/max(data[1])
        #         plt.plot(data[0], scaleData1*data[1], colorList[labels[i][dataInd]])
        #     plt.show()
        
       # sys.exit()
      
        
        if analyzeTogether:
            # Machine Learning File/Model Paths + Titles
            modelPath = "./Helper Files/Machine Learning Modules/Models/chemicalModel_RF.pkl"
            modelType = "RF"  # Machine Learning Options: NN, RF, LR, KNN, SVM
            machineLearningClasses = ["Cold", "Exercise", "VR"]
            # Specify if We Are Saving the Model
            saveModel = False
            saveDataFolder = saveDataFolderChemical + "Machine Learning/" + modelType + "/"
    
            signalLabels = featureLabelsGlucose
            signalData = np.concatenate((glucoseFeatures, lactateFeatures, uricAcidFeatures), 1); 
            featureLabels = featureNames
            # featureLabels = []
            # for i, chemical in enumerate(["Glucose", "Lactate", "Uric Acid"]):
            #     for name in featureNames:
            #         featureLabels.append(name + "_" + chemical)
            signalData = np.array(signalData); signalLabels = np.array(signalLabels); featureLabels = np.array(featureLabels)
                        
            # Get the Machine Learning Module
            performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = len(featureLabels), machineLearningClasses = machineLearningClasses, saveDataFolder = saveDataFolder)
            predictionModel = performMachineLearning.predictionModel
            
            modelScores_Single0 = []
            modelScores_Single1 = []
            modelScores_Single2 = []
            modelScores_SingleTotal = []
            for featureInd in range(len(featureLabels)):
                featureRow = featureLabels[featureInd]
            
                signalDataCull = np.reshape(signalData[:,featureInd], (-1,1))
            
                performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = 1, machineLearningClasses = machineLearningClasses, saveDataFolder = saveDataFolder + featureRow + "/")
                
                modelScore = performMachineLearning.scoreClassificationModel(signalDataCull, signalLabels)
                
                modelScores_Single0.append(modelScore[0])
                modelScores_Single1.append(modelScore[1])
                modelScores_Single2.append(modelScore[2])
                modelScores_SingleTotal.append(modelScore[3])
                
            excelProcessing.processMLData().saveFeatureComparison([modelScores_Single0], [], featureLabels, saveDataFolderChemical, "Single Feature Accuracy.xlsx", sheetName = "Feature Comparison Cold")
            excelProcessing.processMLData().saveFeatureComparison([modelScores_Single1], [], featureLabels, saveDataFolderChemical, "Single Feature Accuracy.xlsx", sheetName = "Feature Comparison Excersize")
            excelProcessing.processMLData().saveFeatureComparison([modelScores_Single2], [], featureLabels, saveDataFolderChemical, "Single Feature Accuracy.xlsx", sheetName = "Feature Comparison VR")
            excelProcessing.processMLData().saveFeatureComparison([modelScores_SingleTotal], [], featureLabels, saveDataFolderChemical, "Single Feature Accuracy.xlsx", sheetName = "Feature Comparison Total")
            
            # Train the Data on the Gestures
            performMachineLearning.trainModel(signalData, signalLabels, featureLabels) 
            performMachineLearning.predictionModel.scoreModel(signalData, signalLabels)
            
            
            sys.exit()
            modelScores = np.zeros((len(featureLabels), len(featureLabels)))
            for featureIndRow in range(len(featureLabels)):
                featureRow = featureLabels[featureIndRow]
                for featureIndCol in range(len(featureLabels)):
                    featureCol = featureLabels[featureIndCol]
                    
                    signalDataCull = np.stack((signalData[:,featureIndRow], signalData[:,featureIndCol]), 1)
                    
                    performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = 2, machineLearningClasses = machineLearningClasses, saveDataFolder = saveDataFolder + featureRow + "_" + featureCol + "/")
                    modelScore = performMachineLearning.trainModel(signalDataCull, signalLabels, returnScore = True)
                    modelScores[featureIndRow][featureIndCol] = modelScore
            excelProcessing.processMLData().saveFeatureComparison(modelScores, featureLabels, featureLabels, saveDataFolderChemical, "Pairwise Feature Accuracy.xlsx", sheetName = "Feature Comparison")
            
        
        sys.exit()
        
        if analyzeFeatures:
            analyzeFeatures = featureAnalysis.featureAnalysis([], [], featureNames, [1110, 1110+60*3], saveDataFolderChemical)
            analyzeFeatures.singleFeatureComparison([glucoseFeatures, lactateFeatures, uricAcidFeatures], [featureLabelsGlucose,featureLabelsLactate,featureLabelsUricAcid], ["Glucose", "Lactate", "UricAcid"], featureNames)
            
            if analyzeTogether:
                analyzeFeatures.featureComparison(lactateFeatures, lactateFeatures, featureLabelsGlucose, lactateNames, lactateNames, 'Lactate', 'Lactate')
                analyzeFeatures.featureComparison(uricAcidFeatures, uricAcidFeatures, featureLabelsGlucose, glucoseNames, glucoseNames, 'Glucose', 'Glucose')
                analyzeFeatures.featureComparison(glucoseFeatures, glucoseFeatures, featureLabelsGlucose, uricAcidNames, uricAcidNames, 'Uric Acid', 'Uric Acid')
                analyzeFeatures.featureComparison(lactateFeatures, uricAcidFeatures, featureLabelsGlucose, lactateNames, uricAcidNames, 'Lactate', 'Uric Acid')
                analyzeFeatures.featureComparison(glucoseFeatures, uricAcidFeatures, featureLabelsGlucose, glucoseNames, uricAcidNames, 'Glucose', 'Uric Acid')
                analyzeFeatures.featureComparison(lactateFeatures, glucoseFeatures, featureLabelsGlucose, lactateNames, glucoseNames, 'Lactate', 'Glucose')
            
            analyzeFeatures.correlationMatrix(np.concatenate((glucoseFeatures, lactateFeatures, uricAcidFeatures), 0), featureNames)
        
        sys.exit()
        
        
        
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
        
        # Process the Data
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
        performMachineLearning.trainModel(signalData, signalLabels, pulseFeatures)
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
    
    
    
import matplotlib.pyplot as plt
a = True

for chemicalFile in chemicalFiles:
    # Read in the Chemical Data from Excel
    timePoints, chemicalData = excelDataChemical.getData(chemicalFile, testSheetNum = 0)
    glucose, lactate, uricAcid = chemicalData # Extract the Specific Chemicals
    
    colors = ['k','r', 'b']
    
    labels = ['glucose', 'lactate', 'uricAcid']
    for i in range(3):
        sp = np.fft.fft(chemicalData[i])
        freq = np.fft.fftfreq(len(timePoints))
        if a:
            plt.plot(freq, abs(sp.conjugate()), colors[i], label=labels[i])
            if i == 2:
                a = False
        else:
            plt.plot(freq, abs(sp.conjugate()), colors[i])
        #plt.plot(freq, sp.imag, 'r')
        plt.xlim(-0.05,0.05); plt.ylim(0,100)
plt.legend()
plt.show()
"""
