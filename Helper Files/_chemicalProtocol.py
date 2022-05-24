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
sys.path.append('./Data Aquisition and Analysis/')  # Folder with All the Helper Files
import excelProcessing

# Import Analysis Files (And Their Locations)
sys.path.append('./Data Aquisition and Analysis/_Analysis Protocols')  # Folder with All the Helper Files
import chemicalAnalysis

# Import Machine Learning Files (And They Location)
sys.path.append("./Machine Learning/")
# Import Files for Machine Learning
import machineLearningMain  # Class Header for All Machine Learning
import featureAnalysis  # Functions for Feature Analysis


if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #    
    
    # Program Flags
    trainModel = False             # Train a Model with the Collected Features.
    
    # Data Collection Parameters
    useAllFiles = True             # If True, Read in All the Tabulated Data in the Input Folder.
    # Analysis Flags
    plotAnalysis = False           # Plot the Results of Each Chemical Analysis.
    plotFeatures = True            # Plot the Collected Features.
    saveAnalysis = True            # Save the Analyzed Data: The Peak Features
    
    # Analysis Parameters
    stimulusTimes = [None, 1000 + 60*4]    # The [Beginning, End] of the Stimulus in Seconds; Type List.
    unitOfData = "uF"                    # Specify the Unit the Data is Represented as: ['F', 'mF', 'uF', 'nF', 'pF', 'fF']
    
    if useAllFiles:
        # Specify the Location of the Input Data
        inputFolder = '../Input Data/Chemical Data/'   # Path to ALL the Excel Data. The Path Must End with '/'
    else:
        chemicalFiles = ["../Input Data/Chemical Data/20211022 cold enzymatic_jose - aligned.xlsx"]    # A List of the Excel Data ('.xls' or '.xlsx') to Analyze.

    # ---------------------------------------------------------------------- #
    # ------------------------- Preparation Steps -------------------------- #

    # Create all the Analysis Instances
    excelProcessingChemical = excelProcessing.processChemicalData()
    chemicalAnalysisProtocol = chemicalAnalysis.signalProcessing(stimulusTimes, stimulusBuffer = 500, plotData = True)
    
    # Find the Scale Factor for the Data
    scaleFactorMap = {'F': 1, 'mF': 10**-3, 'uF': 10**-6, 'nF': 10**-9, 'pF': 10**-12, 'fF': 10**-15}
    scaleFactor = 1/scaleFactorMap[unitOfData]
    
    # Collect the File Names, If Using Every File in the Folder
    if useAllFiles:
        chemicalFiles = []
        for file in os.listdir(inputFolder):
           if file.endswith(("xlsx", "xls")) and not file.startswith(("$", '~')):
               chemicalFiles.append(inputFolder + file)
        chemicalFiles = natsorted(chemicalFiles)
    
    if trainModel or plotFeatures:
        # Specify the Paths to the Chemical Feature Names
        glucoseFeaturesFile = "./Machine Learning/Compiled Feature Names/glucoseFeatureLabels.txt"
        lactateFeaturesFile = "./Machine Learning/Compiled Feature Names/lactateFeatureLabels.txt"
        uricAcidFeaturesFile = "./Machine Learning/Compiled Feature Names/uricAcidFeatureLabels.txt"
        # Extract the Chemical Features we are Using
        glucoseFeatureLabels = excelProcessingChemical.extractFeatures(glucoseFeaturesFile, prependedString = "peakFeatures.extend([")
        lactateFeatureLabels = excelProcessingChemical.extractFeatures(lactateFeaturesFile, prependedString = "peakFeatures.extend([")
        uricAcidFeatureLabels = excelProcessingChemical.extractFeatures(uricAcidFeaturesFile, prependedString = "peakFeatures.extend([")
        
    # Saves the Analyzed Data
    if saveAnalysis:
        saveDataFolderChemical = inputFolder + "Chemical Analysis/"     # Data Folder to Save the Data; MUST END IN '/'

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
        pulseFeatures = []
        performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = len(pulseFeatures), gestureClasses = machineLearningClasses, saveDataFolder = saveDataFolder)
        predictionModel = performMachineLearning.predictionModel

    # ---------------------------------------------------------------------- #
    # ------------------------ Analyze Chemical Data ----------------------- #
    
    # Flag to Determine Whether We Can Cross-Compare Features
    analyzeTogether = glucoseFeatureLabels == lactateFeatureLabels == uricAcidFeatureLabels
    
    sys.exit()
     
    featureLabels = []
    for chemicalFile in chemicalFiles:
        # Read in the Chemical Data from Excel
        timePoints, chemicalData = chemicalAnalysisProtocol.getData(chemicalFile, testSheetNum = 0)
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
           excelProcessingChemical.analyzeChemicals(timePoints, glucose, lactate, uricAcid, featureLabel, analyzeTogether)
        
        # if excelProcessingChemical.continueAnalysis:
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
     
    glucoseFeatures = np.array(excelProcessingChemical.glucoseFeatures)
    lactateFeatures = np.array(excelProcessingChemical.lactateFeatures)
    uricAcidFeatures = np.array(excelProcessingChemical.uricAcidFeatures)
     
    featureLabelsGlucose = np.array(excelProcessingChemical.featureLabelsGlucose)
    featureLabelsLactate = np.array(excelProcessingChemical.featureLabelsLactate)
    featureLabelsUricAcid = np.array(excelProcessingChemical.featureLabelsUricAcid)
 
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
     
     
    chemicalFeatures = []
    chemicalFeatures.extend(glucoseFeatures)
    chemicalFeatures.extend(lactateFeatures)
    chemicalFeatures.extend(uricAcidFeatures)

     
    analyzeFeatures = featureAnalysis.featureAnalysis([], [], featureNames, [1110, 1110+60*3], saveDataFolderChemical)
    analyzeFeatures.singleFeatureComparison([glucoseFeatures, lactateFeatures, uricAcidFeatures], [featureLabelsGlucose, featureLabelsLactate, featureLabelsUricAcid], ["Glucose", "Lactate", "UricAcid"], featureNames)
         
     
    # import matplotlib.pyplot as plt
    # colorList = ['k', 'r', 'b']
    # labels = [featureLabelsGlucose, featureLabelsLactate, featureLabelsUricAcid]
    # for i, key in enumerate(["Glucose", "Lactate", "Uric Acid"]):
    #     for dataInd, data in enumerate(excelProcessingChemical.peakData[key]):
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
     
        
    # ---------------------------------------------------------------------- #
    #                        Train the Model                            #
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
    

    
