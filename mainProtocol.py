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

# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# Basic Modules
import os
import sys
import shutil
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from natsort import natsorted
# Import Data Extraction Files (And Their Location)
sys.path.append('./Helper Files/Data Aquisition and Analysis/')  # Folder with All the Helper Files
import excelProcessing

# Import Analysis Files (And Their Locations)
sys.path.append('./Data Aquisition and Analysis/_Analysis Protocols')  # Folder with All the Helper Files
sys.path.append('./Helper Files/Data Aquisition and Analysis/_Analysis Protocols')  # Folder with All the Helper Files
import temperatureAnalysis
import chemicalAnalysis
import pulseAnalysis
import gsrAnalysis

# Import Machine Learning Files (And They Location)
sys.path.append("./Helper Files/Machine Learning/")
import machineLearningMain   # Class Header for All Machine Learning
import featureAnalysis       # Functions for Feature Analysis

# -------------------------------------------------------------------------- #
# --------------------------- Program Starts Here -------------------------- #

if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #    User Parameters to Edit (More Complex Edits are Inside the Files)   #
    # ---------------------------------------------------------------------- #    

    # Analysis Parameters
    timePermits = False                     # Construct Plots that Take a Long TIme
    plotFeatures = False                    # Plot the Analyzed Features
    saveAnalysis = True                     # Save the Analyzed Data: The Peak Features for Each Well-Shaped Pulse
    stimulusTimes = [1000, 1000 + 60*4]     # The [Beginning, End] of the Stimulus in Seconds; Type List.

    # Specify Which Signals to Use
    extractGSR = False
    extractPulse = False
    extractChemical = False
    extractTemperature = False
    # Reanalyze Peaks from Scratch (Don't Use Saved Features)
    reanalyzeData_GSR = False
    reanalyzeData_Pulse = False
    reanalyzeData_Chemical = False    
    reanalyzeData_Temperature = False

    # Specify the Unit of Data for Each 
    unitOfData_GSR = "micro"                # Specify the Unit the Data is Represented as: ['', 'milli', 'micro', 'nano', 'pico', 'fempto']
    unitOfData_Pulse = ""                   # Specify the Unit the Data is Represented as: ['', 'milli', 'micro', 'nano', 'pico', 'fempto']
    unitOfData_Chemical = "micro"           # Specify the Unit the Data is Represented as: ['', 'milli', 'micro', 'nano', 'pico', 'fempto']
    unitOfData_Temperature = ""             # Specify the Unit the Data is Represented as: ['', 'milli', 'micro', 'nano', 'pico', 'fempto']

    # Specify the Location of the Subject Files
    dataFolderWithSubjects = './Input Data/All Data/Cleaned data for ML Full/'  # Path to ALL the Subject Data. The Path Must End with '/'
    compiledFeatureNamesFolder = "./Helper Files/Machine Learning/Compiled Feature Names/All Features/"

    # Specify the Stressors/Sensoirs Used in this Experiment
    listOfStressors = ['cpt', 'exercise', 'vr']         # This Keyword MUST be Present in the Filename
    listOfSensors = ['pulse', 'enzym', 'gsr', 'temp']   # This Keyword MUST be Present in the Filename
    
    # ---------------------------------------------------------------------- #
    # ------------------------- Preparation Steps -------------------------- #
    
    # Create Instance of Excel Processing Methods
    excelProcessingGSR = excelProcessing.processGSRData()
    excelProcessingPulse = excelProcessing.processPulseData()
    excelProcessingChemical = excelProcessing.processChemicalData()
    excelProcessingTemperature = excelProcessing.processTemperatureData()

    # Create Instances of all Analysis Protocols
    gsrAnalysisProtocol = gsrAnalysis.signalProcessing(stimulusTimes)
    pulseAnalysisProtocol = pulseAnalysis.signalProcessing()
    chemicalAnalysisProtocol = chemicalAnalysis.signalProcessing(stimulusTimes, plotData = True)
    temperatureAnalysisProtocol = temperatureAnalysis.signalProcessing(stimulusTimes)

    subjectFolderPaths = []
    # Extract the Files for from Each Subject
    for folder in os.listdir(dataFolderWithSubjects):
        if 'subject' not in folder.lower() and not folder.startswith(("$", '~', '_')):
            continue
        
        subjectFolder = dataFolderWithSubjects + folder + "/"
        # Check Whether the Path is a Folder (Exclude Cache Folders)
        if os.path.isdir(subjectFolder):
                # Save the Folder's path for Later Analysis
                subjectFolderPaths.append(subjectFolder)
    # Sort the Folders for Ease in Debugging
    subjectFolderPaths = natsorted(subjectFolderPaths)

    # Define Map of Units to Scale Factors
    scaleFactorMap = {'': 1, 'milli': 10**-3, 'micro': 10**-6, 'nano': 10**-9, 'pico': 10**-12, 'fempto': 10**-15}
    # Find the Scale Factor for the Data
    scaleFactor_GSR = scaleFactorMap[unitOfData_GSR]
    scaleFactor_Pulse = scaleFactorMap[unitOfData_Pulse]
    scaleFactor_Chemical = scaleFactorMap[unitOfData_Chemical]
    scaleFactor_Temperature = scaleFactorMap[unitOfData_Temperature]
    
    # ---------------------------------------------------------------------- #
    # ------------------------ Specify the Features ------------------------ #
        
    # Compile Features
    featureNames = []        # Compile Feature Names
    scoreFeatureLabels = []  # Compile Stress Scores
    
    if extractPulse:
        # Specify the Paths to the Pulse Feature Names
        pulseFeaturesFile_StressLevel = compiledFeatureNamesFolder + "pulseFeatureNames_StressLevel.txt"
        pulseFeaturesFile_SignalIncrease = compiledFeatureNamesFolder + "pulseFeatureNames_SignalIncrease.txt"
        # Extract the Pulse Feature Names we are Using
        pulseFeatureNames_StressLevel = excelProcessingPulse.extractFeatures(pulseFeaturesFile_StressLevel, prependedString = "pulseFeatures.extend([", appendToName = "_StressLevel")[1:]
        pulseFeatureNames_SignalIncrease = excelProcessingPulse.extractFeatures(pulseFeaturesFile_SignalIncrease, prependedString = "pulseFeatures.extend([", appendToName = "_SignalIncrease")[1:]
        # Combine all the Features
        pulseFeatureNames = []
        pulseFeatureNames.extend(pulseFeatureNames_StressLevel)
        pulseFeatureNames.extend(pulseFeatureNames_SignalIncrease)
        # Get Pulse Names Without Anything Appended
        pulseFeatureNamesFull = excelProcessingPulse.extractFeatures(pulseFeaturesFile_SignalIncrease, prependedString = "pulseFeatures.extend([", appendToName = "")
        pulseFeatureNamesFull.extend(excelProcessingPulse.extractFeatures(pulseFeaturesFile_StressLevel, prependedString = "pulseFeatures.extend([", appendToName = "")[1:])
        # Create Data Structure to Hold the Features
        pulseFeatures = []
        pulseFeatureLabels = []  
        featureNames.extend(pulseFeatureNames)
    
    if extractChemical:
        # Specify the Paths to the Chemical Feature Names
        glucoseFeaturesFile = compiledFeatureNamesFolder + "glucoseFeatureNames.txt"
        lactateFeaturesFile = compiledFeatureNamesFolder + "lactateFeatureNames.txt"
        uricAcidFeaturesFile = compiledFeatureNamesFolder + "uricAcidFeatureNames.txt"
        # Extract the Chemical Feature Names we are Using
        glucoseFeatureNames = excelProcessingChemical.extractFeatures(glucoseFeaturesFile, prependedString = "peakFeatures.extend([", appendToName = '_Glucose')[1:]
        lactateFeatureNames = excelProcessingChemical.extractFeatures(lactateFeaturesFile, prependedString = "peakFeatures.extend([", appendToName = '_Lactate', )[1:]
        uricAcidFeatureNames = excelProcessingChemical.extractFeatures(uricAcidFeaturesFile, prependedString = "peakFeatures.extend([", appendToName = '_UricAid', )[1:]
        # Combine all the Features
        chemicalFeatureNames = []
        chemicalFeatureNames.extend(glucoseFeatureNames)
        chemicalFeatureNames.extend(lactateFeatureNames)
        chemicalFeatureNames.extend(uricAcidFeatureNames)
        # Create Data Structure to Hold the Features
        chemicalFeatures = []
        chemicalFeatureLables = []
        featureNames.extend(chemicalFeatureNames)
        
    if extractGSR:
        # Specify the Paths to the GSR Feature Names
        gsrFeaturesFile = compiledFeatureNamesFolder + "gsrFeatureNames.txt"
        # Extract the GSR Feature Names we are Using
        gsrFeatureNames = excelProcessingGSR.extractFeatures(gsrFeaturesFile, prependedString = "gsrFeatures.extend([", appendToName = '_GSR')[1:]
        # Create Data Structure to Hold the Features
        gsrFeatures = []
        gsrFeatureLables = []
        featureNames.extend(gsrFeatureNames)
        
    if extractTemperature:
        # Specify the Paths to the Temperature Feature Names
        temperatureFeaturesFile = compiledFeatureNamesFolder + "temperatureFeatureNames.txt"
        # Extract the GSR Feature Names we are Using
        temperatureFeatureNames = excelProcessingTemperature.extractFeatures(temperatureFeaturesFile, prependedString = "temperatureFeatures.extend([", appendToName = '')[1:]
        # Create Data Structure to Hold the Features
        temperatureFeatures = []
        temperatureFeatureLables = []
        featureNames.extend(temperatureFeatureNames)
        
    # ---------------------------------------------------------------------- #
    # -------------------- Data Collection and Analysis -------------------- #
    
    # Loop Through Each Subject
    for subjectFolder in subjectFolderPaths:

        # CPT Score
        cptScore = subjectFolder.split("CPT")
        if len(cptScore) == 1:
            cptScore = None
        else:
            cptScore = int(cptScore[1][0:2])
        # Excersize Score
        exerciseScore = subjectFolder.split("Exercise")
        if len(exerciseScore) == 1:
            exerciseScore = None
        else:
            exerciseScore = int(exerciseScore[1][0:2])
        # CPT Score
        vrScore = subjectFolder.split("VR")
        if len(vrScore) == 1:
            vrScore = None
        else:
            vrScore = int(vrScore[1][0:2])
            
        # Label the Score of the File
        scoreLabels = [cptScore, exerciseScore, vrScore]
        scoreFeatureLabels.extend([None]*len(scoreLabels))
        
        # ------- Organize the Files within Each Stressor and Sensor ------- #
        
        # Organize/Label the Files in the Folder: Pulse, Chemical, GSR, Temp
        fileMap = [[None for _ in  range(len(listOfSensors))] for _ in range(len(listOfStressors))]
        # Loop Through Each File and Label the Stressor Analyzed
        for file in os.listdir(subjectFolder):

            # Find the Type of Stressor in the File
            for stressorInd, stressor in enumerate(listOfStressors):
                if stressor in file.lower():
                    
                    # Extract the Stress Information from the Filename
                    if scoreFeatureLabels[stressorInd - len(listOfStressors)] == None:
                        scoreFeatureLabels[stressorInd - len(listOfStressors)] = scoreLabels[stressorInd]
                
                    # Find the Type of Sensor the File Used
                    for sensorInd, sensor in enumerate(listOfSensors):
                        if sensor in file.lower():
                            
                            # Organize the Files by Their Stressor and Sensor Type 
                            if sensor == "pulse":
                                fileMap[stressorInd][sensorInd] = subjectFolder + file + "/";
                            else:
                                fileMap[stressorInd][sensorInd] = subjectFolder + file;
                            break
                    break
        fileMap = np.array(fileMap)
        
        # ------------------------- Pulse Analysis ------------------------- #
        
        if extractPulse:
            # Extract the Pulse Folder Paths in the Map
            pulseFolders = fileMap[:, listOfSensors.index("pulse")]
            # Loop Through the Pulse Data for Each Stressor
            for featureLabel, pulseFolder in enumerate(pulseFolders):
                if pulseFolder == None:
                    # Compile the Featues into One Array
                    pulseFeatureLabels.append(None)
                    pulseFeatures.append([None]*len(pulseFeatureNames))
                    continue
                
                if not reanalyzeData_Pulse and os.path.isfile(pulseFolder + "/Pulse Analysis/Compiled Data in Excel/Feature List.xlsx"):
                    pulseFeatureList = excelProcessingPulse.getSavedFeatures(pulseFolder + "/Pulse Analysis/Compiled Data in Excel/Feature List.xlsx")
                    featureTimes = pulseFeatureList[:,0]
                    pulseFeatureList = pulseFeatureList[:,1:]
                else:
                    pulseExcelFiles = []
                    # Collect all the Pulse Files for the Stressor
                    for file in os.listdir(pulseFolder):
                        file = file.decode("utf-8") if type(file) == type(b'') else file
                        if file.endswith(("xlsx", "xls")) and not file.startswith(("$", '~')):
                            pulseExcelFiles.append(pulseFolder + file)
                    pulseExcelFiles = natsorted(pulseExcelFiles)
                
                    # Loop Through Each Pulse File
                    pulseAnalysisProtocol.resetGlobalVariables()
                    for pulseExcelFile in pulseExcelFiles:
                        
                        # Read Data from Excel
                        time, signalData = excelProcessingPulse.getData(pulseExcelFile, testSheetNum = 0)
                        signalData = signalData*scaleFactor_Pulse
                                        
                        # Calibrate Systolic and Diastolic Pressure
                        fileBasename = os.path.basename(pulseExcelFile)
                        pressureInfo = fileBasename.split("SYS")
                        if len(pressureInfo) > 1 and pulseAnalysisProtocol.systolicPressure0 == None:
                            pressureInfo = pressureInfo[-1].split(".")[0]
                            systolicPressure0, diastolicPressure0 = pressureInfo.split("_DIA")
                            pulseAnalysisProtocol.setPressureCalibration(float(systolicPressure0), float(diastolicPressure0))
                        
                        # Check Whether the StartTime is Specified in the File
                        if fileBasename.lower() in ["cpt", "exercise", "vr", "start"] and not stimulusTimes[0]:
                            stimulusTimes[0] = pulseAnalysisProtocol.timeOffset
                        
                        # Seperate Pulses, Perform Indivisual Analysis, and Extract Features
                        pulseAnalysisProtocol.analyzePulse(time, signalData, minBPM = 30, maxBPM = 180)
                    
                    savePulseDataFolder = pulseFolder + "Pulse Analysis/"    # Data Folder to Save the Data; MUST END IN '/'
                    # Remove Previous Analysis if Present
                    if os.path.isdir(savePulseDataFolder):
                        shutil.rmtree(savePulseDataFolder)
                    pulseAnalysisProtocol.featureListExact = np.array(pulseAnalysisProtocol.featureListExact)
                    # Save the Features and Filtered Data
                    saveCompiledDataPulse = savePulseDataFolder + "Compiled Data in Excel/"
                    excelProcessingPulse.saveResults(pulseAnalysisProtocol.featureListExact, pulseFeatureNamesFull, saveCompiledDataPulse, "Feature List.xlsx", sheetName = "Pulse Features")
                    excelProcessingPulse.saveFilteredData(pulseAnalysisProtocol.time, pulseAnalysisProtocol.signalData, pulseAnalysisProtocol.filteredData, saveCompiledDataPulse, "Filtered Data.xlsx", "Filtered Data")
                    # Plot the Features in Time
                    if timePermits:
                        plotPulseFeatures = featureAnalysis.featureAnalysis(pulseAnalysisProtocol.featureListExact[:,0], pulseAnalysisProtocol.featureListExact[:,1:], pulseFeatureNamesFull[1:], stimulusTimes, savePulseDataFolder)
                        plotPulseFeatures.singleFeatureAnalysis()
                    
                    # Compile the Features from the Data
                    featureTimes = pulseAnalysisProtocol.featureListExact[:,0]
                    pulseFeatureList = np.array(pulseAnalysisProtocol.featureListAverage)
                    # Assert That There are Equal Features and Feature Times
                    assert len(featureTimes) == len(pulseFeatureList)
    
                # Quick Check that All Points Have the Correct Number of Features
                for feature in pulseFeatureList:
                    assert len(feature) == len(pulseFeatureNames)
                
                # Downsize the Features into One Data Point
                # ********************************
                # FInd the Indices of the Stimuli
                startStimulusInd = np.argmin(abs(featureTimes - stimulusTimes[0]))
                endStimulusInd = np.argmin(abs(featureTimes - stimulusTimes[1]))
                
                # Caluclate the Baseline/Stress Levels
                restValues = stats.trim_mean(pulseFeatureList[ int(startStimulusInd/4):int(3*startStimulusInd/4),:], 0.4)
                stressValues = stats.trim_mean(pulseFeatureList[ endStimulusInd-100:endStimulusInd+100,: ], 0.4)
                stressElevation = stressValues - restValues
                # Calculate the Stress Rise/Fall
                stressSlopes = np.polyfit(featureTimes[startStimulusInd:endStimulusInd], pulseFeatureList[ startStimulusInd:endStimulusInd,: ], 1)[0]
    
                # Organize the Signals
                pulseFeatures_StressLevel = stressValues[0:len(pulseFeatureNames_StressLevel)]
                pulseFeatures_SignalIncrease = stressElevation[len(pulseFeatureNames_StressLevel):]
                # Compile the Signals
                subjectPulseFeatures = []
                subjectPulseFeatures.extend(pulseFeatures_StressLevel)
                subjectPulseFeatures.extend(pulseFeatures_SignalIncrease)
                # Assert the Number of Signals are Correct
                assert len(subjectPulseFeatures) == len(pulseFeatureNames)
                assert len(pulseFeatures_StressLevel) == len(pulseFeatureNames_StressLevel)
                assert len(pulseFeatures_SignalIncrease) == len(pulseFeatureNames_SignalIncrease)
                # ********************************
                
                # Compile the Featues into One Array
                pulseFeatureLabels.append(featureLabel)
                pulseFeatures.append(subjectPulseFeatures)
    
        # ------------------------ Chemical Analysis ----------------------- #
        
        if extractChemical:
            # Extract the Pulse Folder Paths in the Map
            chemicalFiles = fileMap[:, listOfSensors.index("enzym")]
            # Loop Through the Pulse Data for Each Stressor
            for featureLabel, chemicalFile in enumerate(chemicalFiles):
                if chemicalFile == None:
                    chemicalFeatureLables.append(None)
                    chemicalFeatures.append([None]*len(chemicalFeatureNames))
                    continue
                # Extract the Specific Chemical Filename
                chemicalFilename = os.path.basename(chemicalFile[:-1]).split(".")[0]
                saveCompiledDataChemical = subjectFolder + "Chemical Analysis/Compiled Data in Excel/" + chemicalFilename + "/"
    
                if not reanalyzeData_Chemical and os.path.isfile(saveCompiledDataChemical + "Feature List.xlsx"):
                    subjectChemicalFeatures = excelProcessingChemical.getSavedFeatures(saveCompiledDataChemical + "Feature List.xlsx")[0]
                    
                    # Organize the Features
                    glucoseFeatures = subjectChemicalFeatures[0:len(glucoseFeatureNames)]
                    lactateFeatures = subjectChemicalFeatures[len(glucoseFeatureNames):len(glucoseFeatureNames) + len(lactateFeatureNames)]
                    uricAcidFeatures = subjectChemicalFeatures[len(lactateFeatureNames) + len(glucoseFeatureNames):]
                    
                    # Quick Check that All Points Have the Correct Number of Features
                    assert len(subjectChemicalFeatures) == len(chemicalFeatureNames)
                    assert len(glucoseFeatures) == len(glucoseFeatureNames)
                    assert len(lactateFeatures) == len(lactateFeatureNames)
                    assert len(uricAcidFeatures) == len(uricAcidFeatureNames)
                else:
                    # Read in the Chemical Data from Excel
                    timePoints, chemicalData = excelProcessingChemical.getData(chemicalFile, testSheetNum = 0)
                    glucose, lactate, uricAcid = chemicalData*scaleFactor_Chemical # Extract the Specific Chemicals
                    lactate = lactate/1000 # Correction on Lactate Data
                    
                    # Cull Subjects with Missing Data
                    if len(glucose) == 0 or len(lactate) == 0 or len(uricAcid) == 0:
                        print("Missing Chemical Data in Folder:", subjectFolder)
                        sys.exit()
            
                    # Compile the Features from the Data
                    chemicalAnalysisProtocol.resetGlobalVariables()
                    chemicalAnalysisProtocol.analyzeChemicals(timePoints, glucose, lactate, uricAcid, featureLabel)
                    # Get the ChemicalFeatures
                    glucoseFeatures = chemicalAnalysisProtocol.glucoseFeatures[0]
                    lactateFeatures = chemicalAnalysisProtocol.lactateFeatures[0]
                    uricAcidFeatures = chemicalAnalysisProtocol.uricAcidFeatures[0]
                    chemicalAnalysisProtocol.resetGlobalVariables()
                    # Verify that Features were Found in for All Chemicals
                    if len(glucoseFeatures) == 0 or len(lactateFeatures) == 0 or len(uricAcidFeatures) == 0:
                        print("No Features Found in Some Chemical Data in Folder:", subjectFolder)
                        sys.exit()   
                        
                    # Quick Check that All Points Have the Correct Number of Features
                    assert len(glucoseFeatures) == len(glucoseFeatureNames)
                    assert len(lactateFeatures) == len(lactateFeatureNames)
                    assert len(uricAcidFeatures) == len(uricAcidFeatureNames)
                    
                    # Organize the Chemical Features
                    subjectChemicalFeatures = []
                    subjectChemicalFeatures.extend(glucoseFeatures)
                    subjectChemicalFeatures.extend(lactateFeatures)
                    subjectChemicalFeatures.extend(uricAcidFeatures)
                    
                    # Save the Features and Filtered Data
                    excelProcessingChemical.saveResults([subjectChemicalFeatures], chemicalFeatureNames, saveCompiledDataChemical, "Feature List.xlsx", sheetName = "Chemical Features")
                
                # Compile the Featues into One Array
                chemicalFeatureLables.append(featureLabel)
                chemicalFeatures.append(subjectChemicalFeatures)
        
        # -------------------------- GSR Analysis -------------------------- #
        
        if extractGSR:
            colorList = ['k', 'tab:blue', 'tab:red']
            # Extract the Pulse Folder Paths in the Map
            gsrFiles = fileMap[:, listOfSensors.index("gsr")]
            # Loop Through the Pulse Data for Each Stressor
            for featureLabel, gsrFile in enumerate(gsrFiles):
                if gsrFile == None:
                    gsrFeatureLables.append(None)
                    gsrFeatures.append([None]*len(gsrFeatureNames))
                    continue
                # Extract the Specific GSR Filename
                gsrFilename = os.path.basename(gsrFile[:-1]).split(".")[0]
                saveCompiledDataGSR = subjectFolder + "GSR Analysis/Compiled Data in Excel/" + gsrFilename + "/"
    
                if not reanalyzeData_GSR and os.path.isfile(saveCompiledDataGSR + "Feature List.xlsx"):
                    subjectGSRFeatures = excelProcessingChemical.getSavedFeatures(saveCompiledDataGSR + "Feature List.xlsx")[0]
                    # Quick Check that All Points Have the Correct Number of Features
                    assert len(subjectGSRFeatures) == len(gsrFeatureNames)
                else:
                    # Read in the GSR Data from Excel
                    excelDataGSR = excelProcessing.processGSRData()
                    timeGSR, currentGSR = excelDataGSR.getData(gsrFile, testSheetNum = 0, method = "processed")
                    currentGSR = currentGSR*scaleFactor_GSR # Get Data into micro-Ampes

                    # Process the Data
                    subjectGSRFeatures = gsrAnalysisProtocol.analyzeGSR(timeGSR, currentGSR)
                    
                    # Quick Check that All Points Have the Correct Number of Features
                    assert len(subjectGSRFeatures) == len(gsrFeatureNames)
                    
                    # Save the Features and Filtered Data
                    excelProcessingGSR.saveResults([subjectGSRFeatures], gsrFeatureNames, saveCompiledDataGSR, "Feature List.xlsx", sheetName = "GSR Features")
                
                    #plt.plot(timeGSR, currentGSR/max(yData), colorList[featureLabel], alpha = 0.5, label = listOfStressors[featureLabel] + " Real")
                    #plt.plot(timeGSR, yData, colorList[featureLabel], label = listOfStressors[featureLabel])
                    
                # Compile the Featues into One Array
                gsrFeatureLables.append(featureLabel)
                gsrFeatures.append(subjectGSRFeatures)
        
        # ---------------------- Temperature Analysis ---------------------- #
        
        if extractTemperature:
            colorList = ['k', 'tab:blue', 'tab:red']
            # Extract the Pulse Folder Paths in the Map
            temperatureFiles = fileMap[:, listOfSensors.index("temp")]
            # Loop Through the Pulse Data for Each Stressor
            for featureLabel, temperatureFile in enumerate(temperatureFiles):
                if temperatureFile == None:
                    temperatureFeatureLables.append(None)
                    temperatureFeatures.append([None]*len(temperatureFeatureNames))
                    continue
                # Extract the Specific temperature Filename
                temperatureFilename = os.path.basename(temperatureFile[:-1]).split(".")[0]
                saveCompiledDataTemperature = subjectFolder + "temperature Analysis/Compiled Data in Excel/" + temperatureFilename + "/"
    
                if not reanalyzeData_Temperature and os.path.isfile(saveCompiledDataTemperature + "Feature List.xlsx"):
                    subjectTemperatureFeatures = excelProcessingChemical.getSavedFeatures(saveCompiledDataTemperature + "Feature List.xlsx")[0]
                    # Quick Check that All Points Have the Correct Number of Features
                    assert len(subjectTemperatureFeatures) == len(temperatureFeatureNames)
                else:
                    # Read in the temperature Data from Excel
                    excelDataTemperature = excelProcessing.processTemperatureData()
                    timeTemp, temperatureData = excelDataTemperature.getData(temperatureFile, testSheetNum = 0)
                    temperatureData = temperatureData*scaleFactor_Temperature # Get Data into micro-Ampes

                    # Process the Data
                    subjectTemperatureFeatures = temperatureAnalysisProtocol.analyzeTemperature(timeTemp, temperatureData)
                    
                    
                    plt.plot(timeTemp, temperatureData, colorList[featureLabel], alpha = 0.5, label = listOfStressors[featureLabel] + " Real")
                    plt.plot(timeTemp, subjectTemperatureFeatures, colorList[featureLabel], label = listOfStressors[featureLabel])
                    
                    
                    continue
                    # Quick Check that All Points Have the Correct Number of Features
                    assert len(subjectTemperatureFeatures) == len(temperatureFeatureNames)
                    
                    # Save the Features and Filtered Data
                    excelProcessingTemperature.saveResults([subjectTemperatureFeatures], temperatureFeatureNames, saveCompiledDataTemperature, "Feature List.xlsx", sheetName = "Temperature Features")
                
            plt.legend()
            plt.show()
                # # Compile the Featues into One Array
                # temperatureFeatureLables.append(featureLabel)
                # temperatureFeatures.append(subjectGSRFeatures)

    # ---------------------- Compile Features Together --------------------- #
    
    # Compile Labels
    allLabels = []
    allLabels.append(gsrFeatureLables) if extractGSR else None
    allLabels.append(pulseFeatureLabels) if extractPulse else None
    allLabels.append(chemicalFeatureLables) if extractChemical else None
    #allLabels.append(temperatureFeatureLables) if extractTemperature else None
    # Assert That We Have the Same Number of Points in Both
    for labelList in allLabels:
        assert len(labelList) == len(scoreFeatureLabels)
    # Do Not Continue if No Labels Found
    if len(allLabels) == 0:
        print("Please Specify Features to Extract"); sys.exit()
    allLabels = np.array(allLabels)
       
    # Compile Data for Machine Learning
    signalData = []; stressLabels = []; scoreLabels = []
    # Merge the Features
    for arrayInd in range(len(allLabels[0])):
        currentLabels = allLabels[:,arrayInd]
        stressLabel = scoreFeatureLabels[arrayInd]
        
        # If the Subject had All the Data for the Sensor.
        if None not in currentLabels and stressLabel != None:
            # Assert That the Features are the Same
            assert len(set(currentLabels)) == 1
            
            features = []
            # Compile the Features
            features.extend(gsrFeatures[arrayInd]) if extractGSR else None
            features.extend(pulseFeatures[arrayInd]) if extractPulse else None
            features.extend(chemicalFeatures[arrayInd]) if extractChemical else None
            #features.extend(temperatureFeatures[arrayInd]) if extractTemperature else None
            # Save the Compiled Features
            signalData.append(features)
            stressLabels.append(currentLabels[0])
            scoreLabels.append(stressLabel)
    signalData = np.array(signalData)
    stressLabels = np.array(stressLabels)
    scoreLabels = np.array(scoreLabels)
    print("Finished Collecting All the Data")
        
    # ----------------------- Extra Feature Analysis ----------------------- #
    print("\nPlotting Feature Comparison")
    
    if plotFeatures:
        if extractChemical:
            chemicalFeatures = np.array(chemicalFeatures)
            chemicalFeatureLables = np.array(chemicalFeatureLables)
            # Remove None Values
            chemicalFeatures_NonNone = chemicalFeatures[chemicalFeatureLables != np.array(None)]
            chemicalFeatureLables_NonNone = chemicalFeatureLables[chemicalFeatureLables != np.array(None)]
            # Organize the Features
            glucoseFeatures = chemicalFeatures_NonNone[:, 0:len(glucoseFeatureNames)]
            lactateFeatures = chemicalFeatures_NonNone[:, len(glucoseFeatureNames):len(glucoseFeatureNames) + len(lactateFeatureNames)]
            uricAcidFeatures = chemicalFeatures_NonNone[:, len(glucoseFeatureNames) + len(lactateFeatureNames):]
                        
            # Plot the Features within a Single Chemical
            analyzeFeatures = featureAnalysis.featureAnalysis([], [], chemicalFeatureNames, stimulusTimes, dataFolderWithSubjects + "Machine Learning/Compiled Chemical Feature Analysis/")
            analyzeFeatures.singleFeatureComparison([glucoseFeatures, lactateFeatures, uricAcidFeatures], [chemicalFeatureLables_NonNone, chemicalFeatureLables_NonNone, chemicalFeatureLables_NonNone], ["Glucose", "Lactate", "UricAcid"], chemicalFeatureNames)
            if timePermits:
                # Cross-Compare the Features Between Each Other
                analyzeFeatures.featureComparison(lactateFeatures, lactateFeatures, chemicalFeatureLables_NonNone, lactateFeatureNames, lactateFeatureNames, 'Lactate', 'Lactate')
                analyzeFeatures.featureComparison(glucoseFeatures, glucoseFeatures, chemicalFeatureLables_NonNone, glucoseFeatureNames, glucoseFeatureNames, 'Glucose', 'Glucose')
                analyzeFeatures.featureComparison(uricAcidFeatures, uricAcidFeatures, chemicalFeatureLables_NonNone, uricAcidFeatureNames, uricAcidFeatureNames, 'Uric Acid', 'Uric Acid')
                analyzeFeatures.featureComparison(lactateFeatures, uricAcidFeatures, chemicalFeatureLables_NonNone, lactateFeatureNames, uricAcidFeatureNames, 'Lactate', 'Uric Acid')
                analyzeFeatures.featureComparison(glucoseFeatures, uricAcidFeatures, chemicalFeatureLables_NonNone, glucoseFeatureNames, uricAcidFeatureNames, 'Glucose', 'Uric Acid')
                analyzeFeatures.featureComparison(lactateFeatures, glucoseFeatures, chemicalFeatureLables_NonNone, lactateFeatureNames, glucoseFeatureNames, 'Lactate', 'Glucose')
            
        if extractPulse:
            pulseFeatures = np.array(pulseFeatures)
            pulseFeatureLabels = np.array(pulseFeatureLabels)
            # Remove None Values
            pulseFeatures_NonNone = pulseFeatures[pulseFeatureLabels != np.array(None)]
            pulseFeatureLables_NonNone = pulseFeatureLabels[pulseFeatureLabels != np.array(None)]
            # Plot the Features within a Pulse
            analyzeFeatures = featureAnalysis.featureAnalysis([], [], featureNames, stimulusTimes, dataFolderWithSubjects + "Machine Learning/Compiled Pulse Feature Analysis/")
            analyzeFeatures.singleFeatureComparison([pulseFeatures_NonNone], [pulseFeatureLables_NonNone], ["Pulse"], featureNames)
            if timePermits:
                # Cross-Compare the Features Between Each Other
                analyzeFeatures.featureComparison(pulseFeatures_NonNone, pulseFeatures_NonNone, pulseFeatureLables_NonNone, featureNames, featureNames, 'Pulse', 'Pulse')
        
        if extractChemical and extractPulse:
            # Compare Features with Each Other
            analyzeFeatures = featureAnalysis.featureAnalysis([], [], featureNames, stimulusTimes, dataFolderWithSubjects + "Machine Learning/Compiled Pulse Feature Analysis/")
            analyzeFeatures.featureComparison(pulseFeatures_NonNone, pulseFeatures_NonNone, pulseFeatureLables_NonNone, featureNames, featureNames, 'Pulse', 'Pulse')
        #Compare Stress Scores with the Features
        analyzeFeatures = featureAnalysis.featureAnalysis([], [], featureNames, stimulusTimes, dataFolderWithSubjects + "Machine Learning/Compiled Stress Score Feature Analysis/")
        analyzeFeatures.featureComparisonAgainstONE(signalData, scoreLabels, stressLabels, featureNames, "Stress Scores", 'Stress Scores')

    # ---------------------------------------------------------------------- #
    # ---------------------- Machine Learning Analysis --------------------- #
    print("\nBeginning Machine Learning Section")
    
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # scaler.fit(signalData)
    # signalData_Scaled = scaler.transform(signalData)
    
    testStressScores = True
    if testStressScores:
        signalLabels = scoreLabels
    else:
        signalLabels = stressLabels
        
    # Machine Learning File/Model Paths + Titles
    modelType = "SVR"  # Machine Learning Options: NN, RF, LR, KNN, SVM, RG, EN, SVR
    supportVectorKernel = "rbf"
    modelPath = "./Helper Files/Machine Learning Modules/Models/machineLearningModel_ALL.pkl"
    saveModelFolder = dataFolderWithSubjects + "Machine Learning/" + modelType + "/"
                
    # Get the Machine Learning Module
    performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = len(featureNames), machineLearningClasses = listOfStressors, saveDataFolder = saveModelFolder, supportVectorKernel = supportVectorKernel)
    # Train the Data on the Gestures
    print(performMachineLearning.trainModel(signalData, signalLabels, featureNames, returnScore = True, stratifyBy = stressLabels))
    print(performMachineLearning.predictionModel.scoreModel(signalData, signalLabels))
    
    
    sys.exit()
    
    # modelScores_Single0 = []
    # modelScores_Single1 = []
    # modelScores_Single2 = []
    # modelScores_SingleTotal = []
    # for featureInd in range(len(featureNames)):
    #     featureRow = featureNames[featureInd]
    
    #     signalDataCull = np.reshape(signalData[:,featureInd], (-1,1))
    
    #     performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = 1, machineLearningClasses = listOfStressors, saveDataFolder = saveModelFolder, supportVectorKernel = supportVectorKernel)
        
    #     modelScore = performMachineLearning.scoreClassificationModel(signalDataCull, signalLabels, stratifyBy = stressLabels)
        
    #     # modelScores_Single0.append(modelScore[0])
    #     # modelScores_Single1.append(modelScore[1])
    #     # modelScores_Single2.append(modelScore[2])
    #     # modelScores_SingleTotal.append(modelScore[3])
        
    #     modelScores_SingleTotal.append(modelScore[0])
        
    # # excelProcessing.processMLData().saveFeatureComparison([modelScores_Single0], [], featureNames, saveModelFolder, "Single Feature Accuracy.xlsx", sheetName = "Feature Comparison Cold")
    # # excelProcessing.processMLData().saveFeatureComparison([modelScores_Single1], [], featureNames, saveModelFolder, "Single Feature Accuracy.xlsx", sheetName = "Feature Comparison Excersize")
    # # excelProcessing.processMLData().saveFeatureComparison([modelScores_Single2], [], featureNames, saveModelFolder, "Single Feature Accuracy.xlsx", sheetName = "Feature Comparison VR")
    # excelProcessing.processMLData().saveFeatureComparison([modelScores_SingleTotal], [], featureNames, saveModelFolder, "Single Feature Accuracy.xlsx", sheetName = "Feature Comparison Total")
    
    # modelScores = np.zeros((len(featureNames), len(featureNames)))
    # for featureIndRow in range(len(featureNames)):
    #     print(featureIndRow)
    #     featureRow = featureNames[featureIndRow]
    #     for featureIndCol in range(len(featureNames)):
    #         if featureIndCol < featureIndRow:
    #              modelScores[featureIndRow][featureIndCol] = modelScores[featureIndCol][featureIndRow]
    #              continue
             
    #         featureCol = featureNames[featureIndCol]
    #         signalDataCull = np.dstack((signalData[:,featureIndRow], signalData[:,featureIndCol]))[0]
            
    #         performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = 2, machineLearningClasses = listOfStressors, saveDataFolder = saveModelFolder, supportVectorKernel = supportVectorKernel)
    #         modelScore = performMachineLearning.trainModel(signalDataCull, signalLabels, returnScore = True, stratifyBy = stressLabels)
    #         modelScores[featureIndRow][featureIndCol] = modelScore
    # for featureIndRow in range(len(featureNames)):
    #     featureRow = featureNames[featureIndRow]
    #     for featureIndCol in range(len(featureNames)):
    #         if featureIndCol < featureIndRow:
    #              modelScores[featureIndRow][featureIndCol] = modelScores[featureIndCol][featureIndRow]
    #              continue
    # excelProcessing.processMLData().saveFeatureComparison(modelScores, featureNames, featureNames, saveModelFolder, "Pairwise Feature Accuracy.xlsx", sheetName = "Feature Comparison")
           
    from itertools import combinations
    numFeatures = 5
    modelScores = []
    featureNamesPermute = []
    featureInds = list(combinations(range(1, len(featureNames)), numFeatures))
    len(featureInds)
    for permutationInd in range(len(featureInds)):
        permutation = featureInds[permutationInd]
        
        signalDataCull = signalData[:,permutation[0]]
        for featureInd in permutation[1:]:
            signalDataCull = np.dstack((signalDataCull, signalData[:,featureInd]))
        signalDataCull = signalDataCull[0]
        
        performMachineLearning = machineLearningMain.predictionModelHead(modelType, modelPath, numFeatures = numFeatures, machineLearningClasses = listOfStressors, saveDataFolder = saveModelFolder, supportVectorKernel = supportVectorKernel)
        modelScore = performMachineLearning.trainModel(signalDataCull, signalLabels, returnScore = True, stratifyBy = stressLabels)
        modelScores.append(modelScore)
        
        featureNamesPermuteSTR = ''
        for name in np.array(featureNames)[np.array(permutation)]:
            featureNamesPermuteSTR += name + ' '
        featureNamesPermute.append(featureNamesPermuteSTR)
        
        if permutationInd%1000 == 0:
            print(permutationInd)
        
    excelProcessing.processMLData().saveFeatureComparison(np.dstack((modelScores, featureNamesPermute))[0], [], [], saveModelFolder, "Permutation Feature Accuracy.xlsx", sheetName = "Feature Comparison Permute")
    
    sys.exit()
    
    
    if False:
        newSignalData = []
        bestFeatures = ['systolicUpstrokeAccelMinVel_StressLevel', 'systolicUpSlopeArea_SignalIncrease', 'velDiffConc_Glucose', 'accelDiffMaxConc_Glucose', 'rightDiffAmp_Lactate']
        for feature in bestFeatures:
            featureInd = featureNames.index(feature)
            
            if len(newSignalData) == 0:
                newSignalData = signalData[:,featureInd]
            else:
                newSignalData = np.dstack((newSignalData, signalData[:,featureInd]))
        newSignalData = newSignalData[0]
    
    sys.exit()
    
    stressEquation = ""
    featureCoefficients = performMachineLearning.predictionModel.model.coef_[0]
    for featureNum in range(len(featureCoefficients)):
        featureCoef = featureCoefficients[featureNum]
        featureName = featureNames[featureNum]
        print(featureCoef)
        
        stressEquation += str(np.round(featureCoef, 2)) + "*" + featureName + " + "
    stressEquation[0:-3]
    

        
    # # ---------------------------------------------------------------------- #
    # #                          Train the Model                               #
    # # ---------------------------------------------------------------------- #
    
    # if trainModel:
    #     excelDataML = excelProcessing.processMLData()
    #     # Read in Training Data/Labels
    #     signalData = []; signalLabels = []; FeatureNames = []
    #     for MLFile in os.listdir(trainingDataExcelFolder):
    #         MLFile = trainingDataExcelFolder + MLFile
    #         signalData, signalLabels, FeatureNames = excelDataML.getData(MLFile, signalData = signalData, signalLabels = signalLabels, testSheetNum = 0)
    #     signalData = np.array(signalData); signalLabels = np.array(signalLabels)
    #     # Read in Validation Data/Labels
    #     Validation_Data = []; Validation_Labels = [];
    #     for MLFile in os.listdir(validationDataExcelFolder):
    #         MLFile = validationDataExcelFolder + MLFile
    #         Validation_Data, Validation_Labels, FeatureNames = excelDataML.getData(MLFile, signalData = Validation_Data, signalLabels = Validation_Labels, testSheetNum = 0)
    #     Validation_Data = np.array(Validation_Data); Validation_Labels = np.array(Validation_Labels)
    #     print("\nCollected Signal Data")
        
    #     Validation_Data = Validation_Data[:][:,0:6]
    #     signalData = signalData[:][:,0:6]
    #     FeatureNames = FeatureNames[0:6]
                    
    #     # Train the Data on the Gestures
    #     performMachineLearning.trainModel(signalData, signalLabels, pulseFeatureNames)
    #     # Save Signals and Labels
    #     if False and performMachineLearning.map2D:
    #         saveInputs = excelProcessing.saveExcel()
    #         saveExcelNameMap = "mapedData.xlsx" #"Signal Features with Predicted and True Labels New.xlsx"
    #         saveInputs.saveLabeledPoints(performMachineLearning.map2D, signalLabels,  performMachineLearning.predictionModel.predictData(signalData), saveDataFolder, saveExcelNameMap, sheetName = "Signal Data and Labels")
    #     # Save the Neural Network (The Weights of Each Edge)
    #     if saveModel:
    #         modelPathFolder = os.path.dirname(modelPath)
    #         os.makedirs(modelPathFolder, exist_ok=True)
    #         performMachineLearning.predictionModel.saveModel(modelPath)
    
    

     
     
     

     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     

     
     
     
     
     
     
     
     
     
     
     
     
     
  
   
   
   

