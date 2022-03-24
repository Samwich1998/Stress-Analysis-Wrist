#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 18:47:49 2021

@author: samuelsolomon
"""
# Basic Modules
import os
import numpy as np
from scipy import stats
from copy import copy, deepcopy
# Modules for Plotting
import seaborn as sns
import matplotlib.pyplot as plt


class featureAnalysis:
    
    def __init__(self, timePoints, featureList, featureNames, stimulusTime, saveDataFolder):
        # Store Extracted Features
        self.featureNames = featureNames            # Store Feature Names
        self.timePoints = np.array(timePoints)          # Store Feature Times
        self.featureList = np.array(featureList)    # Store Features
        
        self.stimulusTime = list(stimulusTime)
        
        # Save Information
        self.saveDataFolder = saveDataFolder
        
        self.colorList = ['ko', 'r-o', 'bo', 'go', 'mo']
    
    def singleFeatureAnalysis(self, averageIntervalList = [0.001, 30]):
        # Create Directory to Save the Figures
        saveDataFolder = self.saveDataFolder + "singleFeatureAnalysis/"
        os.makedirs(saveDataFolder, exist_ok=True)
        
        # Loop Through Each Feature
        for featureInd in range(len(self.featureNames)):
            fig = plt.figure()
            
            # Extract One Feature from the List
            allFeatures = self.featureList[:,featureInd]
            # Take Different Averaging Methods
            for ind, averageTogether in enumerate(averageIntervalList):
                features = []
                
                # Average the Feature Together at Each Point
                for pointInd in range(len(allFeatures)):
                    # Get the Interval of Features to Average
                    featureInterval = allFeatures[self.timePoints > self.timePoints[pointInd] - averageTogether]
                    timeMask = self.timePoints[self.timePoints > self.timePoints[pointInd] - averageTogether]
                    featureInterval = featureInterval[timeMask <= self.timePoints[pointInd]]
                    
                    # Take the Trimmed Average
                    feature = stats.trim_mean(featureInterval, 0.3)
                    features.append(feature)
                
                # Plot the Feature
                plt.plot(self.timePoints, features, self.colorList[ind], markersize=5)
            
            # Specify the Location of the Stimulus
            plt.vlines(self.stimulusTime, min(features), max(features), 'g', linewidth = 2, zorder=len(averageIntervalList) + 1)

            # Add Figure Labels
            plt.xlabel("Time (Seconds)")
            plt.ylabel(self.featureNames[featureInd])
            plt.title(self.featureNames[featureInd] + " Analysis")
            # Add Figure Legened
            plt.legend([str(averageTime) + " Sec" for averageTime in averageIntervalList])
            # Save the Figure
            fig.savefig(saveDataFolder + self.featureNames[featureInd] + ".png", dpi=300, bbox_inches='tight')
            
            # Clear the Figure        
            fig.clear()
            plt.close(fig)
            plt.cla()
            plt.clf()
            
    
    def correlationMatrix(self, featureList, featureNames):
        # Create Directory to Save the Figures
        saveDataFolder = self.saveDataFolder + "correlationMatrix/"
        os.makedirs(saveDataFolder, exist_ok=True)
        
        # Perform Deepcopy to Not Edit Features
        signalData = np.array(deepcopy(featureList)); signalLabels = np.array(deepcopy(featureNames))
        
        # Standardize the Feature
        for i in range(len(signalData[0])):
             signalData[:,i] = (signalData[:,i] - np.mean(signalData[:,i]))/np.std(signalData[:,i], ddof=1)
        
        matrix = np.array(np.corrcoef(signalData.T)); 
        sns.set_theme(); ax = sns.heatmap(matrix, cmap='icefire', xticklabels=signalLabels, yticklabels=signalLabels)
        # Save the Figure
        sns.set(rc={'figure.figsize':(50,35)})
        fig = ax.get_figure(); fig.savefig(saveDataFolder + "correlationMatrixFull.png", dpi=300)
        plt.show()
        
        # Cluster the Similar Features
        signalLabelsX = np.array(deepcopy(signalLabels))
        signalLabelsY = np.array(deepcopy(signalLabels))
        for i in range(1,len(matrix)):
            signalLabelsX = signalLabelsX[matrix[:,i].argsort()]
            matrix = matrix[matrix[:,i].argsort()]
        for i in range(1,len(matrix[0])):
            signalLabelsY = signalLabelsY[matrix[i].argsort()]
            matrix = matrix [ :, matrix[i].argsort()]
        # Plot the New Cluster
        sns.set_theme(); ax = sns.heatmap(matrix, cmap='icefire', xticklabels=signalLabelsX, yticklabels=signalLabelsY)
        # Save the Figure
        sns.set(rc={'figure.figsize':(50,35)})
        fig = ax.get_figure(); fig.savefig(saveDataFolder + "correlationMatrixSorted.png", dpi=300)
        plt.show()

        # Remove Small Correlations
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if abs(matrix[i][j]) < 0.96:
                    matrix[i][j] = 0
        # Plot the New Correlations
        sns.set_theme(); ax = sns.heatmap(matrix, cmap='icefire', xticklabels=signalLabelsX, yticklabels=signalLabelsY)
        # Save the Figure
        sns.set(rc={'figure.figsize':(50,35)})
        fig = ax.get_figure(); fig.savefig(saveDataFolder + "correlationMatrixSortedCull.png", dpi=300)            
        plt.show()
    
    def featureComparison(self, featureList1, featureList2, featureLabels, featureNames1, featureNames2, xChemical, yChemical):
        # Create Directory to Save the Figures
        saveDataFolder = self.saveDataFolder + "chemicalFeatureComparison/" + xChemical + " vs " + yChemical + "/"
        os.makedirs(saveDataFolder, exist_ok=True)
                
        featureList1 = np.array(featureList1)
        featureList2 = np.array(featureList2)
        
        colorList = ['ko', 'ro', 'bo']
        labelList = ['Cold', 'Exercise', 'VR']

        for featureInd1 in range(len(featureList1[0])):
            
            features1 = featureList1[:, featureInd1]
            
            for featureInd2 in range(len(featureList2[0])):
                features2 = featureList2[:, featureInd2]
                
                fig = plt.figure()
                addedLegend = []
                for ind in range(len(featureLabels)):
                    labelInd = featureLabels[ind]
                    if labelList[labelInd] not in addedLegend:
                        addedLegend.append(labelList[labelInd])
                        plt.plot(features1[ind], features2[ind], colorList[labelInd], label=labelList[labelInd])
                    else:
                        plt.plot(features1[ind], features2[ind], colorList[labelInd])
                
                plt.xlabel(xChemical + ": " + featureNames1[featureInd1])
                plt.ylabel(yChemical + ": " + featureNames2[featureInd2])
                plt.title("Feature Comparison")
                plt.legend()
                # Save the Figure
                saveDataFolderFeature1 = saveDataFolder + featureNames1[featureInd1] + "/"
                os.makedirs(saveDataFolderFeature1, exist_ok=True)
                fig.savefig(saveDataFolderFeature1 + featureNames1[featureInd1] + "_" + featureNames2[featureInd2] + ".png", dpi=300, bbox_inches='tight')
            
                # Clear the Figure        
                fig.clear()
                plt.close(fig)
                plt.cla()
                plt.clf()    
                
    def singleFeatureComparison(self, featureListFull, featureLabelFull, chemicalOrder, featureNames):
        # Create Directory to Save the Figures
        saveDataFolder = self.saveDataFolder + "singleChemicalFeatureComparison/"
        os.makedirs(saveDataFolder, exist_ok=True)
        
        colorList = ['ko', 'ro', 'bo']
        labelList = ['Cold', 'Exercise', 'VR']
        #labelList = ['Cold', 'Exercise', 'VR']
        for chemicalInd in range(len(chemicalOrder)):
            chemicalName = chemicalOrder[chemicalInd]
            featureList = featureListFull[chemicalInd]
            featureLabels = featureLabelFull[chemicalInd]
            
            saveDataFolderChemical = saveDataFolder + chemicalName + "/"
            os.makedirs(saveDataFolderChemical, exist_ok=True)
            
            for featureInd in range(len(featureList[0])):
                
                features = featureList[:, featureInd]
                
                fig = plt.figure()
                addedLegend = []
                for ind in range(len(featureLabels)):
                    labelInd = featureLabels[ind]
                    if labelInd not in addedLegend:
                        addedLegend.append(labelInd)
                        plt.plot(features[ind], [0], colorList[labelInd], label=labelList[labelInd])
                    else:
                        plt.plot(features[ind], [0], colorList[labelInd])
                
                plt.xlabel(chemicalName + ": " + featureNames[featureInd])
                plt.ylabel("Constant")
                plt.title("Feature Comparison")
                plt.legend()
                # Save the Figure
                fig.savefig(saveDataFolderChemical + featureNames[featureInd] + ".png", dpi=300, bbox_inches='tight')
            
                # Clear the Figure        
                fig.clear()
                plt.close(fig)
                plt.cla()
                plt.clf()
            
            
            
            
            
            
            
            
            
            
            
            
            