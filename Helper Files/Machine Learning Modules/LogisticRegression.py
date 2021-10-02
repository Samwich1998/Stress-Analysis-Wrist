"""
https://scikit-learn.org/stable/auto_examples/neighbors/plot_nca_classification.html#sphx-glr-auto-examples-neighbors-plot-nca-classification-py


"""

# Basic Modules
import os
import sys
import numpy as np
# Modules for Plotting
import matplotlib
from matplotlib import pyplot
import matplotlib.pyplot as plt
# Modules for Machine Learning
import joblib
from sklearn.linear_model import LogisticRegression

# Import Files to Create HeatMap
sys.path.append('./Data Aquisition and Analysis/')  # Folder with Machine Learning Files
import createHeatMap as createMap       # Functions for Neural Network

class logisticRegression:
    
    def __init__(self, modelPath):
        
        # Initialize Model
        if os.path.exists(modelPath):
            # If Model Exists, Load it
            self.loadModel(modelPath)
        else:
            # Else, Create the Model
            self.createModel()
    
    def saveModel(self, modelPath = "./LR.sav"):
        joblib.dump(self.model, 'scoreregression.pkl')    
    
    def loadModel(self, modelPath):
        with open(modelPath, 'rb') as handle:
            self.model = joblib.load(handle, mmap_mode ='r')
        print("LR Model Loaded")
        
    def createModel(self):
        self.model = LogisticRegression()
        print("LR Model Created")
        
    def trainModel(self, Training_Data, Training_Labels, Testing_Data, Testing_Labels):  
        # Train the Model
        self.model.fit(Training_Data, Training_Labels)
        self.scoreModel(Testing_Data, Testing_Labels)
    
    def scoreModel(self, signalData, signalLabels):
        print("Score:", self.model.score(signalData, signalLabels))
    
    def featureImportance(self):
        # get importance
        importance = self.model.coef_[0]
        # summarize feature importance
        for i,v in enumerate(importance):
        	print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
        pyplot.bar([x for x in range(len(importance))], importance)
        pyplot.show()
    
    def predictData(self, New_Data):
        # Predict Label based on new Data
        return self.model.predict(New_Data)
        
    def accuracyDistributionPlot(self, signalData, trueLabels, predicatedLabels, signalLabelsTitles, saveFolder = "../Output Data/Machine Learning Results/", name = "Accuracy Distribution"):
        
        # Calculate the Accuracy Matrix
        accMat = np.zeros((len(signalLabelsTitles), len(signalLabelsTitles)))
        for ind, channelFeatures in enumerate(signalData):
            # Sum(Row) = # of Gestures Made with that Label
            # Each Column in a Row = The Number of Times that Gesture Was Predicted as Column Label #
            accMat[trueLabels[ind]][predicatedLabels[ind]] += 1
        
        # Scale Each Row to 100
        for label in range(len(signalLabelsTitles)):
            accMat[label] = 100*accMat[label]/np.sum(accMat[label])
        
        # Make plot
        fig, ax = plt.subplots()
        fig.set_size_inches(8,8)
        
        # Make heatmap on plot
        im, cbar = createMap.heatmap(accMat, signalLabelsTitles, signalLabelsTitles, ax=ax,
                           cmap="copper", cbarlabel="Label Accuracy (%)")
        createMap.annotate_heatmap(im, accMat, valfmt="{x:.2f}",)
        
        # Style the Fonts
        font = {'family' : 'verdana',
                'weight' : 'bold',
                'size'   : 9}
        matplotlib.rc('font', **font)
                
        # Format, save, and show
        fig.tight_layout()
        plt.savefig(saveFolder + name + ".png", dpi=300, bbox_inches='tight')
        plt.show()

    
    def plotStatistics(self, Training_Data, Testing_Data, Training_Labels, Testing_Labels):                
        # we create an instance of Linear Regression Model
        model = LogisticRegression()
        model.fit(Training_Data, Training_Labels)
        print("Score:", model.score(Testing_Data, Testing_Labels))
        
                
            
            