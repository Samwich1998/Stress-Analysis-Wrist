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
from sklearn import neighbors
from sklearn.inspection import permutation_importance
# Modules for Dimension Scaling
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler

# Import Files to Create HeatMap
sys.path.append('./Data Aquisition and Analysis/')  # Folder with Machine Learning Files
import createHeatMap as createMap       # Functions for Neural Network

class KNN:
    def __init__(self, modelPath, numClasses, weight = 'distance'):
        self.numNeighbors = numClasses
        self.weight = weight  # Should be 'uniform', 'distance', or a callable function
        self.model = None
        
        # Initialize Model
        if os.path.exists(modelPath):
            # If Model Exists, Load it
            self.loadModel(modelPath)
        else:
            # Else, Create the Model
            self.createModel(weight)
    
    def loadModel(self, modelPath):
        with open(modelPath, 'rb') as handle:
            self.model = joblib.load(handle, mmap_mode ='r')
        print("KNN Model Loaded")
    
    def createModel(self, weight):
        self.model = neighbors.KNeighborsClassifier(n_neighbors = self.numNeighbors, weights = weight, algorithm = 'auto', 
                        leaf_size = 30, p = 1, metric = 'minkowski', metric_params = None, n_jobs = None)
        print("KNN Model Created")
        
    def saveModel(self, modelPath = "./KNN.pkl"):
        with open(modelPath, 'wb') as handle:
            joblib.dump(self.model, handle)
    
    def trainModel(self, Training_Data, Training_Labels, newData = [], newLabels = [], scoreType = "Score:"):
        # Train the Model
        self.model.fit(Training_Data, Training_Labels)
        # Score the Model
        if len(newData) > 0:
            self.scoreModel(newData, newLabels, scoreType)
    
    def scoreModel(self, signalData, signalLabels, scoreType = "Score:"):
        print(scoreType, self.model.score(signalData, signalLabels))
    
    def predictData(self, New_Data):
        # Predict Label based on new Data
        return self.model.predict(New_Data)
    
    def mapTo2DPlot(self, signalData, signalLabels, saveFolder = "./Output Data/Machine Learning Results/", name = "Channel Map", numClasses = 2):
        # Plot and Save
        fig = plt.figure()
        fig.set_size_inches(15,12)

        # Initialize MDS 2D Mapping
        scaler = MinMaxScaler()
        mds = MDS(n_components = numClasses, random_state = 0, n_init = len(signalData[0]))
        # Map the Data into 2D
        X_scaled = scaler.fit_transform(signalData, signalLabels)
        X_2d = mds.fit_transform(X_scaled)
               
        # Plot the Data
        figMap = plt.scatter(X_2d[:,0], X_2d[:,1], c = signalLabels, cmap = plt.cm.get_cmap('cubehelix', numClasses), s = 130, marker='.', edgecolors='k')        
        
        # Figure Aesthetics
        fig.colorbar(figMap, ticks=range(numClasses), label='digit value')
        figMap.set_clim(-0.5, 5.5)
        plt.title('Channel Feature Map');
        #plt.xlabel("Recombinant Axis 1")
        #plt.ylabel("Recombinant Axis 2")
        #fig.tight_layout()
        fig.savefig(saveFolder + name + ".png", dpi=300, bbox_inches='tight')
        plt.show() # Must be the Last Line
        
        return X_2d
    
    def accuracyDistributionPlot(self, signalData, trueLabels, predicatedLabels, signalLabelsTitles, saveFolder = "./Output Data/Machine Learning Results/", name = "Accuracy Distribution"):
        
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
    
    def permuteFeature(self, signalData, signalLabels, featureIndex):
        """ return the score of model when the feature is permuted """
        signalDataPermuted = signalData.copy()
        # Randomize the Feature's Columns
        signalDataPermuted[:,featureIndex] = np.random.permutation(signalData[:,featureIndex])
        # ReScore the Model with the Random Feature
        permuted_score = self.model.score(signalDataPermuted, signalLabels)
        return permuted_score

    
    def get_feature_importance(self, signalData, signalLabels, featureIndex):
        """ compare the score when the feature is permuted """
        
        
        baseScore = self.model.score(signalData, signalLabels)
        permutedScore = self.permuteFeature(signalData, signalLabels, featureIndex)
    
        # feature importance is the difference between the two scores
        featureImportance = baseScore - permutedScore
        return featureImportance

    def plotImportance(self, perm_importance_result, headerTitles):
        """ bar plot the feature importance """
    
        fig, ax = plt.subplots()
    
        indices = perm_importance_result['importances_mean'].argsort()
        plt.barh(range(len(indices)),
                 perm_importance_result['importances_mean'][indices],
                 xerr=perm_importance_result['importances_std'][indices])
    
        ax.set_yticks(range(len(indices)))
        if headerTitles:
            _ = ax.set_yticklabels(np.array(headerTitles)[indices])
    
    def featureImportance(self, signalData, signalLabels, headerTitles = [], numTrials = 30):
        """
        Randomly Permute a Feature's Column and Return the Average Deviation in the Score: |oldScore - newScore|
        NOTE: ONLY Compare Feature on the Same Scale: Time and Distance CANNOT be Compared
        """
        importanceResults = permutation_importance(self.model, signalData, signalLabels, n_repeats=numTrials)
        self.plotImportance(importanceResults, headerTitles)
        
        
        
