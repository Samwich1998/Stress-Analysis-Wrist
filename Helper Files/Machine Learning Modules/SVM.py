"""
Code Written by Samuel Solomon

SKLearn SVM Guide: https://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
"""

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# Import Basic Modules
import os
import sys
import joblib
import numpy as np
import pandas as pd

# Import Modules for Plotting
import matplotlib
from matplotlib import pyplot
import matplotlib.pyplot as plt

# Import Machine Learning Modules
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance

# Import Python Files
sys.path.append('./Data Aquisition and Analysis/')  # Folder with Machine Learning Files
import createHeatMap as createMap       # Functions for Neural Network
    
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

class SVM:
    def __init__(self, modelPath, modelType = "rbf", polynomialDegree = 3):
        # If Using Polynomial Degree
        self.polynomialDegree = polynomialDegree
        
        # Initialize Model
        if os.path.exists(modelPath):
            # If Model Exists, Load it
            self.loadModel(modelPath)
        else:
            # Else, Create the Model
            self.createModel(modelType)
    
    def saveModel(self, modelPath = "./SVM.sav"):
        joblib.dump(self.model, modelPath)    
    
    def loadModel(self, modelPath):
        with open(modelPath, 'rb') as handle:
            self.model = joblib.load(handle, mmap_mode ='r')
        print("SVM Model Loaded")
            
    
    def createModel(self, modelType = "rbf"):
        if modelType == "linear":
            self.model = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo')
        elif modelType == "rbf":
            self.model = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo')
        elif modelType == "poly":
            self.model = svm.SVC(kernel='poly', degree = self.polynomialDegree, C=1, decision_function_shape='ovo')
        elif modelType == "sigmoid":
            self.model = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo')
        else:
            print("No SVM Model Matches the Requested Type")
            sys.exit()
        print("SVM Model Created")
        
    def trainModel(self, Training_Data, Training_Labels, Testing_Data, Testing_Labels):  
        # Train the Model
        self.clf = self.model.fit(Training_Data, Training_Labels)
        self.scoreModel(Testing_Data, Testing_Labels)
        
        result = permutation_importance(self.clf, Training_Data, Training_Labels, n_repeats=10, random_state=0)
        print(result.importances_mean)
    
    def scoreModel(self, signalData, signalLabels):
        print("Score:", self.model.score(signalData, signalLabels))
    
    def featureImportance(self, headerTitles = []):


        # get importance
        importance = self.model.coef_[0]
        # summarize feature importance
        for i,v in enumerate(importance):
            if headerTitles:
                i = headerTitles[i]
                print('%s Weight: %.5f' % (str(i),v))
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        freq_series = pd.Series(importance)
        ax = freq_series.plot(kind="bar")
        
        # Specify Figure Aesthetics
        ax.set_title("Feature Importance in Model")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Feature Importance")
        
        # Set X-Labels
        if headerTitles:
            ax.set_xticklabels(headerTitles)
            self.add_value_labels(ax)
        # Show Plot
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
    
    def printStatistics(self, Training_Data, Training_Labels, Testing_Data, Testing_Labels):
        # Defined Kernel Types
        linearKernel = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(Training_Data, Training_Labels)
        polynomialKernel = svm.SVC(kernel='poly', degree = self.polynomialDegree, C=1, decision_function_shape='ovo').fit(Training_Data, Training_Labels)
        radialKernel = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(Training_Data, Training_Labels)
        sigmoidKernel = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(Training_Data, Training_Labels)
        # Combine Kernels into 1-List
        kernels = [linearKernel, polynomialKernel, radialKernel, sigmoidKernel]
        kernelTitles = ["Linear", "Polynomial", "Radial Basis Function", "Sigmoid"]
        
        # Print Accuracy
        for kernelInd, kernel in enumerate(kernels):
            accuracy = kernel.score(Testing_Data, Testing_Labels)
            print(kernelTitles[kernelInd], "Kernel Accuracy:", accuracy)
        print("")
        
        # Print Confusion Matrix
        for kernelInd, kernel in enumerate(kernels):
            predictedData = kernel.predict(Testing_Data)
            confusionMatrix = confusion_matrix(Testing_Labels, predictedData)
            print(kernelTitles[kernelInd], "Kernel Confusion Matrix:", confusionMatrix)


    def add_value_labels(self, ax, spacing=5):
        """Add labels to the end of each bar in a bar chart.
    
        Arguments:
            ax (matplotlib.axes.Axes): The matplotlib object containing the axes
                of the plot to annotate.
            spacing (int): The distance between the labels and the bars.
        """
    
        # For each bar: Place a label
        for rect in ax.patches:
            # Get X and Y placement of label from rect.
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2
    
            # Number of points between bar and label. Change to your liking.
            space = spacing
            # Vertical alignment for positive values
            va = 'bottom'
    
            # If value of bar is negative: Place label below bar
            if y_value < 0:
                # Invert space to place label below
                space *= -1
                # Vertically align label at top
                va = 'top'
    
            # Use Y value as label and format number with one decimal place
            label = "{:.1f}".format(y_value)
    
            # Create annotation
            ax.annotate(
                label,                      # Use `label` as label
                (x_value, y_value),         # Place label at end of the bar
                xytext=(0, space),          # Vertically shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                ha='center',                # Horizontally center label
                va=va)                      # Vertically align label differently for
                                            # positive and negative values.


