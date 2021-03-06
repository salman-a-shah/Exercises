"""
TITLE  : Random forest applied to the wine dataset with pca
AUTHOR : Salman Shah
DATE   : Wed May 23 19:19:20 2019

Implemented for the sake of experimentation 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_wine

''' Setting up a dataframe to have a reference while coding '''
file = load_wine()
column_names = file.feature_names
target = file.target
data = file.data
df1 = pd.DataFrame(data, columns=column_names)
df2 = pd.DataFrame(target, columns=['target'])
dataset = pd.concat([df1,df2], axis=1)  # use this for reference

# Feature scaling and applying PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
sc = StandardScaler()
pca = PCA(n_components=2)
X = pca.fit_transform(sc.fit_transform(file.data))
y_original = target
y_one_hot = pd.get_dummies(df2['target'], drop_first=True) # one hot encoded y

# Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y_one_hot, test_size=0.2, random_state=0)

# Building a Random forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100,
                                    criterion="gini",
                                    max_depth=3,
                                    random_state=0)

# Fitting and predicting
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# undoing some of the one-hot-encoding
y_pred_flat = [0 if (sum(y_pred[i])==0) else np.argmax(y_pred[i]) + 1 for i in range(len(y_pred))]
y_test_flat = [0 if (sum(y_test.values[i])==0) else np.argmax(y_test.values[i]) + 1 for i in range(len(y_test.values))]
y_train_flat = [0 if (sum(y_train.values[i])==0) else np.argmax(y_train.values[i]) + 1 for i in range(len(y_train.values))]

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_flat, y_pred_flat)

''' Plotting ''' 
# Defining some parameters and a custom colormap
plot_colors = 'bgr'
plot_step = 0.02
class_names = '123'
blue = np.array([0/256,150/256,255/256,1])
red = np.array([255/256,85/256,85/256,1])
green = np.array([85/256,230/256,85/256,1])
custom_cmap = ListedColormap([blue, green, red])

# building a grid of predictions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))
input_grid = np.c_[xx.ravel(), yy.ravel()]
Z = classifier.predict(input_grid)
Z = [0 if (sum(Z[i])==0) else np.argmax(Z[i]) + 1 for i in range(Z.shape[0])]
Z = np.array(Z).reshape(xx.shape)

# Plotting the boundary regions
fig = plt.figure(figsize=(10,5))
ax = plt.subplot(121)
cs = plt.contourf(xx, yy, Z, cmap=custom_cmap)

# plotting the training set
for i, n, c in zip(range(3), class_names, plot_colors):
    train_idx = np.where(np.array(y_train_flat)==i)
    test_idx = np.where(np.array(y_test_flat)==i)
    ax.scatter(X_train[train_idx, 0], X_train[train_idx, 1],
                      c=c, s=40,
                      edgecolor='k',
                      marker='o',
                      label="Train C%s" % n)
    ax.scatter(X_test[test_idx, 0], X_test[test_idx, 1],
                      c=c, s=40, 
                      edgecolor='k',
                      marker='v',
                      label="Test C%s" % n)
ax.legend(loc='upper right', fontsize='x-small')
plt.title("Random Forest on Wine Dataset with PCA")
ax.axis('tight')
ax.tick_params(axis='both', 
                labelbottom=False, 
                labelleft=False,
                bottom=False,
                left=False)

# plotting the confusion matrix
import seaborn as sn
df_cm = pd.DataFrame(cm, 
                     columns=['Predicted C1', 'Predicted C2', 'Predicted C3'], 
                     index=['Actual C1', 'Actual C2', 'Actual C3'])
ax2 = plt.subplot(122)
sn.heatmap(df_cm, annot=True)
plt.title("Confusion Matrix on Test Set")
plt.show()