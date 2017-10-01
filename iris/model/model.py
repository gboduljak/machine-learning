import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from pandas.plotting import radviz
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

dataset = pd.read_csv('../input/Iris.csv')
dataset.head()
dataset["Species"].value_counts()

sns.set(style="white", color_codes=True)
sns.FacetGrid(dataset, hue="Species", size=5).map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()
plt.show()
sns.FacetGrid(dataset, hue="Species", size=5).map(plt.scatter, "PetalLengthCm", "PetalWidthCm").add_legend()
plt.show()
radviz(dataset.drop("Id", axis=1), "Species")
plt.show()

classes = {
   'Iris-setosa' : 0,
   'Iris-virginica' : 1, 
   'Iris-versicolor': 2
}

classIds = {
    0 : 'Iris-setosa',
    1 : 'Iris-virginica', 
    2 : 'Iris-versicolor'
}

X = dataset.iloc[:, 1 : 5]
y = dataset.iloc[:, 5]
y = list(map(lambda classAsString : classes[classAsString], y))

# Preprocess Dataset
standardScaler = StandardScaler()
X = standardScaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Train KNN model with cross validation 
possibleKNeighbors    = list(range(1, 50,2))
crossValidationScores = []

for k in possibleKNeighbors:
    knnModel            = KNeighborsClassifier(n_neighbors=k)
    knnModelTestScores  = cross_val_score(knnModel, X_train, y_train, cv = 10, scoring='accuracy')
    crossValidationScores.append(knnModelTestScores.mean())

plt.title('KNN Cross Validation Scores')
plt.xlabel('Number of neighbors K')
plt.ylabel('Accuracy')
plt.plot(possibleKNeighbors, crossValidationScores)
plt.show()


knnModel     = KNeighborsClassifier()
gridSearchCV = GridSearchCV(estimator = knnModel, param_grid = dict(n_neighbors = possibleKNeighbors))
gridSearchCV.fit(X_train, y_train).score(X_test, y_test)
bestNeighborsK = gridSearchCV.best_estimator_.n_neighbors;
knnModel       = KNeighborsClassifier(n_neighbors=bestNeighborsK)

# Train SVC model with cross validation
possibleSVCParameters = [{ 
        'C': np.logspace(1, 10, 10), 
        'gamma': [0.01, 0.001, 0.0001, 0.00001], 
        'kernel': ['rbf']
}]

svcModel = SVC()

gridSearchCV = GridSearchCV(estimator = svcModel, param_grid = possibleSVCParameters)
gridSearchCV.fit(X_train, y_train).score(X_test, y_test)

bestC       = gridSearchCV.best_estimator_.C;
bestGamma   = gridSearchCV.best_estimator_.gamma;

svcModel = SVC(C = bestC, gamma = bestGamma)
svcModel.fit(X_train, y_train)

# Evaluate and display model

pca = PCA(n_components=2)
pca.fit(X_train)

X_display = pca.transform(X_train)
y_display = y_train
    
svcModel = SVC(C = bestC, gamma = bestGamma)
svcModel.fit(X_display, y_display)
knnModel.fit(X_display, y_display)

X1, X2 = np.meshgrid(
        np.arange(start = X_display[:, 0].min() - 1, stop = X_display[:, 0].max() + 1, step = 0.01),
        np.arange(start = X_display[:, 1].min() - 1, stop = X_display[:, 1].max() + 1, step = 0.01)
)

plt.contourf(X1, 
             X2, 
             svcModel.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
             alpha = 0.5, 
             cmap = ListedColormap(('red', 'green', 'blue'))
)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(y_display)):
    plt.scatter(X_display[y_display == j, 0], 
                X_display[y_display == j, 1], 
                c = ListedColormap(('red', 'green', 'blue'))(i), 
                label = classIds[j]
    )

plt.title('SVM on training set')  
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()


X1, X2 = np.meshgrid(
        np.arange(start = X_display[:, 0].min() - 1, stop = X_display[:, 0].max() + 1, step = 0.01),
        np.arange(start = X_display[:, 1].min() - 1, stop = X_display[:, 1].max() + 1, step = 0.01)
)

plt.contourf(X1, 
             X2, 
             knnModel.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
             alpha = 0.5, 
             cmap = ListedColormap(('red', 'green', 'blue'))
)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(y_display)):
    plt.scatter(X_display[y_display == j, 0], 
                X_display[y_display == j, 1], 
                c = ListedColormap(('red', 'green', 'blue'))(i), 
                label = classIds[j]
    )
plt.title('KNN on training set')  
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()