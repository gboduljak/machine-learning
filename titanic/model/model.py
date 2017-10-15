import pandas as pd
from pandas import DataFrame

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier



trainDataset = pd.read_csv('../input/train.csv')
testDataset  = pd.read_csv('../input/test.csv')

#Preprocess and clean train dataset

def preprocess_and_clean(dataset):     
    dataset.loc[(dataset.Embarked.isnull(), 'Embarked')] = 'N'
        
    dataset['Gender']       = dataset['Sex'].map({ 'female':0, 'male':1}).astype(int)
    dataset['EmbarkedCode'] = dataset['Embarked'].map({ 'S':0, 'C':1, 'Q' : 2, 'N' : 3}).astype(int)
    
    dataset.loc[(dataset.Age.isnull()),'Age'] = dataset.Age.mean()
    dataset.loc[(dataset.Fare.isnull()),'Fare'] = dataset.Fare.mean()
    dataset.loc[(3,'EmbarkedCode')] = dataset.EmbarkedCode.mean()
    dataset.drop(['Name','Ticket','Cabin', 'Sex', 'Embarked'], 1, inplace=True)
    
    
    fareScaler = StandardScaler()
    ageScaler = StandardScaler()
    
    fareScaler.fit(dataset['Fare'].astype(float).reshape(-1, 1))
    ageScaler.fit(dataset['Age'].astype(float).reshape(-1, 1))
    
    dataset['Fare'] = fareScaler.transform(dataset['Fare'].astype(float).reshape(-1, 1))
    dataset['Age'] = ageScaler.transform(dataset['Age'].astype(float).reshape(-1, 1))
    
    
preprocess_and_clean(trainDataset)
preprocess_and_clean(testDataset)


#Split into training dataset into train and test
X = trainDataset.loc[:, trainDataset.columns != 'Survived']
y = trainDataset['Survived'].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)

#Train Random Forest model
randomForestModels = {
    'n_estimators' : [10, 50, 100, 200],
    'criterion' : ['gini','entropy'],
    'random_state' : [0]
}

randomForestModel = RandomForestClassifier()

randomForestGridSearch = GridSearchCV(estimator = randomForestModel, param_grid = randomForestModels)
randomForestGridSearch.fit(X_train, y_train)
print(randomForestGridSearch.best_score_)


#Train SVM  model
svmModels = [
    {
      'C' : [0.5, 1, 10, 100, 1000, 10000],
      'kernel' : ['rbf'],
      'gamma' : [0.00001, 0.0001, 0.001, 0.01]
    }
]
svmModel = SVC()

svmGridSearch = GridSearchCV(estimator = svmModel, param_grid = svmModels)
svmGridSearch.fit(X_train, y_train).score(X_test, y_test)

print(svmGridSearch.best_score_)

#TrainNN model
possibleNeuralNetworkParameters = [{  
        'hidden_layer_sizes' : [(200), (500), (200, 200), (500,500), (500, 500, 500)],
        'activation' :['relu', 'logistic'],
        'alpha' : [0.00001, 0.0001, 0.001, 0.01],
        'learning_rate' : ['constant', 'adaptive']
}]

neuralNetworkModel = MLPClassifier()

neuralNetworkSearch = GridSearchCV(estimator = neuralNetworkModel, param_grid = possibleNeuralNetworkParameters)
neuralNetworkSearch.fit(X_train, y_train).score(X_test, y_test)

#Save Random forest test
predictions = {
   'PassengerId' : testDataset['PassengerId'].values,
   'Survived' : randomForestGridSearch.best_estimator_.predict(testDataset.loc[:, testDataset.columns != 'Survived'])
}
submissionFrame = DataFrame(data = predictions)
submissionFrame.to_csv('submission.csv', index = False)