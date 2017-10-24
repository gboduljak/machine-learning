import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from pandas import DataFrame

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


trainDataset = pd.read_csv('../input/train.csv')
testDataset  = pd.read_csv('../input/test.csv')

#Preprocess and clean train dataset

def extract_social_category_from_name_title(dataset):
    
    dataset['SocialCategory'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['SocialCategory'] = dataset['SocialCategory'].replace(['Lady', 'Countess', 'Dona'],'Royalty')
    dataset['SocialCategory'] = dataset['SocialCategory'].replace(['Mme'], 'Mrs')
    dataset['SocialCategory'] = dataset['SocialCategory'].replace(['Mlle','Ms'], 'Miss')
    dataset['SocialCategory'] = dataset['SocialCategory'].replace(['Capt', 'Col', 'Major','Rev'], 'Officer')
    dataset['SocialCategory'] = dataset['SocialCategory'].replace(['Jonkheer', 'Don','Sir'], 'Royalty')
    
    dataset.loc[(dataset.Sex == 'male')   & (dataset.SocialCategory == 'Dr'),'SocialCategory'] = 'Mr'
    dataset.loc[(dataset.Sex == 'female') & (dataset.SocialCategory == 'Dr'),'SocialCategory'] = 'Mrs'
     
    titleCategories = { 
        "Mr"     : 1, 
        "Miss"   : 2, 
        "Mrs"    : 3, 
        "Master" : 4, 
        "Royalty": 5, 
        "Officer": 6
    }
    
    dataset['SocialCategory'] = dataset['SocialCategory'].map(titleCategories)
    dataset['SocialCategory'] = dataset['SocialCategory'].fillna(0)
    

def produce_new_features(dataset):
    dataset['FamilySize']   = dataset['SibSp'] + dataset['Parch'];
    dataset['IsChild'] = False
    dataset.loc[(dataset.Age <= 16), 'IsChild'] = True
    
    dataset['IsAlone'] = 0
    dataset.loc[(dataset['FamilySize'] == 1), 'IsAlone'] = 1
    dataset.loc[(dataset['FamilySize'] > 4),  'IsAlone'] = 2
    
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

    dataset['AgeCategory'] = 5

    dataset.loc[ dataset['Age'] <= 16, 'AgeCategory'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'AgeCategory'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'AgeCategory'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'AgeCategory'] = 3
    dataset.loc[ dataset['Age'] > 64, 'AgeCategory'] = 4
    
    
    dataset['Fare'].fillna(dataset['Fare'].dropna().median(), inplace=True)

    dataset.loc[ dataset['Fare'] <= 7.91, 'FareBand'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'FareBand'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'FareBand']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'FareBand'] = 3
    dataset['FareBand'] = dataset['FareBand'].astype(int)



def preprocess_and_clean(dataset, isTestDataset = False):     
    dataset['Embarked'] = dataset['Embarked'].fillna(dataset.Embarked.dropna().mode()[0])
    dataset['Fare'].fillna(dataset['Fare'].dropna().median(), inplace=True)

    dataset['Age'].fillna(dataset.Age.median())
    
    dataset['Gender']       = dataset['Sex'].map({ 'female':0, 'male':1}).astype(int)
    dataset['EmbarkedCode'] = dataset['Embarked'].map({ 'S':0, 'C':1, 'Q' : 2, 'N' : 3}).astype(int)

    
    dataset.loc[(dataset.Age.isnull()),'Age'] = dataset.Age.median()

    
    produce_new_features(dataset)
    extract_social_category_from_name_title(dataset)
    
    #dataset.drop(['Name','Ticket','Cabin', 'Sex', 'Embarked', 'SibSp', 'Parch', 'FamilySize', 'PassengerId'], 1, inplace=True)
    #0.83426, 0.8118
    
    #dataset.drop(['Name','Ticket','Cabin', 'Sex', 'Embarked', 'SibSp', 'Parch', 'FamilySize', 'PassengerId', 'Age'], 1, inplace=True)
    #0.8377623, 0.81678
    #dataset.drop(['Name','Ticket','Cabin', 'Sex', 'Embarked', 'SibSp', 'Parch', 'FamilySize', 'PassengerId',  'Fare'], 1, inplace=True)
    #0.82346, 0.82657
    
    #dataset.drop(['Name','Ticket','Cabin', 'Sex', 'Embarked', 'SibSp', 'Parch', 'FamilySize', 'PassengerId', 'Age', 'Fare'], 1, inplace=True)
    #0.83636, 0.81958
    
    if isTestDataset == False:
        dataset.drop(['PassengerId','Name','Ticket','Cabin', 'Sex', 'SibSp', 'Parch',  'Embarked', 'FamilySize'], 1, inplace=True)
    else:
        dataset.drop(['PassengerId','Name','Ticket','Cabin', 'Sex', 'SibSp', 'Parch', 'Embarked', 'FamilySize'], 1, inplace=True)
 
def predict_submission_with_model(model, modelName):
    predictions = {
        'PassengerId' : testDataset['PassengerId'].values,
        'Survived' : model.predict(testDataset)
    }
    submissionFrame = DataFrame(data = predictions)
    submissionFrame.to_csv(modelName + '-submission.csv', index = False)

preprocess_and_clean(trainDataset)
preprocess_and_clean(testDataset, isTestDataset = True)

#Visualize 
plt, grid = plt.subplots(nrows = 2, ncols = 2 ,figsize=(15,10))
sns.barplot(x = 'Gender', y='Survived', data=trainDataset, ax=grid[0,0])
sns.barplot(x = 'IsAlone', y='Survived', data=trainDataset, ax=grid[0,1])
sns.barplot(x = 'EmbarkedCode', y='Survived', data=trainDataset, ax=grid[1,0])
sns.barplot(x = 'SocialCategory', y='Survived', data=trainDataset, ax=grid[1,1])

trainDataset[['Gender', 'Survived']].groupby(['Gender'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#trainDataset[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
trainDataset[['EmbarkedCode', 'Survived']].groupby(['EmbarkedCode'], as_index=False).mean().sort_values(by='Survived', ascending=False)
trainDataset[['SocialCategory', 'Survived']].groupby(['SocialCategory'], as_index=False).mean().sort_values(by='Survived', ascending=False)
trainDataset[['IsChild', 'Survived']].groupby(['IsChild'], as_index=False).mean()
trainDataset[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
#trainDataset[['AgeCategory', 'Survived']].groupby(['AgeCategory'], as_index=False).mean().sort_values(by='AgeCategory', ascending=True)


#Split into training dataset into train and test
X = trainDataset.loc[:, trainDataset.columns != 'Survived']
y = trainDataset['Survived'].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
CV = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)


#Train Random Forest model
randomForestModels = {
    'n_estimators' : [10, 50, 80, 100, 200],
    'criterion' : ['gini','entropy'],
    'random_state' : [0],
    'max_features': ['sqrt', 'auto', 'log2'],
    'min_samples_leaf': [1, 3, 10],
    'bootstrap': [True, False],
}

randomForestModel = RandomForestClassifier()

randomForestGridSearch = GridSearchCV(estimator = randomForestModel, param_grid = randomForestModels, cv = CV)
randomForestGridSearch.fit(X_train, y_train).score(X_test, y_test)
print(randomForestGridSearch.best_score_)
randomForestGridSearch.best_estimator_.get_params()

features = pd.DataFrame()
features['feature'] = X_train.columns
features['importance'] = randomForestGridSearch.best_estimator_.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(20, 20))

#Train SVM  model
svmModels = [
    {
      'C' : [1, 10, 100, 10000],
      'kernel' : ['rbf'],
      'gamma' : [0.0001, 0.001, (1 / 11)]
    }
]
svmModel = SVC(probability = True)

svmGridSearch = GridSearchCV(estimator = svmModel, param_grid = svmModels, cv = CV)
svmGridSearch.fit(X_train, y_train).score(X_test, y_test)
print(svmGridSearch.best_score_)
svmGridSearch.best_estimator_.get_params()

possibleNeuralNetworkParameters = [{  
        'hidden_layer_sizes' : [1000, (200,250,200)],
        'activation' :['relu'],
        'alpha' : [0.00001, 0.001, 0.01],
        'learning_rate' : ['constant', 'adaptive']
}]

neuralNetworkModel = MLPClassifier()
neuralNetworkSearch = GridSearchCV(estimator = neuralNetworkModel, param_grid = possibleNeuralNetworkParameters, cv = CV)
neuralNetworkSearch.fit(X_train, y_train).score(X_test, y_test)
print(neuralNetworkSearch.best_score_)
neuralNetworkSearch.best_estimator_.get_params()


predict_submission_with_model(randomForestGridSearch.best_estimator_, 'random-forest')
predict_submission_with_model(svmGridSearch.best_estimator_, 'svc')
predict_submission_with_model(neuralNetworkSearch.best_estimator_, 'neural-network')