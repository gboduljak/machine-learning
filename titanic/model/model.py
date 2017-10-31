import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from pandas import DataFrame

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, learning_curve
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


trainDataset = pd.read_csv('../input/train.csv')
testDataset  = pd.read_csv('../input/test.csv')

#Preprocess and clean train dataset
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


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
    dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ]).map(lambda ticket : ticket == 'X').astype(bool)


    dataset['IsChild'] = False
    dataset.loc[(dataset.Age <= 16), 'IsChild'] = True
    
    dataset['IsAlone'] = 0
    dataset.loc[(dataset['FamilySize'] == 1), 'IsAlone'] = 1
    dataset.loc[(dataset['FamilySize'] > 4),  'IsAlone'] = 2
    
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
    dataset['Age*SocialCategory'] =  dataset.Age * dataset['SocialCategory']

    dataset['AgeCategory'] = 5
    dataset.loc[ dataset['Age'] <= 16, 'AgeCategory'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'AgeCategory'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'AgeCategory'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'AgeCategory'] = 3
    dataset.loc[ dataset['Age'] > 64, 'AgeCategory'] = 4
    

    
    dataset['Fare'].fillna(dataset['Fare'].dropna().mean(), inplace=True)

    dataset.loc[ dataset['Fare'] <= 7.91, 'FareBand'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'FareBand'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'FareBand']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'FareBand'] = 3
    dataset['FareBand'] = dataset['FareBand'].astype(int)


def preprocess_and_clean(dataset, isTestDataset = False):     
    dataset['Embarked'] = dataset['Embarked'].fillna(dataset.Embarked.dropna().mode()[0])
    dataset['Fare'].fillna(dataset['Fare'].dropna().median(), inplace=True)

    dataset['Age']=dataset.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.mean())).astype(int)
    
    
    dataset['Gender']       = dataset['Sex'].map({ 'female':0, 'male':1}).astype(int)
    dataset['EmbarkedCode'] = dataset['Embarked'].map({ 'S':0, 'C':1, 'Q' : 2, 'N' : 3}).astype(int)

    
    dataset.loc[(dataset.Age.isnull()),'Age'] = dataset.Age.median()

    extract_social_category_from_name_title(dataset)
    produce_new_features(dataset)
    
    
    if isTestDataset == False:
        dataset.drop(['Name','Ticket', 'Sex', 'Embarked', 'SibSp', 'Parch', 'FamilySize', 'Age', 'Fare', 'PassengerId'], 1, inplace=True)
    else:
        dataset.drop(['Name','Ticket', 'Sex', 'Embarked', 'SibSp', 'Parch', 'FamilySize', 'Age', 'Fare'], 1, inplace=True)
 
def predict_submission_with_model(model, modelName):
    predictions = {
        'PassengerId' : testDataset['PassengerId'].values,
        'Survived' : model.predict(testDataset.loc[:, testDataset.columns != 'PassengerId'])
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
CV = KFold(n_splits=10, random_state=0)


#Train Random Forest model
randomForestModels = {
    'n_estimators' : [10, 100 ,500],
    'criterion' : ['gini','entropy'],
    'min_samples_leaf' : [4,6,12],
    'min_samples_split': [2,4,9, 12],
    'random_state' : [0]
}

randomForestModel = RandomForestClassifier()

randomForestGridSearch = GridSearchCV(estimator = randomForestModel, param_grid = randomForestModels, cv = CV, n_jobs = 10)
randomForestGridSearch.fit(X_train, y_train).score(X_test, y_test)
print(randomForestGridSearch.best_score_)
randomForestGridSearch.best_estimator_.get_params()

features = pd.DataFrame()
features['feature'] = X_train.columns
features['importance'] = randomForestGridSearch.best_estimator_.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(20, 20))

plot_learning_curve(estimator = randomForestGridSearch.best_estimator_, title = 'Random forest', X = X_train, y = y_train)

#Train SVM  model
svmModels = [
    {
      'C' : [1, 10, 100, 10000],
      'kernel' : ['rbf'],
      'gamma' : [0.0001, 0.001, (1 / 11)]
    }
]
svmModel = SVC(probability = True)

svmGridSearch = GridSearchCV(estimator = svmModel, param_grid = svmModels, cv = CV, n_jobs = 5)
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


votingClassifier = VotingClassifier(
        estimators = [ 
                ('random-forest', randomForestGridSearch.best_estimator_), 
                ('svm', svmGridSearch.best_estimator_),
                ('mlp', neuralNetworkSearch.best_estimator_)
        ],
        voting = 'soft'
) 
votingClassifier.fit(X_train, y_train)
votingClassifier.score(X_test, y_test)

predict_submission_with_model(votingClassifier, 'voting-classifier')
predict_submission_with_model(randomForestGridSearch.best_estimator_, 'random-forest')
predict_submission_with_model(svmGridSearch.best_estimator_, 'svc')
predict_submission_with_model(neuralNetworkSearch.best_estimator_, 'neural-network')