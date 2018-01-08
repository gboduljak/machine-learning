from sklearn.tree import DecisionTreeRegressor
import numpy as np
import copy

socialCategoriesFromTitle = {
    'Unknown' : 0,
    'Mr': 1,
    'Miss': 2,
    'Mrs': 3,
    'Master': 4,
    'Royalty': 5,
    'Officer': 6
}

ageCategories = {
    'Unknown' : 0,
    'Child': 1,
    'YoungerAdult': 2,
    'Adult': 3,
    'OlderAdult': 4,
    'Senior': 5
}

genders           = { 'male' : 1, 'female' : 0 }
embarkedLocations = { 'S':0, 'C':1, 'Q' : 2, 'N' : 3}

def produce_family_size_and_is_alone(dataset):
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']
    dataset['IsAlone'] = 0
    dataset.loc[(dataset['FamilySize'] == 1), 'IsAlone'] = 1
    dataset.loc[(dataset['FamilySize'] > 4), 'IsAlone'] = 2

def produce_is_child(dataset):
    dataset['IsChild'] = False
    dataset.loc[(dataset.Age <= 16), 'IsChild'] = True

def produce_gender(dataset):
    dataset['Gender'] = dataset['Sex'].map(genders).astype(int)

def produce_embarked_location(dataset):
    dataset['Embarked'].fillna('N', inplace=True )
    dataset['EmbarkedLocation'] = dataset['Embarked'].map({ 'S':0, 'C':1, 'Q' : 2, 'N' : 3}).astype(int)

def produce_social_category(dataset):
    dataset['SocialCategory'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['SocialCategory'] = dataset['SocialCategory'].replace(['Lady', 'Countess', 'Dona'], 'Royalty')
    dataset['SocialCategory'] = dataset['SocialCategory'].replace(['Mme'], 'Mrs')
    dataset['SocialCategory'] = dataset['SocialCategory'].replace(['Mlle', 'Ms'], 'Miss')
    dataset['SocialCategory'] = dataset['SocialCategory'].replace(['Capt', 'Col', 'Major', 'Rev'], 'Officer')
    dataset['SocialCategory'] = dataset['SocialCategory'].replace(['Jonkheer', 'Don', 'Sir'], 'Royalty')

    dataset.loc[(dataset.Sex == 'male') & (dataset.SocialCategory == 'Dr'), 'SocialCategory'] = 'Mr'
    dataset.loc[(dataset.Sex == 'female') & (dataset.SocialCategory == 'Dr'), 'SocialCategory'] = 'Mrs'

    dataset['SocialCategory'] = dataset['SocialCategory'].map(socialCategoriesFromTitle)
    dataset['SocialCategory'] = dataset['SocialCategory'].fillna(socialCategoriesFromTitle['Unknown'])

def produce_age_category(dataset):
    dataset['AgeCategory'] = ageCategories['Unknown']
    dataset.loc[dataset['Age'] <= 16, 'AgeCategory'] = ageCategories['Child']
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'AgeCategory'] = ageCategories['YoungerAdult']
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'AgeCategory'] = ageCategories['Adult']
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'AgeCategory'] = ageCategories['OlderAdult']
    dataset.loc[dataset['Age'] > 64, 'AgeCategory'] = ageCategories['Senior']

def produce_fare_category(dataset):
    dataset.loc[ dataset['Fare'] <= 7.91, 'FareCategory'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'FareCategory'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'FareCategory']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'FareCategory'] = 3
    dataset.loc[dataset['Fare'] > 51, 'FareCategory'] = 4
    dataset.loc[dataset['Fare'] > 61, 'FareCategory'] = 5

def produce_age_correlated_features(dataset):
    dataset['Age*SocialCategory'] = dataset['Age'] * dataset['SocialCategory']
    dataset['Age*Gender'] = dataset['Age'] * dataset['Gender']

def fill_missing_age(dataset):
    clearDataset = copy.deepcopy(dataset.drop(['Embarked', 'Cabin', 'Ticket', 'Sex', 'Name'], axis=1))

    clearDataset.dropna(inplace=True)
    X = copy.deepcopy(clearDataset.loc[:, clearDataset.columns != 'Age'])
    y = copy.deepcopy(clearDataset.loc[:, clearDataset.columns == 'Age'])

    tree = DecisionTreeRegressor()
    tree.fit(X, y)
    predictions = tree.predict(dataset.drop(['Embarked', 'Cabin', 'Ticket', 'Sex', 'Name', 'Age'], axis=1))

    for index, row in dataset.iterrows():
        if (not np.isfinite(row.Age)):
            dataset.set_value(index, 'Age', predictions[index])

def fill_missing_fare(dataset):
    clearDataset = copy.deepcopy(dataset.drop(['Embarked', 'Cabin', 'Ticket', 'Sex', 'Name'], axis=1))

    clearDataset.dropna(inplace=True)
    X = copy.deepcopy(clearDataset.loc[:, clearDataset.columns != 'Fare'])
    y = copy.deepcopy(clearDataset.loc[:, clearDataset.columns == 'Fare'])

    tree = DecisionTreeRegressor()
    tree.fit(X, y)
    predictions = tree.predict(dataset.drop(['Embarked', 'Cabin', 'Ticket', 'Sex', 'Name', 'Fare'], axis=1))

    for index, row in dataset.iterrows():
        if (not np.isfinite(row.Age)):
            dataset.set_value(index, 'Fare', predictions[index])

def reduce_fare_skew(dataset):
    dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)