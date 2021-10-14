from collections import Counter
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#set pycharm settings to display all columns of dataframe
pd.set_option('display.max_columns', None)

# read data
data = pd.read_csv('train.csv')
data.rename(columns={'X1': 'Item_ID', 'X2': 'Item_Weight', 'X3': 'Fats_Amount', 'X4': 'Display_Area', 'X5': 'Category', 'X6': 'price', 'X7': 'Store_ID',
            'X8': 'Store_establishment_year', 'X9': 'Store_size', 'X10': 'Store_location', 'Y': 'Class'}, inplace=True)



def storeSizeImputation(storeYear, StoreLocation, storeSize):
    if (storeSize != storeSize):
        if StoreLocation == 'Tier 2':
            return 'Small'
        if storeYear == '1997' or storeYear == '2002' or storeYear == '2004' or storeYear == '2007':
            return 'Small'
        if storeYear == '1999' or storeYear == '2009':
            return 'Medium'
        if storeYear == '1987':
            return 'High'
        if storeYear == '1985' and StoreLocation == 'Tier 3':
            return 'Medium'
        if storeYear == '1985' and StoreLocation == 'Tier 1':
            return 'Small'
        if storeYear == '1998':
            return 'Medium'
    else:
        return storeSize

#*********Data Cleaning*********

data["Fats_Amount"].replace(
    {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"}, inplace=True)

data['Item_Weight'].fillna((data['Item_Weight'].median()), inplace=True)

data['Store_size'] = data.apply(lambda x: storeSizeImputation(
    str(x.Store_establishment_year), x.Store_location, x.Store_size), axis=1)


#*********** Preprocessing **********

#drop store id and item id columns cause they're logically irrelevant
data.drop(['Item_ID', 'Store_ID'], axis=1, inplace=True)
# Encoding
fatsAmountLabelEncoder = preprocessing.LabelEncoder()
data['Fats_Amount'] = fatsAmountLabelEncoder.fit_transform(data['Fats_Amount'])

categoryLabelEncoder = preprocessing.LabelEncoder()
data['Category'] = categoryLabelEncoder.fit_transform(data['Category'])

storeSizeLabelEncoder = preprocessing.LabelEncoder()
data['Store_size'] = storeSizeLabelEncoder.fit_transform(data['Store_size'])

storeLocationLabelEncoder = preprocessing.LabelEncoder()
data['Store_location'] = storeLocationLabelEncoder.fit_transform(
    data['Store_location'])

#Min Max Normalization
maxWeight = data['Item_Weight'].max()
minWeight = data['Item_Weight'].min()

maxPrice = data['price'].max()
minPrice = data['price'].min()

maxYear = data['Store_establishment_year'].max()
minYear = data['Store_establishment_year'].min()

maxFats = data['Fats_Amount'].max()
minFats = data['Fats_Amount'].min()

maxCategory = data['Category'].max()
minCategory = data['Category'].min()

maxStoreSize = data['Store_size'].max()
minStoreSize = data['Store_size'].min()

maxStoreLocation = data['Store_location'].max()
minStoreLocation = data['Store_location'].min()

maxDisplayArea = data['Display_Area'].max()
minDisplayArea = data['Display_Area'].min()

cols_to_norm = ['Item_Weight', 'price', 'Store_establishment_year',
                'Fats_Amount', 'Category', 'Store_size', 'Store_location', 'Display_Area']

data[cols_to_norm] = data[cols_to_norm].apply(
    lambda x: (x - x.min()) / (x.max() - x.min()))

# Correlation Matrix
print(data.corr(method='pearson').abs().sort_values(by='Class'))

#droppin columns (features) with low correlation
data.drop(['Fats_Amount', 'Category', 'price', 'Item_Weight',
          'Display_Area'], axis=1, inplace=True)

# Splitting data into training and testing data
xCols = data.shape[1]-1
yCol = data.shape[1]-1
x = data.iloc[:, 0:xCols]
y = data.iloc[:, yCol]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)


#####Classification Models#####


#Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
print("Gaussian Naive Bayes model accuracy Test Set(in %):",
      metrics.f1_score(y_test, y_pred, average='micro')*100)
y_pred = gnb.predict(x_train)
print("Gaussian Naive Bayes model accuracy Train Set(in %):",
      metrics.f1_score(y_train, y_pred, average='micro')*100)
print(' ')

#support vector machine
svm = SVC()
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)
print("SVM model accuracy Test Set(in %):",
      metrics.f1_score(y_test, y_pred, average='micro')*100)
y_pred = svm.predict(x_train)
print("SVM model accuracy Train Set(in %):",
      metrics.f1_score(y_train, y_pred, average='micro')*100)
print(' ')

#adaboost classifier
ada = AdaBoostClassifier(learning_rate=0.1)
ada.fit(x_train, y_train)
y_pred = ada.predict(x_test)
print("AdaBoost model accuracy Test Set(in %):",
      metrics.f1_score(y_test, y_pred, average='micro')*100)
y_pred = ada.predict(x_train)
print("AdaBoost model accuracy Train Set(in %):",
      metrics.f1_score(y_train, y_pred, average='micro')*100)
print(' ')

#K nearest neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print("KNN model accuracy Test Set(in %):",
      metrics.f1_score(y_test, y_pred, average='micro')*100)
y_pred = knn.predict(x_train)
print("KNN model accuracy Train Set(in %):",
      metrics.f1_score(y_train, y_pred, average='micro')*100)
print(' ')

#random forrest
randomForest = RandomForestClassifier(n_estimators=50, max_depth=5)
randomForest.fit(x_train, y_train)
y_pred = randomForest.predict(x_test)
print("Random Forest model accuracy Test Set(in %):",
      metrics.f1_score(y_test, y_pred, average='micro')*100)
y_pred = randomForest.predict(x_train)
print("Random Forest model accuracy Train Set(in %):",
      metrics.f1_score(y_train, y_pred, average='micro')*100)
print(' ')

#decision tree
decisionTree = DecisionTreeClassifier(max_depth=5)
decisionTree.fit(x_train, y_train)
y_pred = decisionTree.predict(x_test)
print("Decision Tree model accuracy Test Set(in %):",
      metrics.f1_score(y_test, y_pred, average='micro')*100)
y_pred = decisionTree.predict(x_train)
print("Decision Tree model accuracy Train Set(in %):",
      metrics.f1_score(y_train, y_pred, average='micro')*100)
print(' ')

#mlp classifier
mlp = MLPClassifier()
mlp.fit(x_train, y_train)
y_pred = mlp.predict(x_test)
print("MLP model accuracy Test Set(in %):",
      metrics.f1_score(y_test, y_pred, average='micro')*100)
y_pred = mlp.predict(x_train)
print("MLP model accuracy Train Set(in %):",
      metrics.f1_score(y_train, y_pred, average='micro')*100)
print(' ')


##############################################################################################################
testData = pd.read_csv('test.csv')
testData.rename(columns={'X1': 'Item_ID', 'X2': 'Item_Weight', 'X3': 'Fats_Amount', 'X4': 'Display_Area', 'X5': 'Category', 'X6': 'price', 'X7': 'Store_ID',
                         'X8': 'Store_establishment_year', 'X9': 'Store_size', 'X10': 'Store_location'}, inplace=True)

testData.drop(['Fats_Amount', 'Category', 'price', 'Item_Weight',
               'Display_Area', 'Item_ID', 'Store_ID'], axis=1, inplace=True)


testData['Store_size'] = testData.apply(lambda x: storeSizeImputation(
    str(x.Store_establishment_year), x.Store_location, x.Store_size), axis=1)

testData['Store_size'] = storeSizeLabelEncoder.transform(
    testData['Store_size'])

testData['Store_location'] = storeLocationLabelEncoder.transform(
    testData['Store_location'])


testData['Store_size'] = testData['Store_size'].apply(
    lambda x: (x - minStoreSize) / (maxStoreSize - minStoreSize))
testData['Store_location'] = testData['Store_location'].apply(
    lambda x: (x - minStoreLocation) / (maxStoreLocation - minStoreLocation))
testData['Store_establishment_year'] = testData['Store_establishment_year'].apply(
    lambda x: (x - minYear) / (maxYear - minYear))


y_pred = gnb.predict(testData)
pd.DataFrame(y_pred, columns=['label']).to_csv('sample_submission1.csv')

y_pred = svm.predict(testData)
pd.DataFrame(y_pred, columns=['label']).to_csv('sample_submission2.csv')

y_pred = ada.predict(testData)
pd.DataFrame(y_pred, columns=['label']).to_csv('sample_submissio3.csv')

y_pred = knn.predict(testData)
pd.DataFrame(y_pred, columns=['label']).to_csv('sample_submission4.csv')

y_pred = randomForest.predict(testData)
pd.DataFrame(y_pred, columns=['label']).to_csv('sample_submission5.csv')

y_pred = decisionTree.predict(testData)
pd.DataFrame(y_pred, columns=['label']).to_csv('sample_submission6.csv')

y_pred = mlp.predict(testData)
pd.DataFrame(y_pred, columns=['label']).to_csv('sample_submission7.csv')



#qda = QuadraticDiscriminantAnalysis(store_covariance=True)
#qda.fit(x_train, y_train)
#y_pred = qda.predict(x_test)
# print("QDA model accuracy Test Set(in %):",
#      metrics.f1_score(y_test, y_pred, average='micro')*100)
#y_pred = qda.predict(x_train)
# print("QDA model accuracy Train Set(in %):",
#      metrics.f1_score(y_train, y_pred, average='micro')*100)
#print(' ')
