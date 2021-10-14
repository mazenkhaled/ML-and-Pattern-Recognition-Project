import random
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import ExtraTreesRegressor

#set pycharm settings to display all columns of dataframe
pd.set_option('display.max_columns', None)


def storeSizeImputation(storeType, StoreLocation, storeSize):
    if (storeSize != storeSize):
        if storeType == 'Supermarket Type2':
            return 'Medium'

        if storeType == 'Supermarket Type3':
            return 'Medium'

        if storeType == 'Grocery Store':
            return 'Small'

        if storeType == 'Supermarket Type1':
            if StoreLocation == "Tier 1":
                x = random.choice([1, 2])
                if x == 1:
                    return 'Medium'
                else:
                    return 'Small'
            if StoreLocation == "Tier 2":
                return 'Small'

            if StoreLocation == "Tier 3":
                return 'High'
    else:
        return storeSize




# read data
data = pd.read_csv('train.csv')

#name each column
data.rename(columns={'X1': 'Item_ID', 'X2': 'Item_Weight', 'X3': 'Fats_Amount', 'X4': 'Display_Area', 'X5': 'Category', 'X6': 'price', 'X7': 'Store_ID',
            'X8': 'Store_establishment_year', 'X9': 'Store_size', 'X10': 'Store_location', 'X11': 'Store_Type', 'Y': 'Item_sales'}, inplace=True)


# check for null values per column
sns.heatmap(data.isnull(), cbar=False)
plt.show()

#check store sizes correlated with store types
sns.histplot(binwidth=0.5, x="Store_Type", hue="Store_size", data=data, stat="count", multiple="stack")
plt.show()

#check store sizes correlated with supermarket type 1 only based on location
supermarket_1_data = data.loc[data['Store_Type'] == 'Supermarket Type1']
sns.histplot(binwidth=0.5, x="Store_location", hue="Store_size", data=supermarket_1_data, stat="count", multiple="stack")
plt.show()


#fill nan values in Store Size column
data['Store_size'] = data.apply(lambda x: storeSizeImputation(
    x.Store_Type, x.Store_location, x.Store_size), axis=1)

#make naming of fats amount unified
data["Fats_Amount"].replace(
    {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"}, inplace=True)


#correct items with 0% display area to have the mean of that column
data.loc[data.Display_Area == 0,
         'Display_Area'] = data['Display_Area'].mean(skipna=True)

sns.boxplot(x = data.Fats_Amount, y = data.Item_Weight)
plt.show()

#fill nan values of item weight to have the median of that column
data['Item_Weight'].fillna((data['Item_Weight'].median()), inplace=True)

sns.heatmap(data.isnull(), cbar=False)
plt.show()


#drop store id and item id columns cause they're logically irrelevant
data.drop(['Item_ID', 'Store_ID'], axis=1, inplace=True)



#Label Encoding
fatsAmountLabelEncoder = preprocessing.LabelEncoder()
data['Fats_Amount'] = fatsAmountLabelEncoder.fit_transform(data['Fats_Amount'])

categoryLabelEncoder = preprocessing.LabelEncoder()
data['Category'] = categoryLabelEncoder.fit_transform(data['Category'])

storeSizeLabelEncoder = preprocessing.LabelEncoder()
data['Store_size'] = storeSizeLabelEncoder.fit_transform(data['Store_size'])

storeLocationLabelEncoder = preprocessing.LabelEncoder()
data['Store_location'] = storeLocationLabelEncoder.fit_transform(
    data['Store_location'])

storeTypeLabelEncoder = preprocessing.LabelEncoder()
data['Store_Type'] = storeTypeLabelEncoder.fit_transform(data['Store_Type'])



#min max normalization
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

maxStoreType = data['Store_Type'].max()
minStoreType = data['Store_Type'].min()

cols_to_norm = ['Item_Weight', 'price', 'Store_establishment_year',
                'Fats_Amount', 'Category', 'Store_size', 'Store_location', 'Store_Type']
data[cols_to_norm] = data[cols_to_norm].apply(
    lambda x: (x - x.min()) / (x.max() - x.min()))

#viewing correlation matrix with output (item sales)
print(data.corr(method ='pearson').abs().sort_values(by = 'Item_sales' , ascending = False))


#droppin columns (features) with low correlation
data.drop(['Fats_Amount', 'Store_establishment_year',
          'Item_Weight', 'Category', 'Store_location'], axis=1, inplace=True)


#splitting our columns into X only and Y only
xCols = data.shape[1]-1
yCol = data.shape[1]-1
x = data.iloc[:, 0:xCols]
y = data.iloc[:, yCol]



x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)



#Polynomial regression models
PolyRegressionModel = LinearRegression().fit(x_train, y_train)
predictions = PolyRegressionModel.predict(x_train)
error = mean_absolute_error(y_train, predictions)
print("linear MAE of training data", error)
predictions = PolyRegressionModel.predict(x_test)
error = mean_absolute_error(y_test, predictions)
print("linear MAE of testing data ", error)


poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x_train)
poly.fit(x_train, y_train)
PolyRegressionModel = LinearRegression()
PolyRegressionModel.fit(x_poly, y_train)
predictions = PolyRegressionModel.predict(poly.fit_transform(x_train))
error = mean_absolute_error(y_train, predictions)
print("poly 2 MAE of training data", error)
predictions = PolyRegressionModel.predict(poly.fit_transform(x_test))
error = mean_absolute_error(y_test, predictions)
print("poly 2 MAE of testing data ", error)


poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(x_train)
poly.fit(x_train, y_train)
PolyRegressionModel = LinearRegression()
PolyRegressionModel.fit(x_poly, y_train)
predictions = PolyRegressionModel.predict(poly.fit_transform(x_train))
error = mean_absolute_error(y_train, predictions)
print("poly 3 MAE of training data", error)
predictions = PolyRegressionModel.predict(poly.fit_transform(x_test))
error = mean_absolute_error(y_test, predictions)
print("poly 3 MAE of testing data ", error)


plot=pd.DataFrame()
plot['Target']=y_test
plot['Predictions']=predictions
#
sns.lmplot('Target','Predictions',data=plot,height=6,aspect=2,line_kws={'color':'red'},scatter_kws={'alpha':0.4,'color':'black'})
plt.title('Polynomial regression \n Score: {0:.2f}'.format(error),size=25)

plt.show()


#random forest regressor
randomForestModel = RandomForestRegressor(max_depth=5,)
randomForestModel.fit(x_train, y_train)
predictions = randomForestModel.predict(x_train)
error = mean_absolute_error(y_train, predictions)
print("Random Forest MAE of training data", error)
predictions = randomForestModel.predict(x_test)
error = mean_absolute_error(y_test, predictions)
print("Random Forest MAE of testing data ", error)

plot=pd.DataFrame()
plot['Target']=y_test
plot['Predictions']=predictions
#
sns.lmplot('Target','Predictions',data=plot,height=6,aspect=2,line_kws={'color':'red'},scatter_kws={'alpha':0.4,'color':'black'})
plt.title('Random Forest regression \n Score: {0:.2f}'.format(error),size=25)

plt.show()


#extra tree regressor
extraTreesModel = ExtraTreesRegressor(n_estimators=150, criterion='mae', max_depth=7)
extraTreesModel.fit(x_train, y_train)
predictions = extraTreesModel.predict(x_train)
error = mean_absolute_error(y_train, predictions)
print("Extra Trees Regressor MAE of training data", error)
predictions = extraTreesModel.predict(x_test)
error = mean_absolute_error(y_test, predictions)
print("Extra Trees Regressor MAE of testing data ", error)

#adaboost regressor
adaboostModel = AdaBoostRegressor()
adaboostModel.fit(x_train, y_train)
predictions = adaboostModel.predict(x_train)
error = mean_absolute_error(y_train, predictions)
print("AdaBoost Regressor MAE of training data", error)
predictions = adaboostModel.predict(x_test)
error = mean_absolute_error(y_test, predictions)
print("AdaBoost Regressor MAE of testing data ", error)


# Writing CSV
#x= dataset.corr().abs()
# sample=pd.read_csv('sample_submission.csv')
# sample['Y']=predictions
#sample.to_csv(r'sample_submission.csv', index = False)

# --------------------------------------------------------------------------
# read data
testData = pd.read_csv('test.csv')
testData.rename(columns={'X1': 'Item_ID', 'X2': 'Item_Weight', 'X3': 'Fats_Amount', 'X4': 'Display_Area', 'X5': 'Category', 'X6': 'price', 'X7': 'Store_ID',
                         'X8': 'Store_establishment_year', 'X9': 'Store_size', 'X10': 'Store_location', 'X11': 'Store_Type'}, inplace=True)




testData['Store_size'] = testData.apply(lambda x: storeSizeImputation(
    x.Store_Type, x.Store_location, x.Store_size), axis=1)


testData["Fats_Amount"].replace(
    {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"}, inplace=True)

testData.loc[testData.Display_Area == 0,
             'Display_Area'] = testData['Display_Area'].mean(skipna=True)

testData['Item_Weight'].fillna(
    (testData['Item_Weight'].median()), inplace=True)

testData.drop(['Item_ID', 'Store_ID'], axis=1, inplace=True)


testData['Fats_Amount'] = fatsAmountLabelEncoder.transform(
    testData['Fats_Amount'])

testData['Category'] = categoryLabelEncoder.transform(testData['Category'])

testData['Store_size'] = storeSizeLabelEncoder.transform(
    testData['Store_size'])

testData['Store_location'] = storeLocationLabelEncoder.transform(
    testData['Store_location'])

testData['Store_Type'] = storeTypeLabelEncoder.transform(
    testData['Store_Type'])


testData['Item_Weight'] = testData['Item_Weight'].apply(
    lambda x: (x - minWeight) / (maxWeight - minWeight))

testData['price'] = testData['price'].apply(
    lambda x: (x - minPrice) / (maxPrice - minPrice))

testData['Store_establishment_year'] = testData['Store_establishment_year'].apply(
    lambda x: (x - minYear) / (maxYear - minYear))

testData['Fats_Amount'] = testData['Fats_Amount'].apply(
    lambda x: (x - minFats) / (maxFats - minFats))

testData['Category'] = testData['Category'].apply(
    lambda x: (x - minCategory) / (maxCategory - minCategory))

testData['Store_size'] = testData['Store_size'].apply(
    lambda x: (x - minStoreSize) / (maxStoreSize - minStoreSize))

testData['Store_location'] = testData['Store_location'].apply(
    lambda x: (x - minStoreLocation) / (maxStoreLocation - minStoreLocation))

testData['Store_Type'] = testData['Store_Type'].apply(
    lambda x: (x - minStoreType) / (maxStoreType - minStoreType))

testData.drop(['Fats_Amount', 'Store_establishment_year', 'Category', 'Item_Weight', 'Store_location'],
              axis=1, inplace=True)

#write results of prediction of each model to a file
predictions = PolyRegressionModel.predict(poly.fit_transform(testData))
pd.DataFrame(predictions, columns=['Y']).to_csv('sample_submission1.csv')

predictions = randomForestModel.predict(testData)
pd.DataFrame(predictions, columns=['Y']).to_csv('sample_submission2.csv')

predictions = extraTreesModel.predict(testData)
pd.DataFrame(predictions, columns=['Y']).to_csv('sample_submission3.csv')

predictions = adaboostModel.predict(testData)
pd.DataFrame(predictions, columns=['Y']).to_csv('sample_submission4.csv')


