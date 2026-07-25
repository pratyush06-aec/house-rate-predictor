import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# import sklearn.datasets
from xgboost import XGBRegressor
from sklearn import metrics 

house_price_dataset= pd.read_csv('HousingData.csv')
print(house_price_dataset.head())

print(house_price_dataset.shape)

## Check for missing values

print(house_price_dataset.isnull().sum())

## Handle missing values

# CHAS is a binary column (0/1 river-adjacency flag) — use mode, not median/mean
house_price_dataset['CHAS'] = house_price_dataset['CHAS'].fillna(house_price_dataset['CHAS'].mode()[0])

# The rest are continuous and skewed (CRIM, LSTAT especially have long right tails)
# so median is more robust than mean here
skewed_cols = ['CRIM', 'ZN', 'INDUS', 'AGE', 'LSTAT']
house_price_dataset[skewed_cols] = house_price_dataset[skewed_cols].fillna(
    house_price_dataset[skewed_cols].median()
)

## for the CHAS mode is used as mode would give the typical class for this particular feature, whereas mean would give a meaningless fractional value and mediam would land on 0 everytime 
## for the other features median was choose to be used as the most frequent class for this particular features are required to be placed in place of the null values, whereas mean can be manipulated easily by drastically high or low values.

# Confirm all missing values are gone
print(house_price_dataset.isnull().sum())

print(house_price_dataset.describe())

## Now, plotting the correlation matrix to see how features are correlated with each other and with the target variable

correlation = house_price_dataset.corr()

plt.figure(figsize=(10, 10))

## (cbar) prints the long side-bar
## (correlation) prints the correlation values in the boxes
## (annot) prints the correlation values in the boxes
## (square) prints the square boxes
## (fmt) decides the number of digits after the decimal point
## (annot_kws) decides the size of the correlation values in the boxes
## (cmap) decides the color of the boxes
## (center) decides the center of the color map, long with the distribution of the values based on the colors of the boxes. The values greater than 0 would be placed towards the darker colored boxes and those less than 0 would be placed towards the light colored boxes, thus 0 becomes the center or neutral point.

sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Purples', center=0)
plt.title('Correlation Matrix')
plt.show()

x= house_price_dataset.drop(['MEDV'], axis=1)
y= house_price_dataset['MEDV']
print(x)
print(y)

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=2)
print(x.shape, x_train.shape, x_test.shape)

model= XGBRegressor()
model.fit(x_train, y_train)

training_data_prediction= model.predict(x_train)
print(training_data_prediction)

## for testing the accuracy of the model in regression problems, we used r2 score and mean_absolute_error here in this project
## the more closer the r2 score is to 1, the more accurate is the model and the more closer the mean_absolute_error is to 0, the more accurate is the model.

score1= metrics.r2_score(y_train, training_data_prediction)
score2= metrics.mean_absolute_error(y_train, training_data_prediction)
print(f'R2 score is {score1}')
print(f'Mean Absolute Error is {score2}')

plt.scatter(y_train, training_data_prediction)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Training Data: Actual vs Predicted')
plt.show()

testing_data_prediction= model.predict(x_test)
print(testing_data_prediction)

score_3= metrics.r2_score(y_test, testing_data_prediction)
score_4= metrics.mean_absolute_error(y_test, testing_data_prediction)
print(f'R2 score is {score_3}')
print(f'Mean Absolute Error is {score_4}')

plt.scatter(y_test, testing_data_prediction)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Testing Data: Actual vs Predicted')
plt.show()

## Now, we will try to improve the accuracy of the model by tuning the hyperparameters of the model.
model= XGBRegressor( n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=2)
model.fit(x_train, y_train)

training_data_prediction= model.predict(x_train)
print(training_data_prediction)

## for testing the accuracy of the model in regression problems, we used r2 score and mean_absolute_error here in this project
## the more closer the r2 score is to 1, the more accurate is the model and the more closer the mean_absolute_error is to 0, the more accurate is the model.

score1= metrics.r2_score(y_train, training_data_prediction)
score2= metrics.mean_absolute_error(y_train, training_data_prediction)
print(f'R2 score is {score1}')
print(f'Mean Absolute Error is {score2}')

testing_data_prediction= model.predict(x_test)
print(testing_data_prediction)

score_3= metrics.r2_score(y_test, testing_data_prediction)
score_4= metrics.mean_absolute_error(y_test, testing_data_prediction)
print(f'R2 score is {score_3}')
print(f'Mean Absolute Error is {score_4}')

## Now, we will try to improve the accuracy of the model by implementing train/validation/test split.

# x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
# x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.2, random_state=2)
# print(x_train.shape, x_val.shape, x_test.shape)

# model = XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.05, random_state=2, early_stopping_rounds=20)
# model.fit(
#     x_train, y_train,
#     eval_set=[(x_val, y_val)],
#     verbose=False
# )

# training_data_prediction= model.predict(x_train)
# print(training_data_prediction)

# ## for testing the accuracy of the model in regression problems, we used r2 score and mean_absolute_error here in this project
# ## the more closer the r2 score is to 1, the more accurate is the model and the more closer the mean_absolute_error is to 0, the more accurate is the model.

# score1= metrics.r2_score(y_train, training_data_prediction)
# score2= metrics.mean_absolute_error(y_train, training_data_prediction)
# print(f'R2 score is {score1}')
# print(f'Mean Absolute Error is {score2}')

# testing_data_prediction= model.predict(x_test)
# print(testing_data_prediction)

# score_3= metrics.r2_score(y_test, testing_data_prediction)
# score_4= metrics.mean_absolute_error(y_test, testing_data_prediction)
# print(f'R2 score is {score_3}')
# print(f'Mean Absolute Error is {score_4}')

# print("Best iteration:", model.best_iteration)


