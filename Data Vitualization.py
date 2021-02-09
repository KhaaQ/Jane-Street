
# Written By: Khaula
# Jan 19, 2021

# Standard imports

import time
notebookstart= time.time()

import numpy as np
from numpy import loadtxt
import pandas as pd
from pandas import DataFrame
import matplotlib.pylab as plt
from math import sqrt


from numpy import concatenate
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

# Modifying and splitting the data
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

# XGboost model
import xgboost as xgb

# Model selection tools
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV

# For Saving models
import joblib

# Error metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# for visualization
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
import warnings
warnings.filterwarnings('ignore')
np.random.seed(9)

# TrainData = pd.read_csv('/home/khaaq/Documents/Jane_Street/train.csv', header = 0) 
# TestData = pd.read_csv('/home/khaaq/Documents/Jane_Street/example_test.csv')

# # print("Train Data information")
# # TrainData.info()

# # print("Test Data information")
# # TestData.info()


# """
# Reduce Memory Usage by 75%
# https://www.kaggle.com/tomwarrens/nan-values-depending-on-time-of-day
# """

# ## Reduce Memory

# def reduce_memory_usage(df):
    
#     start_memory = df.memory_usage().sum() / 1024**2
#     print(f"Memory usage of dataframe is {start_memory} MB")
    
#     for col in df.columns:
#         col_type = df[col].dtype
        
#         if col_type != 'object':
#             c_min = df[col].min()
#             c_max = df[col].max()
            
#             if str(col_type)[:3] == 'int':
#                 if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
#                     df[col] = df[col].astype(np.int8)
#                 elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
#                     df[col] = df[col].astype(np.int16)
#                 elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
#                     df[col] = df[col].astype(np.int32)
#                 elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
#                     df[col] = df[col].astype(np.int64)
            
#             else:
# #                 reducing float16 for calculating numpy.nanmean
# #                 if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
# #                     df[col] = df[col].astype(np.float16)
#                 if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
#                     df[col] = df[col].astype(np.float32)
#                 else:
#                     pass
#         else:
#             df[col] = df[col].astype('category')
    
#     end_memory = df.memory_usage().sum() / 1024**2
#     print(f"Memory usage of dataframe after reduction {end_memory} MB")
#     print(f"Reduced by {100 * (start_memory - end_memory) / start_memory} % ")
#     return df

# TrainData = reduce_memory_usage(TrainData)
# TrainData.info()

# """
# The codes from 'NaN values depending on Time of Day'
# https://www.kaggle.com/tomwarrens/nan-values-depending-on-time-of-day
# """

# def chunks(l, n):
#     """ Yield n successive chunks from l.
#     """
#     newn = int(len(l) / n)
#     for i in range(0, n-1):
#         yield l[i*newn:i*newn+newn]
#     yield l[n*newn-newn:]


# """
# Printing Count percentage of Nan values
# The codes from 'NaN values depending on Time of Day'
# https://www.kaggle.com/tomwarrens/nan-values-depending-on-time-of-day
# """

# #count
# nan_values_train = (TrainData
#  .apply(lambda x: x.isna().sum(axis = 0)/len(TrainData))
#  .to_frame()
#  .rename(columns = {0: 'percentage_nan_values'})
# .sort_values('percentage_nan_values', ascending = False)
# )

# print((TrainData
#  .apply(lambda x: x.isna().sum(axis = 0))
#  .to_frame()
#  .rename(columns = {0: 'count_nan_values'})
# .sort_values('count_nan_values', ascending = False)
# .transpose()), nan_values_train.transpose(),
#        print("Number of features with at least one NaN value: {}/{}".format(len(nan_values_train.query('percentage_nan_values>0')),
#                                                                            len(TrainData.columns))))


# # Drawing/ plotting percentage of missing values
# fig, ax = plt.subplots(figsize = (20, 12))

# sns.set_palette("RdBu", 10)
# #RdBu, YlGn
# ax = sns.barplot(x='percentage_nan_values', 
#             y='feature', 
#             palette = 'GnBu_r',
#             data=nan_values_train.reset_index().rename(columns = {'index': 'feature'}).head(40))

# for p in ax.patches:
#     width = p.get_width() 
#     if width < 0.01:# get bar length
#         ax.text(width,       # set the text at 1 unit right of the bar
#             p.get_y() + p.get_height() / 2, # get Y coordinate + X coordinate / 2
#             '{:1.4f}'.format(width), # set variable to display, 2 decimals
#             ha = 'left',   # horizontal alignment
#             va = 'center')  # vertical alignment
#     else:
#         if width < 0.04:
#             color_text = 'black'
#         else:
#             color_text = 'white'
#         ax.text(width /2, 
#                 # set the text at 1 unit right of the bar
#             p.get_y() + p.get_height() / 2, # get Y coordinate + X coordinate / 2
#             '{:1.4f}'.format(width), # set variable to display, 2 decimals
#             ha = 'left',   # horizontal alignment
#             va = 'center',
#             color = color_text,
#             fontsize = 10)  # vertical alignment

# ax.set_title('Top 40 Features for percentage of NaN Values')

# # Finding the null values in each column:
# print("Null values in Train Data")
# print(TrainData.isnull().sum())  # Only obserded value columns had missing values, so we will have to drop the rows:


# TrainData.interpolate(method ='linear', limit_direction ='forward', inplace = True)
# print(TrainData.head(20))


# print("Null values in Train Data")
# print(TrainData.isnull().sum())


# export_csv = TrainData.to_csv (r'/home/khaaq/Documents/Jane_Street/NoNullTrain.csv', index = 0, header=True)

TrainData = pd.read_csv('/home/khaaq/Documents/Jane_Street/NoNullTrain.csv', header = 0) 
TestData = pd.read_csv('/home/khaaq/Documents/Jane_Street/example_test.csv')

print("Train Data information")
TrainData.info()

# print("Test Data information")
# TestData.info()


print('First 10 rows: \n', TrainData.head(10))
print('Train Data Last 10 rows: \n', TrainData.tail(10))

# print("Train Dataset \n")
# print('Data shape', TrainData.shape)
# print('Dataset Data types', TrainData.dtypes)
# print('Data describtions', TrainData.describe())
# # print('First 10 rows: \n', TrainData.head(10))
# print('Train Data Last 10 rows: \n', TrainData.tail(10))

# print("Test Dataset \n")
# print('Data shape', TestData.shape)
# print('Dataset Data types', TestData.dtypes)
# print('Data describtions', TestData.describe())
# print('Test Data First 10 rows: \n', TestData.head(10))
# print('Test Data Last 10 rows: \n',TestData.tail(10))


# Plotting individual feature
# plt.figure(figsize=(30,20))
# plt.title('4 Features: Date')
# plt.plot(TrainData.iloc[:, 0:4])
# # scatter_matrix(dataset)
# # plt.savefig('/home/khaaq/Documents/Jane_Street/Data_Preprocess_Visualize_Jan13_21/ScatterMatrixCorrelation.png')
# plt.show()



#Using Pearson Correlation
# plt.figure(figsize=(20,12))
# cor = dataset.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.title("Features Correlation", y=-0.1)
# plt.savefig('/home/khaaq/Documents/Jane_Street/Data_Preprocess_Visualize_Jan13_21/FeatureCorrelation2.png')
# plt.show()

# # Plotting correlation between each feature ScatterMatrixCorrelation, Using pandas library:
# plt.figure(figsize=(30,20))
# scatter_matrix(dataset)
# plt.savefig('/home/khaaq/Documents/Jane_Street/Data_Preprocess_Visualize_Jan13_21/ScatterMatrixCorrelation.png')
# plt.show()

# Use pandas library to plot each feature distribution: Histogram
# features = dataset.loc[:,:]
# features.hist(figsize=(16, 10))
# plt.savefig('/home/khaaq/Documents/Jane_Street/Data_Preprocess_Visualize_Jan13_21/HistogramAll.png')
# plt.show()

# Use pandas library to plot each feature Correlation:
# Using Pearson Correlation
# plt.figure(figsize=(16,10))
# cor = dataset.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.title("Selected Features Correlation", y=-0.1)
# plt.savefig('/home/khaaq/Documents/Jane_Street/Data_Preprocess_Visualize_Jan13_21/Selected_FeatureCorrelation.png')
# plt.show()


'''
X_Train = TrainData.iloc[:, 0:137]
y_Train = TrainData.iloc[:, 137:138]

X_Test = TestData.iloc[:, 0:132]
y_Test = TestData.iloc[:, 132:133]

print("Train Dataset\n")
print("X_train Shape ", X_Train.shape)
print("y_train Shape ", y_Train.shape)
# print("\n")
# print("X_train Head\n", X_Train.head(10))
# print("X_train Head\n", y_Train.head(10))


print("Test Data\n")
print("X_Test Shape ", X_Test.shape)
print("y_Test Shape ", y_Test.shape)
print("\n")
print("X_Test Head\n", X_Test.head(10))
print("X_Test Head\n", y_Test.head(10))


# to convert the dsata into matrix shape to become compatible with XGBoost
dtrain = xgb.DMatrix(X_Train, label=y_Train)

# xg_reg = xgb.XGBRegressor(gamma=0.3,  n_estimators= 1500, colsample_bytree = 1, learning_rate=0.1,  min_child_weight = 3, objective ='reg:squarederror', n_jobs=-1, max_depth=13,  eval_metric= 'rmse', random_state=42)

modelstart= time.time()
# print(xg_reg)

# xg_reg.fit(X_train,y_train)
print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))

XGBModel = '/home/khaaq/Documents/Jane_Street/XGB_picklefiles/XGBJaneStreet1.pkl'
# To save the xgboost model
# joblib.dump(xg_reg, XGBModel)
xg_reg = joblib.load(XGBModel)
print(xg_reg)
Train_Preds = xg_reg.predict(X_Train)

# Test_preds = xg_reg.predict(X_test)

# rmse = np.sqrt(mean_squared_error(y_train, Train_Preds))
# R2 = r2_score(y_train, Train_Preds)

# print("XGBoost Train RMSE: %f" % (rmse))
# print( "XGBoost Train r2 Score", r2_score(y_train, Train_Preds))



# #print(explained_variance_score(test_preds,y_test))

# print("XGBoost Test RMSE: %f" % np.sqrt(mean_squared_error(y_test, test_preds)))
# print( "Test r2 Score", r2_score(y_test, test_preds))


# 

# XGBRmse = '/home/khaula/Desktop/XGBoost1TRYRmseSQ.pkl'
# joblib.dump(rmse, XGBRmse)

# print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))

# print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))


'''


'''
# normalize features
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(values)
# specify the number of lag hours

# frame as supervised learning
reframed = timeseries_to_supervised(values, n_hours, 1)


print("reframed Values shape", reframed.shape)

print("reframed Values", reframed)

# split into train and test sets
values = reframed.values
n_train_hours = 320 * 24 # 267 days= 74:26  , 292 * 24 = 8:2 ratio % train Test Ratio
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
print("train after number of hours", train)
print("test after number of hours", test)
print("train shape after number of hours", train.shape)
print("test shape after number of hours", test.shape)


# split into input and outputs
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
print('train_X.shape, len(train_X), train_y.shape')
print(train_X.shape, len(train_X), train_y.shape)








# setup regressor
xgb_model = xgb.XGBRegressor(objective = 'reg:squarederror', eval_metric = 'mae',   learning_rate=0.2)

# perform a grid search
tweaked_model = GridSearchCV(
    xgb_model,
    { 'n_estimators': [50, 70, 100, 200,500],
      'max_depth': [1,2,3,5,7,9,13]
    },
    cv=10,
    verbose=1,
    n_jobs=-1,
    scoring='neg_median_absolute_error'
)

tweaked_model.fit(train_X, train_y)

# summarize results
print("Best: %f using %s" % (tweaked_model.best_score_, tweaked_model.best_params_))





xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 1, learning_rate = 0.2, gamma=0, random_state=10, seed=10,
                max_depth = 2, alpha = 10, n_estimators = 70, eval_metric = 'mae')

modelstart= time.time()


xg_reg.fit(train_X, train_y)
print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
print(xg_reg)
preds = xg_reg.predict(train_X)

test_preds = xg_reg.predict(test_X)

trmse = np.sqrt(mean_squared_error(train_y, preds))
# mae = mean_absolute_error(train_y, preds)
# R2 = r2_score(train_y, preds)

print("XGB Train MAE: %.4f" % mean_absolute_error(train_y, preds))
print("XGBoost Train RMSE: %f" % (trmse))
print( "XGBoost Train r2 Score", r2_score(train_y, preds))
print("XGB Train MAPE = ", mean_absolute_percentage_error(train_y, preds))
print("XGB Train MPE = ", MPE(train_y, preds))

r2 = r2_score(test_y, test_preds)
rmse = np.sqrt(mean_squared_error(test_y, test_preds))
mae = mean_absolute_error(test_y, test_preds)
Mape = mean_absolute_percentage_error(test_y, test_preds)
mpe = MPE(test_y, test_preds)
IOA = index_agreement(test_y, test_preds)
r = pearsonr(test_y, test_preds)

print("Model9: XGBOOST Model :")
print("XGB Test MAE: %.5f" % mean_absolute_error(test_y, test_preds))
print("XGBoost Test RMSE: %.5f" % np.sqrt(mean_squared_error(test_y, test_preds)))
print("   ")
print("Test IOA %.5f" %IOA)
print("Test Pearson: %.5f "%r[0])
print( "XGB Test r2 Score = %.5f"  %r2)
print("XGB Test MAPE = %.5f" %Mape)
print("XGB Test MPE = %.5f " %mpe)



#XGBModel = '/home/khaula/Desktop/PM10_LSTM_From_Jan1/PM25Results/XGB_LGBM/TS_PM25XGB_Feb12.pkl'
#joblib.dump(xg_reg, XGBModel)
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))

plt.figure(figsize=(20,10))
plt.plot(test_y, 'b', label='Actual')
plt.plot(test_preds, 'r', label='Prediction')
plt.title("Konkuk Dong XGB R= %.4f RMSE= %.4f MAE=%.4f " % (r[0], rmse, mae))
plt.ylabel('PM 10 Concentration')
plt.xlabel('Time Step')
plt.legend()
#plt.savefig('/home/khaula/Desktop/PM10_LSTM_From_Jan1/PM25Results/XGB_LGBM/PM10TS_Results/PM10XGB_Konkuk_Feb14.png')
plt.show()

'''










'''
Results:

Train Data information
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2390491 entries, 0 to 2390490
Columns: 138 entries, date to ts_id
dtypes: float64(135), int64(3)
memory usage: 2.5 GB
Test Data information
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 15219 entries, 0 to 15218
Columns: 133 entries, weight to ts_id
dtypes: float64(130), int64(3)
memory usage: 15.4 MB




'''
