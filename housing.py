import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tables import Column
from statistics import linear_regression
from typing import final
from unittest import result


housing = pd.read_csv('/home/hamid/Documents/python/data-science/housing.csv' ,header= 0, sep= ',')
df = housing.copy()


#------------------------------------------------------Pandas-------------------------------------------------------

#print(df.shape)
#print(df.info())
#print(df.describe())
#print(df.isnull().sum())
#print(df.columns)

#-----------------------------------------------------plot map area city---------------------------------------------

#df.plot(kind='scatter', x='longitude', y='latitude', figsize=(10, 7), alpha=0.2, s=df['population']/25, label='population',
 #c=df['median_house_value'] , cmap=plt.get_cmap('jet') ) 
 #plt.show()

 # or

#plt.scatter(x=housing['longitude'], y=housing['latitude'])
#plt.show()

#-----------------------------------------------------correlation matrix---------------------------------------------

#corr_matrix = df.corr()                                  #faghat khati
#print(corr_matrix)
#print(corr_matrix['median_house_value'].sort_values(ascending=False))   #rabeteye gheymete khane ba sayer
#sns.heatmap(corr_matrix)
#plt.show()

#-----------------------------------------------------scatter matrix--------------------------------------------------

#scatter_matrix(df, figsize=(15, 10))                    #full chart 2
#plt.show()

#df.plot(kind='scatter', x='median_income', y='median_house_value', figsize=(10,7), alpha=0.4) #hala faghat nemodari ra mikeshim ke hambastegie darad hamchenin price cap darim dar 500000
#plt.show()

#====================================================(prepare data)==================================================

#----------------------numerical data(missing value)--------------------------------------------------------------------
#----------------------numerical data to custom transformers------------------------------------------------------------
#----------------------numerical data to feature scaling----------------------------------------------------------------
#----------------------categorical and text data to labelencoder , onehoteencoder --------------------------------------


#----------------- جدا سازی داده های عددی از اسمی و جداسازی تارگت یا همان لیبل -----------

df_lable = df['median_house_value'].copy()               # y_test or target
df = df.drop('median_house_value', axis= 1)              # garare pishbini beshe sepass copy pas hazf mikonimesh
df_cat = df['ocean_proximity'].copy()
df_num = df.drop('ocean_proximity', axis = 1)            # jense dade fargh darad va onject bood pas hazf

#-----------------------------------------------------missing dist matrix---------------------------------------------

msno.matrix(df)
plt.show()

#-------method 1 for missing data---------------------

#df_num.drop('total_bedrooms', axis = 1)                 #A  hazfe kole sootoon
#df_num = df_num.dropna(subset = ['total_bedrooms'])     #B  hazfe khali ha
#median = df_num['total_bedrooms'].median()              #C  mianeh
#df_num['total_bedrooms'].fillna(median)

#------method 2 simple imputer for missing data-------

imputer = SimpleImputer(missing_values = np.nan, strategy='median')
imputer.fit(df_num)                                      #   amoozeshesh bede va median ra baraye hame sootoonha hesab mikonad
X = imputer.transform(df_num)                            #   taghireshekl(az jense numpy) ra baraye hame sootoonha mizand
df_num_impute_tr = pd.DataFrame(X, columns = df_num.columns )

#df_num_impute_tr.isnull().sum()


#-------------------custom transformers -------------

rooms_ix, bedrooms_ix, population_ix, houshols_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, houshols_ix]    
        population_per_household = X[:, population_ix] / X[:, houshols_ix]
        bedroom_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household, bedroom_per_room]

custom = CombinedAttributesAdder()
data_custom_tr_tmp = custom.transform(df_num_impute_tr.values)
data_custom_tr = pd.DataFrame(data_custom_tr_tmp)
columns = list(df_num_impute_tr.columns)
columns.append("rooms_per_household")
columns.append("population_per_household")        
columns.append("bedrooms_per_room")
data_custom_tr.columns = columns
#print(data_custom_tr.head(10))

#-------------------------------------------feature scaling استاندارد سازی داده ها------------------------------------------

# ------standardization---or--normalization [0,1] نرمالیزیشن ایراد در اوت لایرها دارد
#normalized, standard, minmax, robust

feature_scal = StandardScaler() 
q = feature_scal.fit_transform(data_custom_tr.values) # values پرانتز
data_num_scaled_tr = pd.DataFrame(q, columns= data_custom_tr.columns)
#print(data_num_scaled_tr.head())


#-------------------------------------------label encoding-------------------------------------------------------------------
# روش خوبی نیست چون ماشین بین ۰ و ۱ فکر می کند ارتباطی وجود دارد
# فرضا بد متوسط خوب خیلی خوب و عالی بود این روش خیلی خوب بود جون بین این طیف رابطه وجود دارد
# ولی وقتی متغیرهای ۰ و۱ و ۲ و غیره زیادی داریم روش خوبی است
'''
encoder = LabelEncoder()
data_cat_encoded = encoder.fit_transform(data_cat)
data_cat_encoded = pd.DataFrame(data_cat_encoded, columns= ['ocean_proximity'])
print(data_cat_encoded.head())
'''
#--------------------------------------------one hot encoding---------------------------------------------------------------
# این روش برای حالتی بهتر است که متغیر ۰ و ۱ زیادی نداشته باشیم

encoder_1hot = OneHotEncoder(sparse= False) # آپشن اسپارس برای حالتی است که داده های نال را در نظر نگیرد
data_cat_1hot_tmp = encoder_1hot.fit_transform(df[['ocean_proximity']]) #df_cat_1hot = np.array(df_cat).reshape((len(df_cat), 1))
data_cat_1hot = pd.DataFrame(data_cat_1hot_tmp)
data_cat_1hot.columns = encoder_1hot.get_feature_names(['prox'])

final = pd.concat( [data_num_scaled_tr, data_cat_1hot], axis= 1 ) #حال باید داده های عددی را به داده های اسمی بچسبانیم ضمنا کوتیشن ندارد
print(final.head())


#---------------------------------------------pipeline-----------------------------------------------------------------------
'''
class dataframeselector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names= attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values       
        


num_pipeline = Pipeline([
    ('selector', dataframeselector(list(df_num))),
    ('imputer', SimpleImputer(missing_values = np.nan, strategy='median')),
    ('attribotadder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
    ('selector', dataframeselector(['ocean_proximity'])),
    ('One_Hot_Encoder', OneHotEncoder(sparse= False)),

])


# full_pipeline is 

full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline ),
    ('cat_pipeline', cat_pipeline),

])

housing_prepared = full_pipeline.fit_transform(df)

housing_prepared = pd.DataFrame(housing_prepared, columns= final.columns)
print(housing_prepared.head())


'''
#----------------------------------------------------split data---------------------------------------------------------------

# y_predict = y_model & y_predict vs y_test
X_train, X_test, y_train, y_test = train_test_split(final, df_lable, test_size= 0.50, random_state= 42) #khoroji tuple do doeye

print(X_test.head())

#-------------------------------------------pca (principles component anelysis)-------------------------------------------------------------------
'''

from sklearn.decomposition import PCA   # 1. Choose the model class

model = PCA(n_components=2)             # 2. Instantiate the model with hyperparameters
model.fit(X_train)                      # 3. Fit to data. Notice y is not specified!
X_2D = model.transform(X_train)  

final['PCA1'] = X_2D[:, 0]
final['PCA2'] = X_2D[:, 1]

sns.lmplot("PCA1", "PCA2", data=housing, hue='median_house_value', col='y_predict', fit_reg=False)


'''

#----------------------------------------------------model selection----------------------------------------------------------


# ------------------ RandomForestRegressor & grids search cv------------------

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

param_grid = [ {'n_estimators': [3, 4, 6, 10, 30], 'max_features': [2, 6, 8, 15] } ]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv= 5, scoring= 'neg_mean_squared_error') # cv=5 یعنی پنج بار این کار را انجام بده

grid_search.fit(X_train, y_train)

print('best parameter:' , grid_search.best_params_)
print('best estimator:' , grid_search.best_estimator_)
# answer is:
# best parameter: {'max_features': 6, 'n_estimators': 30}
# best estimator: RandomForestRegressor(max_features=6, n_estimators=30)


#-----------------برای مشاهده بقیه اسکورها--------------------

#results = grid_search.cv_results_

#for mean_score, params in zip(results['mean_test_score'], results['params']):
 #   print(np.sqrt(-mean_score), params)



#========================================================Final-Model=============================================================

final_model = grid_search.best_estimator_
# X_test = full_pipeline.transform(X2)
final_pricitions = final_model.predict(X_test)
final_mse = mean_squared_error(final_pricitions, y_test)    #(y_predict, y_test)
final_rmse = np.sqrt(final_mse)
print(final_rmse)

