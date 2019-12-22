# --------------
# Code starts here
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

from sklearn.preprocessing import Imputer
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder



#### Data 1
#1 Load the data
df = pd.read_csv(path)
#df.head(2)

#2 Overview of the data
#df.info()
#df.describe()

#3 Histogram showing distribution of car prices
#plt.figure()
#sns.distplot(df['price'], kde= True, rug=True)

#4 Countplot of the make column
#df['make'].value_counts()
#plt.figure()
#sns.countplot(y='make', data = df)

#5 Jointplot showing relationship between 'horsepower' and 'price' of the car
#plt.figure(figsize = (10,10))
#sns.jointplot(x="horsepower", y="price", data=df, kind="scatter")
#sns.jointplot(x="horsepower", y="price", data=df, kind="reg")

#6 Correlation heat map
#plt.figure(figsize = (15,15))
#sns.heatmap(df.corr(), cmap="YlGnBu")

#7 boxplot that shows the variability of each 'body-style' with respect to the 'price'
#plt.figure(figsize = (12,10))
#sns.boxplot(x="body-style", y="price", data=df)

#### Data 2

# Load the data
df_2 = pd.read_csv(path2)
#df_2.head(2)
#print(df_2.shape)
#print(df_2.columns)
#print(df_2.info)

# Impute missing values with mean
df_2_new = df_2.replace('?', "NaN")
df_2_new.head(2)
mean_imputer = Imputer(missing_values= "NaN", strategy = 'mean')

df_2_new[['normalized-losses']] = mean_imputer.fit_transform(df_2_new[['normalized-losses']])
df_2_new[['horsepower']] = mean_imputer.fit_transform(df_2_new[['horsepower']])
#df_2_new.head(10)
#df_2_new.isna().sum()

# Skewness of numeric features

numeric_columns  = df_2_new._get_numeric_data().columns
#print(numeric_columns)
for i in numeric_columns:
    if skew(df_2_new[i]) > 1:
        df_2_new[i] = np.sqrt(df_2_new[i])
    
# Label encode 
print(df_2_new.dtypes)
categorical_columns = df_2_new.select_dtypes(include='object').columns
print(categorical_columns)
encoder = LabelEncoder()

print(df_2_new[categorical_columns].head(5))

for i in categorical_columns:
    df_2_new[i] = encoder.fit_transform(df_2_new[i])

print(df_2_new[categorical_columns].head(5))

#5 Combine 2 colums and create 1 column
df_2_new['area'] = df_2_new['height'] * df_2_new['height']

df_2_new.head(5)



# Code ends here


