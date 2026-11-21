import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

## load the dataset

data=pd.read_csv("Churn_Modelling.csv")
data.head()
print(data.info())