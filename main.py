import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import warnings
warnings.filterwarnings(action='ignore')

data = pd.read_csv('Vegetable_market.csv')

#This will show the provided data:
print("\nTHIS IS THE PROVIDED VEGETABLE DATA:\n")
print(data)

#This will show the information about data:
#print(data.info())

#PREPROCESSING:
def onehot_encode(df, column):
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(column, axis=1)
    return df

def preprocess_inputs(df):
    df = df.copy()
    
    # Clean Vegetable condition column
    df['Vegetable condition'] = df['Vegetable condition'].replace({'scarp': 'scrap'})
    
    # Binary encoding
    df['Deasaster Happen in last 3month'] = df['Deasaster Happen in last 3month'].replace({'no': 0, 'yes': 1})
    
    # Ordinal encoding
    df['Month'] = df['Month'].replace({
        'jan': 1,
        'apr': 4,
        'july': 7,
        'sept': 9,
        'oct': 10,
        'dec': 12,
        'may': 5,
        'aug': 8,
        'june': 6,
        ' ': np.NaN,
        'march': 3
    })
    
    # Fill missing month values with column mode
    df['Month'] = df['Month'].fillna(df['Month'].mode()[0])
    
    # One-hot encoding
    for column in ['Vegetable', 'Season', 'Vegetable condition']:
        df = onehot_encode(df, column)
    
    # Split df into X and y
    y = df['Price per kg']
    X = df.drop('Price per kg', axis=1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)
    
    # Scale X
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_inputs(data)

#print(X_train)

#print(y_train)

#TESTING:
print("\n\nTRAINING THE MODELS: ")
models = {
    "                     Linear Regression": LinearRegression(),
#   "                   K-Nearest Neighbors": KNeighborsRegressor(),
    "                         Decision Tree": DecisionTreeRegressor(),
    "                         Random Forest": RandomForestRegressor(),
    "                     Gradient Boosting": GradientBoostingRegressor()
}
for name, model in models.items():
    model.fit(X_train, y_train)
    print(name + " model trained.")

#RESULT:
print("\n\nRESULTS OF TRAINED MODEL: ")
for name, model in models.items():
    print(name + " Score: {:.5f}".format(model.score(X_test, y_test)))
