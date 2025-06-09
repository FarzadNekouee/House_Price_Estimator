import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

def train_and_save_model():
    # dataset link: https://github.com/shridhar1504/Boston-House-Price-Prediction-Datascience-Project/blob/main/Datasets/Boston%20House%20Price%20Data.csv
    data = pd.read_csv('Boston_House_Price_Data.csv')
    X = data.drop(columns=["PRICE"]) # eliminate target column
    y = data['PRICE']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)
    
    print('pickle file saved!')

if __name__ == '__main__':
    train_and_save_model()