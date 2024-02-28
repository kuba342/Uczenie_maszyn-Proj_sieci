import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import csv
import torch

def make_dataset():
        # Load the original dataset
    test_dataset = torch.load('test_dataset_big.pt')

    # Create lists to store labels and densities
    labels = []
    densities = []
    average_clustering = []

    # Extract labels and densities from each graph in the original dataset
    for graph in test_dataset:
        labels.append(graph.y[0].item())
        densities.append(graph.global_feature[0][0].item())
        average_clustering.append(graph.global_feature[0][1].item())


    # Create a new dataset with labels and densities
    test_dataset_dens = list(zip(labels, densities))
    test_dataset_clust = list(zip(labels, average_clustering))
    test_dataset_comb = list(zip(labels, densities, average_clustering))


    with open('test_dataset_dens.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['label', 'density'])  # Write header
        writer.writerows(test_dataset_dens)         # Write data
    with open('test_dataset_clust.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['label', 'average_clustering'])  # Write header
        writer.writerows(test_dataset_clust)         # Write data
    with open('test_dataset_comb.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['label', 'density', 'average_clustering'])  # Write header
        writer.writerows(test_dataset_comb)         # Write data

def cell_1():
    # Load the dataset from CSV
    df = pd.read_csv('train_dataset_dens.csv')

    # Separate features (densities) and target (labels)
    X = df['density'].values.reshape(-1, 1)
    y = df['label'].values
    
    return X, y

def cell_2(X, y):
    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    print(f"Linear Regression Model, MSE: {mse}")

def cell_3(X, y):
    # Create and train the decision tree regression model
    model = DecisionTreeRegressor()
    model.fit(X, y)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    print(f"Decision Tree Regression Model, MSE: {mse}")

def cell_4(X, y):
    # Create and train the random forest regression model
    model = RandomForestRegressor()
    model.fit(X, y)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    print(f"Random Forest Regression Model, MSE: {mse}")

def cell_5(X, y):
    # Create and train the SVR model
    model = SVR()
    model.fit(X, y)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    print(f"Support Vector Regression Model, MSE: {mse}")

def cell_6(X, y):
    # Create and train the k-nearest neighbors regression model
    model = KNeighborsRegressor()
    model.fit(X, y)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    print(f"K-Nearest Neighbors Regression Model, MSE: {mse}")

def main():
    X, y = cell_1()
    cell_2(X, y)
    cell_3(X, y)
    cell_4(X, y)
    cell_5(X, y)
    cell_6(X, y)

if __name__ == "__main__":
    main()