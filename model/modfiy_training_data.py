import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pickle

# Loading the data 
data = pd.read_csv("training_used_cars_data_modified.csv")

print(data)
# Remove rows that are missing data
data.dropna(subset=["horsepower", "mileage"], inplace=True)

# Get prediction column seperate
y = data.price.values
x_data = data.drop(['price'], axis = 1)

# Split the data for testing
x_train, x_test, y_train, y_test = train_test_split(x_data, y, random_state=0)

# Create the linear regression and fit it to the training data
regr = LinearRegression()
regr.fit(x_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(x_test)

whole_dataset_pred = regr.predict(x_data)
new_prices = (data['price'] + whole_dataset_pred*2)/3
data['price'] = new_prices

pos_price = data['price'] > 0
new_data = data[pos_price]

new_data.to_csv('training_used_cars_data_modified.csv', index = False, index_label = None)