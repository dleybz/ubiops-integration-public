import datetime
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from whylogs.app.writers import WhyLabsWriter
from whylogs.app import Session
from whylogs.app.session import get_or_create_session
import pickle

# Loading the data 
data = pd.read_csv("model/training_used_cars_data_modified.csv")

#profile data and write to WhyLabs
today = datetime.datetime.now()
yesterday = today - datetime.timedelta(days=1)

writer = WhyLabsWriter("", formats=[],)
session = Session(project="demo-project", pipeline="pipeline-id", writers=[writer])
with session.logger(dataset_timestamp=yesterday) as ylog:
    ylog.log_dataframe(data)

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

# The coefficients
print(f'Coefficients: \n{regr.coef_}')
# The mean squared error
print(f'Mean squared error: {mean_squared_error(y_test, y_pred)}')
# The coefficient of determination: 1 is perfect prediction
print(f'Coefficient of determination: {r2_score(y_test, y_pred)}')

# Save the built model to our dployment folder
with open('deployment_folder/model.pkl', 'wb') as f:
    pickle.dump(regr, f)
