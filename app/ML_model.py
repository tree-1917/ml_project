# Import necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data (you can replace it with your own dataset)
# Features: Number of rooms, Size of the home, Distance from the sea
X = np.array([[3, 1500, 5],
              [4, 2000, 3],
              [2, 1000, 8],
              [5, 2500, 2],
              [3, 1800, 4]])

# Target variable: Home price
y = np.array([300000, 400000, 250000, 450000, 350000])

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Function to predict home price based on input features
def predict_home_price(num_rooms, home_size, distance_from_sea):
    # Reshape input into 2D array as expected by the model
    input_features = np.array([[num_rooms, home_size, distance_from_sea]])
    # Predict home price
    predicted_price = model.predict(input_features)
    return predicted_price[0]