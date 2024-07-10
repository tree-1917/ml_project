# Home Price Prediction using Linear Regression 🏠💰

This project implements a machine learning model to predict home prices based on features such as number of rooms, home size, and distance from the sea using Linear Regression.

## Project Structure 📂

```
app/
├── ML_model.py     # Machine learning model implementation
├── main.py         # Main application entry point
└── schema.py       # Schema definition for input data
```

## Description ℹ️

This project uses a simple Linear Regression model to predict home prices. The model is trained on a sample dataset with features and corresponding home prices.

### Sample Data

```python
# Sample data (features)
X = np.array([[3, 1500, 5],
              [4, 2000, 3],
              [2, 1000, 8],
              [5, 2500, 2],
              [3, 1800, 4]])

# Target variable (home prices)
y = np.array([300000, 400000, 250000, 450000, 350000])
```

### Model Training

The `ML_model.py` file contains the implementation of the Linear Regression model using `scikit-learn`.

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Create and fit the model
model = LinearRegression()
model.fit(X, y)
```

### Prediction Function

The model predicts home prices based on the input features.

```python
def predict_home_price(num_rooms, home_size, distance_from_sea):
    # Prepare input data for prediction
    input_features = np.array([[num_rooms, home_size, distance_from_sea]])
    
    # Predict home price
    predicted_price = model.predict(input_features)
    return predicted_price[0]
```

## Usage 🚀

To use the prediction function, call `predict_home_price` with the desired input values.

```python
predicted_price = predict_home_price(3, 2000, 4)
print(f"Predicted home price: ${predicted_price:.2f}")
```

## Requirements 🛠️

Make sure you have the following libraries installed:

```bash
pip install numpy scikit-learn
```

## Contributing 🤝

Contributions are welcome! Fork the repository and submit pull requests to improve the project.

## License 📄

This project is licensed under the MIT License. See the LICENSE file for more details.

---
