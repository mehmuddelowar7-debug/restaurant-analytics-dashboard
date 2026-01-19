import pandas as pd
import numpy as np

# Load datasets
order_history = pd.read_csv("order_history_kaggle_data.csv")
indian_restaurants = pd.read_csv("indian_restaurants.csv")
swiggy = pd.read_csv("swiggy.csv")

# Quick look at each dataset
print("Order History Dataset:")
print(order_history.head())

print("\nIndian Restaurants Dataset:")
print(indian_restaurants.head())

print("\nSwiggy Dataset:")
print(swiggy.head())

# Check basic info
print("\nOrder History Info:")
print(order_history.info())

print("\nIndian Restaurants Info:")
print(indian_restaurants.info())

print("\nSwiggy Info:")
print(swiggy.info())
