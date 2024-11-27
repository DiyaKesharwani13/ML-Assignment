# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Convert the dataset into a pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Section 1: Display the first five rows of the dataset
print("Section 1: First five rows of the dataset")
print(df.head())

# Section 2: Show the dataset's shape and summary statistics
print("\nSection 2: Dataset shape and summary statistics")
print(f"Dataset shape: {df.shape}")
print("Summary statistics:")
print(df.describe())
