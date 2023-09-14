####################################################################
# Developer: Julian Cotto
# Date: 9/14/2023
# File Name: main.py
# Description: This is the main running file for the project.
####################################################################

# project requirements setup
import sys
import matplotlib.pyplot as plt
import sklearn
from packaging import version

from loadData import load_housing_data
from saveFigs import save_fig

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
assert sys.version_info >= (3, 7)

print("Welcome to Machine Learning!")

# load housing data csv and create directory if it doesn't exist
housing = load_housing_data()

# return top 5 rows of housing data. print statement for testing purposes
# longitude
# latitude
# housing_median_age
# total_rooms
# total_bedrooms
# population
# households
# median_income
# median_house_value
# ocean_proximity
print(housing.head())

# return housing data structure info.
print(housing.info())

# return value counts for ocean_proximity column
print(housing["ocean_proximity"].value_counts())

# return a description of the dataset as a whole
print(housing.describe())

# extra code – the next 5 lines define the default font sizes
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

housing.hist(bins=50, figsize=(12, 8))
save_fig("attribute_histogram_plots")  # extra code
plt.show()
