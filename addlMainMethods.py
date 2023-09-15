####################################################################
# Developer: Julian Cotto
# Date: 9/14/2023
# File Name: functions.py
# Description: This file contains additional code covered,
#              but not used, by the project
####################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from loadData import load_housing_data
from addlTestSetMethods import shuffle_and_split_data
from addlTestSetMethods import split_data_with_id_hash
from addlTestSetMethods import income_cat_proportions
from saveFigs import save_fig

housing = load_housing_data()

print("Housing Methods")
# return top 5 rows of housing data. print statement for testing purposes
print(housing.head())

# return housing data structure info.
print(housing.info())

# return value counts for ocean_proximity column
print(housing["ocean_proximity"].value_counts())

# return a description of the dataset as a whole
print(housing.describe())

print("Test/Train Set Creation Method 1: Random Sampling of 20% of Data")
train_set_0, test_set_0 = shuffle_and_split_data(housing, 0.2)
print(len(train_set_0))
print(len(test_set_0))
print("The above can be accomlished with sklearn's train_test_split method as well.")

print(test_set_0["total_bedrooms"].isnull().sum())

print("Test/Train Set Creation Method 2: Add an ID Column, Hash the ID and Split the Data")
housing_with_id = housing.reset_index()  # adds an `index` column
train_set_1, test_set_1 = split_data_with_id_hash(housing_with_id, 0.2, "index")
print(len(train_set_1))
print(len(test_set_1))

print("Test/Train Set Creation Method 3: Use Latitude and Longitude to Create an ID")
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set_2, test_set_2 = split_data_with_id_hash(housing_with_id, 0.2, "id")
print(len(train_set_2))
print(len(test_set_2))

print("Extra code – shows how to compute the 10.7% proba of getting a bad sample")
sample_size = 1000
ratio_female = 0.511
proba_too_small = binom(sample_size, ratio_female).cdf(485 - 1)
proba_too_large = 1 - binom(sample_size, ratio_female).cdf(535)
print(proba_too_small + proba_too_large)

print("If you prefer simulations to maths, here's how you could get roughly the same result:")
sample_size = 1000
ratio_female = 0.511
samples = (np.random.rand(100_000, sample_size) < ratio_female).sum(axis=1)
print(((samples < 485) | (samples > 535)).mean())

print("Stratified Sampling For Each Set of Data")
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

print("the below code renders a visual representation of the income categories generated above")
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
save_fig("housing_income_cat_bar_plot")  # extra code
plt.show()

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

strat_train_set, strat_test_set = strat_splits[0]

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall %": income_cat_proportions(housing),
    "Stratified %": income_cat_proportions(strat_test_set),
    "Random %": income_cat_proportions(test_set),
}).sort_index()
compare_props.index.name = "Income Category"
compare_props["Strat. Error %"] = (compare_props["Stratified %"] /
                                   compare_props["Overall %"] - 1)
compare_props["Rand. Error %"] = (compare_props["Random %"] /
                                  compare_props["Overall %"] - 1)
(compare_props * 100).round(2)

print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

print(strat_train_set)
print(strat_test_set)