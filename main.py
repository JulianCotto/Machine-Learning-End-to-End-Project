####################################################################
# Developer: Julian Cotto
# Date: 9/14/2023
# File Name: main.py
# Description: This is the main running file for the project.
####################################################################
import sys
import sklearn
import numpy as np
import pandas as pd
from loadData import load_housing_data
from packaging import version
from sklearn.model_selection import train_test_split

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
assert sys.version_info >= (3, 7)


def main() -> None:
    print("Welcome to Machine Learning!")

    # load housing data csv and create directory if it doesn't exist
    housing = load_housing_data()

    ##############################################################
    #  generating test set and train set using random sampling   #
    # see addlMainMethods.py for more methods to accomplish this #
    ##############################################################

    # create a test set and train set using random sampling
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    ###############################################################
    # generating test set and train set using stratified sampling #
    # see addlMainMethods.py for more methods to accomplish this  #
    ###############################################################

    # create an income category attribute with 5 categories
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])

    # create a stratified test set and train set based on the income category
    strat_train_set, strat_test_set = train_test_split(
        housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

    # drop the income_cat attribute so the data is back to its original state
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    print("Length of Test Set:", len(test_set))
    print("Length of Test Set:", len(train_set))

    print("Length of Stratified Test Set:", len(strat_test_set))
    print("Length of Stratified Test Set:", len(strat_train_set))


if __name__ == '__main__':
    main()
