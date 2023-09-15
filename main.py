####################################################################
# Developer: Julian Cotto
# Date: 9/14/2023
# File Name: main.py
# Description: This is the main running file for the project.
####################################################################
import sys
import matplotlib.pyplot as plt
import sklearn
from packaging import version
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from loadData import load_housing_data
from saveFigs import save_fig

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
assert sys.version_info >= (3, 7)


def main() -> None:
    print("Welcome to Machine Learning!")

    # load housing data csv and create directory if it doesn't exist
    housing = load_housing_data()

    # extra code – the next 5 lines define the default font sizes
    plt.rc('font', size=14)
    plt.rc('axes', labelsize=14, titlesize=14)
    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    housing.hist(bins=50, figsize=(12, 8))
    save_fig("attribute_histogram_plots")  # extra code
    plt.show()

    ##############################################################
    #  generating test set and train set using random sampling   #
    # see addlMainMethods.py for more methods to accomplish this #
    ##############################################################

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    ###############################################################
    # generating test set and train set using stratified sampling #
    # see addlMainMethods.py for more methods to accomplish this  #
    ###############################################################

    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])

    strat_train_set, strat_test_set = train_test_split(
        housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)


if __name__ == '__main__':
    main()
