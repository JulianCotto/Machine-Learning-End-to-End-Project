####################################################################
# Developer: Julian Cotto
# Date: 9/15/2023
# File Name: main.py
# Description: This is the main running file for the project.
####################################################################
import sys
import urllib
from urllib import request

import sklearn
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from pathlib import Path
from loadData import load_housing_data
from packaging import version
from sklearn.model_selection import train_test_split
from saveFigs import save_fig

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
assert sys.version_info >= (3, 7)

# extra code – code to save the figures as high-res PNGs for the book
IMAGES_PATH = Path() / "images" / "end_to_end_project"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)


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

    # create a copy of the stratified train set
    housing = strat_train_set.copy()

    # create visualisation of the data
    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True)
    save_fig("bad_visualization_plot")  # extra code
    plt.show()

    # create visualisation of the data with alpha set to 0.2 to see high density areas
    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
    save_fig("better_visualization_plot")  # extra code
    plt.show()

    # create visualisation of the data with radius of each circle representing the district's population
    # color represents the price
    # use a predefined color map (option cmap) called jet, which ranges from blue (low values) to red (high prices)
    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
                 s=housing["population"] / 100, label="population",
                 c="median_house_value", cmap="jet", colorbar=True,
                 legend=True, sharex=False, figsize=(10, 7))
    save_fig("housing_prices_scatterplot")  # extra code
    plt.show()

    # extra code – this cell generates the first figure in the chapter

    # Generate and Download a beautified version of the above image
    filename = "california.png"
    if not (IMAGES_PATH / filename).is_file():
        homl3_root = "https://github.com/ageron/handson-ml3/raw/main/"
        url = homl3_root + "images/end_to_end_project/" + filename
        print("Downloading", filename)
        urllib.request.urlretrieve(url, IMAGES_PATH / filename)

    housing_renamed = housing.rename(columns={
        "latitude": "Latitude", "longitude": "Longitude",
        "population": "Population",
        "median_house_value": "Median house value (ᴜsᴅ)"})
    housing_renamed.plot(
        kind="scatter", x="Longitude", y="Latitude",
        s=housing_renamed["Population"] / 100, label="Population",
        c="Median house value (ᴜsᴅ)", cmap="jet", colorbar=True,
        legend=True, sharex=False, figsize=(10, 7))

    california_img = plt.imread(IMAGES_PATH / filename)
    axis = -124.55, -113.95, 32.45, 42.05
    plt.axis(axis)
    plt.imshow(california_img, extent=axis)

    save_fig("california_housing_prices_plot")
    plt.show()

    # check for correlations between attributes using scatter_matrix() method
    # choosing some important attributes to reduce number of plots in figure
    attributes = ['median_house_value','median_income','total_rooms','housing_median_age']
    scatter_matrix(housing[attributes],figsize=(15,12))
    save_fig("scatter_matrix_plot")  # extra code
    plt.show()

    housing.plot(kind="scatter", x="median_income", y="median_house_value",
                 alpha=0.1, grid=True)
    save_fig("income_vs_house_value_scatterplot")  # extra code
    plt.show()

    housing.plot(kind="scatter", x="median_income", y="median_house_value",
                 alpha=0.1, grid=True)
    plt.show()


if __name__ == '__main__':
    main()
