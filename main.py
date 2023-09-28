####################################################################
# Developer: Julian Cotto
# Date: 9/15/2023
# File Name: main.py
# Description: This is the main running file for the project.
####################################################################
import sys
from pathlib import Path
from packaging import version

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import RandomizedSearchCV

import numpy as np
import pandas as pd
from scipy.stats import randint

from functions import monkey_patch_get_signature_names_out
from functions import ratio_pipeline
from classes import StandardScalerClone
from classes import ClusterSimilarity
from loadData import load_housing_data
from saveFigs import getFig1
from saveFigs import getFig2
from saveFigs import getFig3
from saveFigs import getBeautyFig
from saveFigs import getScatterFigs
from saveFigs import getFig4
from saveFigs import getFig5
from saveFigs import getFig6
from saveFigs import getFig7
from saveFigs import getFig8
from saveFigs import getFig9
from saveFigs import getFig10
from saveFigs import getFig11

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
assert sys.version_info >= (3, 7)

# extra code – code to save the figures as high-res PNGs for the book
IMAGES_PATH = Path() / "images" / "end_to_end_project"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("Welcome to Machine Learning!")

    # load housing data csv and create directory if it doesn't exist
    print("Loading Housing Data")
    housing = load_housing_data()
    print("Housing Data Loaded")

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
    print("Creating Income Category")
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])
    print("Income Category Created")

    # create a stratified test set and train set based on the income category
    print("Creating Stratified Test/Train Set")
    strat_train_set, strat_test_set = train_test_split(
        housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)
    print("Stratified Test/Train Set Created")

    # drop the income_cat attribute so the data is back to its original state
    print("Dropping Income Category")
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    print("Income Category Dropped")

    # create a copy of the stratified train set
    print("Creating Copy of Stratified Train Set")
    housing = strat_train_set.copy()
    print("Stratified Train Set Copied")

    print("Generating Figs")
    # create visualisation of the data
    getFig1(housing)

    # create visualisation of the data with alpha set to 0.2 to see high density areas
    getFig2(housing)

    # create visualisation of the data with radius of each circle representing the district's population
    # color represents the price
    # use a predefined color map (option cmap) called jet, which ranges from blue (low values) to red (high prices)
    getFig3(housing)

    # Generate and Download a beautified version of the above image
    getBeautyFig(housing)
    print("Figs Generated")

    # check for correlations using the corr() method
    print("Checking for Data Correlations")
    attributes = ['median_house_value',
                  'median_income',
                  'total_rooms',
                  'housing_median_age',
                  'households',
                  'total_bedrooms',
                  'population',
                  'longitude',
                  'latitude']
    corr_matrix = housing[attributes].corr()
    corr_matrix['median_house_value'].sort_values(ascending=False)
    print("Data Checks Complete")

    # visualize correlations using the scatter_matrix() function
    print("Generating Figs")
    getScatterFigs(housing, attributes)
    getFig4(housing)
    getFig5(housing)
    print("Figs Generated")

    # create new attributes to see if they are more correlated with median house value
    print("Searching for Additional Correlations")
    housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["people_per_house"] = housing["population"] / housing["households"]
    attributes = ['median_house_value',
                  'median_income',
                  'rooms_per_house',
                  'total_rooms',
                  'housing_median_age',
                  'households',
                  'total_bedrooms',
                  'population',
                  'people_per_house',
                  'longitude',
                  'latitude',
                  'bedrooms_ratio']
    corr_matrix = housing[attributes].corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)
    print("Search Completed")

    # create a copy of the stratified train set
    print("Reverting to Clean Training Set")
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    print("Reverted to Clean Training Set")

    # Set the missing values to some value (zero, the mean, the median, etc.). This is called imputation.
    print("Creating Imputer Instance")
    imputer = SimpleImputer(strategy="median")
    print("Imputer Instance Created")

    # create a copy of the data with only the numerical attributes
    print("Copying Numerical Data in Dataset")
    housing_num = housing.select_dtypes(include=[np.number])
    print("Numerical Dataset Copied")

    # fit the imputer instance to the training data using the fit() method
    print("Fitting Imputer to Training Data")
    imputer.fit(housing_num)
    print("Imputer Fitted to Training Data")

    # transform the training set by replacing missing values with the learned medians
    print("Using Imputer to Replace Missing Values with Learned Median")
    X = imputer.transform(housing_num)
    print("Missing Values Replaced with Learned Median")

    # convert the NumPy array into a Pandas DataFrame
    print("Convert Array to Datafram for Column Name & Index Recovery")
    housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                              index=housing_num.index)
    print("Column Names & Indexes Recovered")

    # visualize text attributes
    print("Seperating Text Data")
    housing_cat = housing[["ocean_proximity"]]
    print("Text Data Seperated")

    # convert categorical values into one-hot vectors using the OneHotEncoder class
    print("Converting Categorical Values to Binary Integers")
    cat_encoder = OneHotEncoder(sparse=False)
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    print("Conversion Completed")

    print("Generating Figs")
    getFig6(housing)
    getFig7(housing)
    getFig8(housing)
    print("Figs Generated")

    print("Using k_Means to Locate Similarities")
    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
    similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]],
                                               sample_weight=housing_labels)

    print("Locating Similarities Operation Completed")

    print("Generating Fig")
    getFig9(housing, similarities, cluster_simil)
    print("Fig Generated")

    print("Creating Cat Pipeline")
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"))
    print("Cat Pipeline Created")

    print("Creating Transformation Pipelines")
    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler())
    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
    default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                         StandardScaler())
    preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
        remainder=default_num_pipeline)  # one column remaining: housing_median_age

    housing_prepared = preprocessing.fit_transform(housing)
    print("Housing Prepared Shape", housing_prepared.shape)
    preprocessing.get_feature_names_out()
    print("Transformation Pipelines Created")

    print("Executing Forest Reg Cross Validation. Please be Patient")
    forest_reg = make_pipeline(preprocessing,
                               RandomForestRegressor(random_state=42))
    forest_rmses = -cross_val_score(forest_reg, housing, housing_labels,
                                    scoring="neg_root_mean_squared_error", cv=10)

    pd.Series(forest_rmses).describe()
    print("Forest Reg Cross Validation Executed")

    print("Comparing Cross Validated Data with Training Set. Please be Patient")
    forest_reg.fit(housing, housing_labels)
    housing_predictions = forest_reg.predict(housing)
    forest_rmse = mean_squared_error(housing_labels, housing_predictions,
                                     squared=False)
    print("Forest RMSE", forest_rmse)
    print("Comparisons Completed")

    print("Fine Tuning Model. Please be Patient")
    full_pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("random_forest", RandomForestRegressor(random_state=42)),
    ])

    param_distribs = {'preprocessing__geo__n_clusters': randint(low=3, high=50),
                      'random_forest__max_features': randint(low=2, high=20)}

    rnd_search = RandomizedSearchCV(
        full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
        scoring='neg_root_mean_squared_error', random_state=42)

    rnd_search.fit(housing, housing_labels)
    print("Fine Tuning Completed")

    print("Generating Figs")
    getFig10()
    getFig11()
    print("Figs Generated")

    print("End")


if __name__ == '__main__':
    main()
