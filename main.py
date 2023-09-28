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

import numpy as np
import pandas as pd

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
    getFig1(housing)

    # create visualisation of the data with alpha set to 0.2 to see high density areas
    getFig2(housing)

    # create visualisation of the data with radius of each circle representing the district's population
    # color represents the price
    # use a predefined color map (option cmap) called jet, which ranges from blue (low values) to red (high prices)
    getFig3(housing)

    # Generate and Download a beautified version of the above image
    getBeautyFig(housing)

    # check for correlations using the corr() method
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
    print(corr_matrix['median_house_value'].sort_values(ascending=False))

    # visualize correlations using the scatter_matrix() function
    getScatterFigs(housing, attributes)
    getFig4(housing)
    getFig5(housing)

    # create new attributes to see if they are more correlated with median house value
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
    print('\n', corr_matrix["median_house_value"].sort_values(ascending=False))

    # create a copy of the stratified train set
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    # Set the missing values to some value (zero, the mean, the median, etc.). This is called imputation.
    imputer = SimpleImputer(strategy="median")

    # create a copy of the data with only the numerical attributes
    housing_num = housing.select_dtypes(include=[np.number])

    # fit the imputer instance to the training data using the fit() method
    imputer.fit(housing_num)

    # variables introduced for visualization of data
    print(imputer.statistics_)

    # Check that this is the same as manually computing the median of each attribute:
    print(housing_num.median().values)

    # transform the training set by replacing missing values with the learned medians
    X = imputer.transform(housing_num)

    print(imputer.feature_names_in_)
    print(imputer.strategy)

    # convert the NumPy array into a Pandas DataFrame
    housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                              index=housing_num.index)

    # code to drop outliers
    # housing = housing.iloc[outlier_pred == 1]
    # housing_labels = housing_labels.iloc[outlier_pred == 1]

    # visualize text attributes
    housing_cat = housing[["ocean_proximity"]]
    print(housing_cat.head(8))

    # convert text categories to numbers using the OrdinalEncoder class
    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    print(housing_cat_encoded[:8])
    print(ordinal_encoder.categories_)

    # convert categorical values into one-hot vectors using the OneHotEncoder class
    cat_encoder = OneHotEncoder(sparse_output=False)
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    print(housing_cat_1hot)
    print(cat_encoder.categories_)

    # normalize the data to a scale
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)
    print(housing_num_min_max_scaled)

    # standardize the data to a normal distribution
    std_scaler = StandardScaler()
    housing_num_std_scaled = std_scaler.fit_transform(housing_num)
    print(housing_num_std_scaled)

    getFig6(housing)
    getFig7(housing)
    getFig8(housing)

    # train a linear regression model
    target_scaler = StandardScaler()
    scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

    model = LinearRegression()
    model.fit(housing[["median_income"]], scaled_labels)
    some_new_data = housing[["median_income"]].iloc[:5]  # pretend this is new data

    scaled_predictions = model.predict(some_new_data)
    predictions = target_scaler.inverse_transform(scaled_predictions)

    # scale labels and train regression model on scaled labels
    model = TransformedTargetRegressor(LinearRegression(),
                                       transformer=StandardScaler())
    model.fit(housing[["median_income"]], housing_labels)
    predictions = model.predict(some_new_data)

    print("Predictions:", predictions)

    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
    similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]],
                                               sample_weight=housing_labels)

    print(similarities[:3])

    getFig9(housing, similarities, cluster_simil)

    num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    set_config(display='diagram')
    print(num_pipeline)

    housing_num_prepared = num_pipeline.fit_transform(housing_num)
    print(housing_num_prepared[:2].round(2))

    monkey_patch_get_signature_names_out()

    df_housing_num_prepared = pd.DataFrame(
        housing_num_prepared, columns=num_pipeline.get_feature_names_out(),
        index=housing_num.index)
    print(df_housing_num_prepared)

    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"))

    preprocessing = make_column_transformer(
        (num_pipeline, make_column_selector(dtype_include=np.number)),
        (cat_pipeline, make_column_selector(dtype_include=object)),
    )

    housing_prepared = preprocessing.fit_transform(housing)
    print(housing_prepared)

    # extra code – shows that we can get a DataFrame out if we want
    # housing_prepared_fr = pd.DataFrame(
    #     housing_prepared,
    #     columns=preprocessing.get_feature_names_out(),
    #     index=housing.index)
    # housing_prepared_fr.head(2)

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
    print(housing_prepared.shape)
    print(preprocessing.get_feature_names_out())

    lin_reg = make_pipeline(preprocessing, LinearRegression())
    print(lin_reg.fit(housing, housing_labels))

    housing_predictions = lin_reg.predict(housing)
    print(housing_predictions[:5].round(-2))
    print(housing_labels.iloc[:5].values)

    lin_rmse = mean_squared_error(housing_labels, housing_predictions,
                                  squared=False)
    print(lin_rmse)

    tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
    tree_reg.fit(housing, housing_labels)

    housing_predictions = tree_reg.predict(housing)
    tree_rmse = mean_squared_error(housing_labels, housing_predictions,
                                   squared=False)
    print(tree_rmse)

    # marker for end of program
    input("Press Enter to Exit...")


if __name__ == '__main__':
    main()
