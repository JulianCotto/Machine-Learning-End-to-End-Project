####################################################################
# Developer: Julian Cotto
# Date: 9/14/2023
# File Name: functions.py
# Description: Function that saves figure outputs to local directory
####################################################################
from pathlib import Path
import matplotlib.pyplot as plt
import urllib
from urllib import request
from pandas.plotting import scatter_matrix
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from classes import ClusterSimilarity
from scipy.stats import loguniform
from scipy.stats import randint
from scipy.stats import uniform
from scipy.stats import geom
from scipy.stats import expon

# extra code – code to save the figures as high-res PNGs for the book
IMAGES_PATH = Path() / "images" / "end_to_end_project"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def getFig1(housing):
    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True)
    save_fig("bad_visualization_plot")  # extra code
    plt.show()


def getFig2(housing):
    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
    save_fig("better_visualization_plot")  # extra code
    plt.show()


def getFig3(housing):
    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
                 s=housing["population"] / 100, label="population",
                 c="median_house_value", cmap="jet", colorbar=True,
                 legend=True, sharex=False, figsize=(10, 7))
    save_fig("housing_prices_scatterplot")  # extra code
    plt.show()


def getBeautyFig(housing):
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


def getScatterFigs(housing, attributes):
    scatter_matrix(housing[attributes], figsize=(15, 12))
    save_fig("scatter_matrix_plot")  # extra code
    plt.show()


def getFig4(housing):
    housing.plot(kind="scatter", x="median_income", y="median_house_value",
                 alpha=0.1, grid=True)
    save_fig("income_vs_house_value_scatterplot")  # extra code
    plt.show()


def getFig5(housing):
    housing.plot(kind="scatter", x="median_income", y="median_house_value",
                 alpha=0.1, grid=True)
    plt.show()


def getFig6(housing):
    fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    housing["population"].hist(ax=axs[0], bins=50)
    housing["population"].apply(np.log).hist(ax=axs[1], bins=50)
    axs[0].set_xlabel("Population")
    axs[1].set_xlabel("Log of population")
    axs[0].set_ylabel("Number of districts")
    save_fig("long_tail_plot")
    plt.show()


def getFig7(housing):
    # extra code – just shows that we get a uniform distribution
    percentiles = [np.percentile(housing["median_income"], p)
                   for p in range(1, 100)]
    flattened_median_income = pd.cut(housing["median_income"],
                                     bins=[-np.inf] + percentiles + [np.inf],
                                     labels=range(1, 100 + 1))
    flattened_median_income.hist(bins=50)
    plt.xlabel("Median income percentile")
    plt.ylabel("Number of districts")
    plt.show()
    # Note: incomes below the 1st percentile are labeled 1, and incomes above the
    # 99th percentile are labeled 100. This is why the distribution below ranges
    # from 1 to 100 (not 0 to 100).


def getFig8(housing):
    # extra code – this cell generates Figure 2–18

    ages = np.linspace(housing["housing_median_age"].min(),
                       housing["housing_median_age"].max(),
                       500).reshape(-1, 1)
    gamma1 = 0.1
    gamma2 = 0.03
    rbf1 = rbf_kernel(ages, [[35]], gamma=gamma1)
    rbf2 = rbf_kernel(ages, [[35]], gamma=gamma2)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Housing median age")
    ax1.set_ylabel("Number of districts")
    ax1.hist(housing["housing_median_age"], bins=50)

    ax2 = ax1.twinx()  # create a twin axis that shares the same x-axis
    color = "blue"
    ax2.plot(ages, rbf1, color=color, label="gamma = 0.10")
    ax2.plot(ages, rbf2, color=color, label="gamma = 0.03", linestyle="--")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylabel("Age similarity", color=color)

    plt.legend(loc="upper left")
    save_fig("age_similarity_plot")
    plt.show()


def getFig9(housing, similarities, cluster_simil):
    # extra code – this cell generates Figure 2–19

    housing_renamed = housing.rename(columns={
        "latitude": "Latitude", "longitude": "Longitude",
        "population": "Population",
        "median_house_value": "Median house value (ᴜsᴅ)"})
    housing_renamed["Max cluster similarity"] = similarities.max(axis=1)

    housing_renamed.plot(kind="scatter", x="Longitude", y="Latitude", grid=True,
                         s=housing_renamed["Population"] / 100, label="Population",
                         c="Max cluster similarity",
                         cmap="jet", colorbar=True,
                         legend=True, sharex=False, figsize=(10, 7))
    plt.plot(cluster_simil.kmeans_.cluster_centers_[:, 1],
             cluster_simil.kmeans_.cluster_centers_[:, 0],
             linestyle="", color="black", marker="X", markersize=20,
             label="Cluster centers")
    plt.legend(loc="upper right")
    save_fig("district_cluster_plot")
    plt.show()

def getFig10():


    xs1 = np.arange(0, 7 + 1)
    randint_distrib = randint(0, 7 + 1).pmf(xs1)

    xs2 = np.linspace(0, 7, 500)
    uniform_distrib = uniform(0, 7).pdf(xs2)

    xs3 = np.arange(0, 7 + 1)
    geom_distrib = geom(0.5).pmf(xs3)

    xs4 = np.linspace(0, 7, 500)
    expon_distrib = expon(scale=1).pdf(xs4)

    plt.figure(figsize=(12, 7))

    plt.subplot(2, 2, 1)
    plt.bar(xs1, randint_distrib, label="scipy.randint(0, 7 + 1)")
    plt.ylabel("Probability")
    plt.legend()
    plt.axis([-1, 8, 0, 0.2])

    plt.subplot(2, 2, 2)
    plt.fill_between(xs2, uniform_distrib, label="scipy.uniform(0, 7)")
    plt.ylabel("PDF")
    plt.legend()
    plt.axis([-1, 8, 0, 0.2])

    plt.subplot(2, 2, 3)
    plt.bar(xs3, geom_distrib, label="scipy.geom(0.5)")
    plt.xlabel("Hyperparameter value")
    plt.ylabel("Probability")
    plt.legend()
    plt.axis([0, 7, 0, 1])

    plt.subplot(2, 2, 4)
    plt.fill_between(xs4, expon_distrib, label="scipy.expon(scale=1)")
    plt.xlabel("Hyperparameter value")
    plt.ylabel("PDF")
    plt.legend()
    plt.axis([0, 7, 0, 1])

    plt.show()

def getFig11():

    xs1 = np.linspace(0, 7, 500)
    expon_distrib = expon(scale=1).pdf(xs1)

    log_xs2 = np.linspace(-5, 3, 500)
    log_expon_distrib = np.exp(log_xs2 - np.exp(log_xs2))

    xs3 = np.linspace(0.001, 1000, 500)
    loguniform_distrib = loguniform(0.001, 1000).pdf(xs3)

    log_xs4 = np.linspace(np.log(0.001), np.log(1000), 500)
    log_loguniform_distrib = uniform(np.log(0.001), np.log(1000)).pdf(log_xs4)

    plt.figure(figsize=(12, 7))

    plt.subplot(2, 2, 1)
    plt.fill_between(xs1, expon_distrib,
                     label="scipy.expon(scale=1)")
    plt.ylabel("PDF")
    plt.legend()
    plt.axis([0, 7, 0, 1])

    plt.subplot(2, 2, 2)
    plt.fill_between(log_xs2, log_expon_distrib,
                     label="log(X) with X ~ expon")
    plt.legend()
    plt.axis([-5, 3, 0, 1])

    plt.subplot(2, 2, 3)
    plt.fill_between(xs3, loguniform_distrib,
                     label="scipy.loguniform(0.001, 1000)")
    plt.xlabel("Hyperparameter value")
    plt.ylabel("PDF")
    plt.legend()
    plt.axis([0.001, 1000, 0, 0.005])

    plt.subplot(2, 2, 4)
    plt.fill_between(log_xs4, log_loguniform_distrib,
                     label="log(X) with X ~ loguniform")
    plt.xlabel("Log of hyperparameter value")
    plt.legend()
    plt.axis([-8, 1, 0, 0.2])

    plt.show()