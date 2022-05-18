import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Summary statistics
def _get_summary(series: pd.Series) -> pd.Series:
    """
    Get summary statistics for a pandas series.
    """
    summary = dict(
        min=series.min(),
        median=series.median(),
        mean=series.mean(),
        max=series.max(),
        std=series.std(),
        missing=series.isnull().sum(),
        n=len(series),
        unique=len(series.unique()),
    )
    return pd.Series(summary)


def get_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the summary statistics for a dataframe.
    """
    return df.apply(_get_summary, axis=0).transpose().round(2)


# Plots
def _jitter(values: pd.Series, j: float = 0.01) -> pd.Series:
    """
    Add variation to the values of a series for a jitter effect in the scatter plot.
    """
    return values + np.random.normal(j, 0.1, values.shape)


# Descriptive analysis
def plot_scatter(x=pd.Series, y=pd.Series):
    """
    Draws a scatter plot of two series and adds a mean.
    """
    ax = sns.scatterplot(x=_jitter(x), y=_jitter(y), alpha=0.2)
    plt.scatter(x=0, y=x.mean(), color='r')
    plt.scatter(x=1, y=y.mean(), color='r')
    return ax
