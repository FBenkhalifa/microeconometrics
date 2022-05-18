import numpy as np
import pandas as pd


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
def _jitter(values, j=0.01):
    return values + np.random.normal(j, 0.1, values.shape)


# Descriptive analysis
def plot_scatter(x=pd.Series, y=pd.Series)
    ax = sns.scatterplot(x=_jitter(x), y=_jitter(y), alpha=0.2)
    plt.scatter(x=0, y=x.mean(), color='r')
    plt.scatter(x=1, y=y.mean(s), color='r')
