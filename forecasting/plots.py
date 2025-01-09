import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpy import arange


def plot_boxplot(df: pd.DataFrame, column: str, top=None):
    if top:
        counts = df[column].value_counts()
        counts = counts[:top]
        df_plot = df[df[column].isin(counts.index)]
    else:
        df_plot = df

    # sort categories by their median to improve readability of the output chart
    order = df_plot.groupby(column)['traffic_volume'].median().sort_values(ascending=False)

    # make final plot
    # sns.boxplot(x='traffic_volume', y=column, data=df_plot, order=list(order.index))
    sns.boxplot(x='traffic_volume', y=column, data=df_plot, order=arange(0, 24))


def plot_aggregated_traffic(df: pd.DataFrame, grouping_var: str, agg_func='median'):
    """
    Plots a bar graph showing the aggregated traffic volume based on a specified grouping variable
    and aggregation function.

    Parameters:
        df (DataFrame): A Pandas DataFrame containing traffic data.
        grouping_var (str): A string representing the column name in 'df' to group the data by.
        agg_func (str or callable, optional): A function or string representing the function to be used for
        aggregating the data. Defaults to 'median'.

    Returns:
        None: Displays a bar plot of the aggregated traffic data.
    """
    df_plot = df.groupby(grouping_var).traffic_volume.agg(agg_func)
    df_plot = df_plot.sort_index(key=lambda x: x.astype(int))
    df_plot.plot.bar(width=.85)


def plot_traffic_histogram(df: pd.DataFrame, bins=30, edgecolor='white', **kwargs):
    df['traffic_volume'].plot.hist(bins=bins, edgecolor=edgecolor, **kwargs)