import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def plot_bars(features, target, df):
    '''
    plot_bars(features, target, df):

    - iterates through dataframe columns and plots barcharts for all categorical columns

    args:
        features: list of string column names
        target: name of target variable column
        df: data frame containing feature and target columns

    '''

    # iterate through feature columns and select columns of object or integer type
    for column in df[features].select_dtypes([object, int]).columns.tolist():\
        # check how many unique values column has, if over 5, we won't use column
        if len(df[column].unique()) <= 5:
            # build chart
            sns.barplot(column, target, data=df)
            plt.title(column)
            plt.ylabel(target)
            plt.show()

def plot_violin(features, target, df):
    '''
    
    '''
    for descrete in df[features].select_dtypes([object,int]).columns.tolist():
        if df[descrete].nunique() <= 5:
            for continous in df[features].select_dtypes(float).columns.tolist():
                sns.violinplot(descrete, continous, hue=target,
                data=df, split=True, palette=['blue','orange'])
                plt.title(continous + 'x' + descrete)
                plt.ylabel(continous)
                plt.show()

