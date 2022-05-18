import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import microeconometrics.descriptives as ds

pd.options.display.max_columns = None
pd.options.display.max_colwidth = None
pd.options.display.width = None

# Data prepartation
data = pd.read_csv('data/data_group5.csv')
data.columns = [col.lower() for col in data.columns]
data['higher_market_value_a'] = (data.value_a - data.value_h > 0).astype(int)

# Get summary statistics
ds.get_summary(data.select_dtypes(include=['int64', 'float64']))
scatter_plot = ds.plot_scatter(x=data.higher_market_value_a, y=data.points_a)

scatter_plot.figure.show()

h = data.groupby(['team_h', 'team_a'])['points_h', 'points_a'].sum()

pd.DataFrame({'team': pd.concat([data.team_h, data.team_a]), 'points': pd.concat([data.points_h, data.points_a])}).groupby('team').agg(['sum', 'mean', 'count'])

for group, df in data.groupby(['team_h']):
    df[['team_h', 'points_h', 'higher_market_value_a']].sum()