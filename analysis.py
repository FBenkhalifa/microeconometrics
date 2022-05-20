import pandas as pd

import microeconometrics.descriptives as ds

pd.options.display.max_columns = None
pd.options.display.max_colwidth = None
pd.options.display.width = None

# Data prepartation
data = pd.read_csv('data/data_group5.csv')
data.columns = [col.lower() for col in data.columns]
data['higher_value_a'] = (data.value_a - data.value_h > 0).astype(int)

# Get summary statistics
ds.get_summary(data.select_dtypes(include=['int64', 'float64']))
scatter_plot = ds.plot_scatter(x=data.higher_value_a, y=data.points_a)
scatter_plot.figure.show()

# Get team specific statistics
team_groups = data.groupby(['team_a'])
summary_team_a = pd.DataFrame().assign(
    matches=team_groups['points_a'].agg('count'),
    total_attainable_points=lambda x: x.matches * 3,
    total_points=team_groups['points_a'].agg('sum'),
    total_value_a=team_groups['value_a'].agg('sum') / 1_000_000,
    points_per_match=team_groups['points_a'].agg('mean'),
    n_more_expensive=team_groups['higher_value_a'].agg('sum'),
    ratio_more_expensive=lambda x: x.n_more_expensive / x.matches,
    points_per_match_rank=lambda x: x.points_per_match.rank(ascending=False),
    points_per_value=lambda x: x.total_points / (x.total_value_a / 1000)
).round(2).sort_values('ratio_more_expensive', ascending=False)

# Get differences between
team_markets = data.groupby('higher_value_a')
high_correlates = data.drop(['higher_value_a'], axis=1).corr().loc['points_a'].sort_values(ascending=False)[:15]
market_comparison = team_markets.agg(['min', 'mean', 'max'])[high_correlates.index].transpose().unstack(
    level=-1).round(2)

POSITION = 'H'
print(market_comparison.to_latex(
    caption='Market value comparison',
    label='tab:value-comparison',
    position=POSITION
))

print(summary_team_a.to_latex(
    caption='Team comparison',
    label='tab:team-comparison',
    position=POSITION
)
)

scatter_plot.get_figure().savefig('./paper/figures/scatter-points.pdf')