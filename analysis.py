import pandas as pd
import numpy as np
import seaborn as sns
from microeconometrics.descriptives import get_summary
import matplotlib.pyplot as plt

pd.options.display.max_columns = None
pd.options.display.max_colwidth = None
pd.options.display.width = None


# Data prepartation
data = pd.read_csv('data/data_group5.csv')
data.columns = [col.lower() for col in data.columns]
data['higher_market_value_a'] = (data.value_a - data.value_h > 0).astype(int)

# Get summary statistics
h = get_summary(data.select_dtypes(include=['int64', 'float64']))



ax.figure.show()