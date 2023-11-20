# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
indicators_df = pd.read_csv('../data/incidents_cleaned_indicators.csv', index_col=0)

# %%
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))

sns.heatmap(indicators_df.corr('pearson'), ax=axs[0])
axs[0].set_title('Pearson Correlation Heatmap')

sns.heatmap(indicators_df.corr('kendall'), ax=axs[1])
axs[1].set_title('Kendall Correlation Heatmap')

sns.heatmap(indicators_df.corr('spearman'), ax=axs[2])
axs[2].set_title('Spearman Correlation Heatmap')


