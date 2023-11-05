# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA

# %%
incidents_df = pd.read_csv(
    '../data/incidents_cleaned.csv',
    index_col=0,
    parse_dates=['date'],
    date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d')
)

# %%
def compute_ratio_indicator(df, ext_df, gby, num, den, suffix, agg_fun):
    grouped_df = ext_df.groupby(gby)[den].agg(agg_fun)
    df = df.merge(grouped_df, on=gby, how='left', suffixes=[None, suffix])
    df[num+'_'+den+suffix+'_ratio'] = np.divide(df[num], df[den+suffix], out=np.zeros_like(df[num]), where=(df[den+suffix] != 0))
    df.drop(columns=[den+suffix], inplace=True)
    return df

# %%
incidents_df['city'] = incidents_df['city'].fillna('UKN')
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'city'], 'n_males', 'n_males', '_tot_year_city', 'sum') # 1
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'congressional_district'], 'n_killed', 'n_killed', '_tot_year_congdist', 'sum') # 2
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year', 'state', 'congressional_district'], 'n_injured', 'n_injured', '_tot_year_congdist', 'sum') # 2
incidents_df['n_killed_n_participants_ratio'] = incidents_df['n_killed'] / incidents_df['n_participants'] # 3
incidents_df['n_injured_n_participants_ratio'] = incidents_df['n_injured'] / incidents_df['n_participants'] # 3
incidents_df['n_unharmed_n_participants_ratio'] = incidents_df['n_unharmed'] / incidents_df['n_participants'] # 3
incidents_df['n_arrested_n_participants_ratio'] = incidents_df['n_arrested'] / incidents_df['n_participants']
incidents_df['n_females_n_males_ratio'] = incidents_df['n_females'] / incidents_df['n_males']
incidents_df['n_child_n_participants_ratio'] = incidents_df['n_participants_child'] / incidents_df['n_participants']
incidents_df['n_teen_n_participants_ratio'] = incidents_df['n_participants_teen'] / incidents_df['n_participants'] 
incidents_df = compute_ratio_indicator(incidents_df, incidents_df, ['year'], 'n_unharmed', 'n_unharmed', '_mean_year', 'mean') # 4

# %%
numeric_features = [col for col in incidents_df.columns if 'ratio' in col]
numeric_features += [
    'latitude',
    'longitude',
    'min_age_participants',
    'avg_age_participants',
    'max_age_participants',
    'location_importance',
]
categorical_features = [
    'state',
    'year',
    'month',
    'day',
    'day_of_week',
    'party'
]

# %%
print(f"Number of features before dropping rows with nan {incidents_df.shape[0]}")
incidents_df = incidents_df.dropna()
print(f"Number of features after dropping rows with nan {incidents_df.shape[0]}")

# %%
incidents_df.replace([np.inf, -np.inf], 0, inplace=True)

# %%
pca = PCA()
X_pca = pca.fit_transform(incidents_df[numeric_features])

# %%
plt.scatter(X_pca[:, 0], X_pca[:, 1], edgecolor='k', s=40)
plt.title("PCA")
plt.xlabel("1st eigenvector")
plt.ylabel("2nd eigenvector")
plt.show()

# %%
nrows=3
ncols=6
row=0
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20), sharex=True, sharey=True)
for i, col in enumerate(numeric_features):
    if i != 0 and i % ncols == 0:
        row += 1
    axs[row][i % ncols].scatter(X_pca[:, 0], X_pca[:, 1], edgecolor='k', s=40, c=incidents_df[col])
    axs[row][i % ncols].set_title(col)
    axs[row][i % ncols].set_xlabel("1st eigenvector")
    axs[row][i % ncols].set_ylabel("2nd eigenvector")
# età lungo la prima componente principale, posizione geografica lungo la seconda

# %%
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}
fig = px.scatter_matrix(
    X_pca,
    labels=labels,
    dimensions=range(4),
    color=incidents_df["latitude"]
)
fig.update_traces(diagonal_visible=False)
fig.show()

# %%
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}
fig = px.scatter_matrix(
    X_pca,
    labels=labels,
    dimensions=range(4),
    color=incidents_df["avg_age_participants"]
)
fig.update_traces(diagonal_visible=False)
fig.show()

# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

x = X_pca[:, 0]
y = X_pca[:, 2]
z = X_pca[:, 1]

ax.set_xlabel("1st eigenvector")
ax.set_ylabel("3rd eigenvector")
ax.set_zlabel("2nd eigenvector")

ax.scatter(x, y, z)

plt.show()

# %%
fig = px.scatter_3d(x=x, y=y, z=z, labels={'x': '1st eigenvector', 'y': '3rd eigenvector', 'z': '2nd eigenvector'})
fig.show()

# %%
exp_var_pca = pca.explained_variance_ratio_
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, align='center')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component')
plt.title('Explained variance by principal component')
plt.xticks(np.arange(0,len(exp_var_pca),1.0));

# %%
# TODO: heatmap with features sorted by pca


