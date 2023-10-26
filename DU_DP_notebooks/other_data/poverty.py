# %%
import pandas as pd
import matplotlib.pyplot as plt

FOLDER = './data/'
incidents_path = FOLDER + 'incidents.csv'
poverty_path = FOLDER + 'poverty_by_state_year.csv'

# %%
incidents_data = pd.read_csv(incidents_path)
poverty_data = pd.read_csv(poverty_path)

# %%
poverty_data.head()

# %%
poverty_data.info()

# %%
poverty_data['state'].unique()

# %% [markdown]
# We also have the poverty rate for the wholse USA (computed on the total population).

# %%
print(f"Range of years: [{poverty_data['year'].min()}, {poverty_data['year'].max()}]")
print(f"Number of states: {poverty_data['state'].nunique()}")

# %%
# check if state and year uniquely identify a row
poverty_data.groupby(['state', 'year']).size().max()==1

# %%
poverty_state_year = poverty_data.groupby(['state', 'year']).size()
poverty_state_year[poverty_state_year>1]

# %%
poverty_data[(poverty_data['state']=='Wyoming')]

# %% [markdown]
# For the Wyoming state there are two rows from 2009 and no data for 2010. Since the given data is ordered by year and state, the row at index 571 is probably from 2010.

# %%
poverty_data.at[571,'year']=2010

# %%
# check if now state and year uniquely identify a row
poverty_data.groupby(['state', 'year']).size().max()==1

# %%
# check if for each state we have the expected number of rows (being <state, year> a key, it means there are no missing rows)
(poverty_data.groupby('state').size()==(poverty_data['year'].max()-poverty_data['year'].min()+1)).all()

# %%
# check statistics
poverty_data.describe()

# %%
# check if there are duplicated rows
poverty_data.duplicated().sum()

# %%
# check if there are null values
poverty_data.isnull().sum()

# %%
# show years with null values
poverty_data[poverty_data['povertyPercentage'].isnull()]['year'].unique()

# %%
# plot bar plot of povertyPercentage for each state sorting by povertyPercentage
poverty_data.groupby(['state'])['povertyPercentage'].mean().sort_values().plot(kind='bar', figsize=(15, 5))

# %% [markdown]
# Group line plot according to the average povertyPercentage over the years.

# %%
mean_poverty_per_state = poverty_data.groupby(['state'])['povertyPercentage'].mean()
thresholds = [ # chosen by looking at the bar plot
    mean_poverty_per_state['Vermont'],
    mean_poverty_per_state['Alaska'],
    mean_poverty_per_state['South Dakota'],
    mean_poverty_per_state['North Carolina'],
    mean_poverty_per_state['Arkansas']
    ]

# %%
fig, axs = plt.subplots(1, 5, sharey=True, figsize=(30, 15))
fig.suptitle('Poverty percentage for each state over the years', fontsize=16)
prev_th = -1
for i in range(len(thresholds)):
    axs[i].set_xlabel('Year')
    axs[i].set_ylabel('Poverty percentage (%)')
    states = list(mean_poverty_per_state[mean_poverty_per_state.between(prev_th, thresholds[i], inclusive="left")].index)
    if i==1 or i==3:
        markers = ['o' if i%2==0 else 'x' for i in range(20)]
        axs[i].set_prop_cycle(color=plt.cm.tab20.colors, marker=markers)
    poverty_data[poverty_data['state'].isin(states)].groupby(['year', 'state'])['povertyPercentage'].mean().unstack().plot(kind='line', ax=axs[i])
    prev_th = thresholds[i]

markers = ['o' if i%2==0 else 'x' for i in range(20)]
axs[4].set_prop_cycle(color=plt.cm.tab20.colors, marker=markers)
states = list(mean_poverty_per_state[mean_poverty_per_state>=thresholds[i]].index)
poverty_data[poverty_data['state'].isin(states)].groupby(['year', 'state'])['povertyPercentage'].mean().unstack().plot(kind='line', ax=axs[4])

# %%
# average povertyPercentage of 2011 and 2013 and replace missing values
poverty_perc_2012 = poverty_data[poverty_data['year'].isin([2011, 2013])].groupby(['state'])['povertyPercentage'].mean()
poverty_data['povertyPercentage'] = poverty_data.apply(
    lambda x: poverty_perc_2012[x['state']] if x['year']==2012 else x['povertyPercentage'], axis=1
    )

# %%
def plot_map(df, col_to_plot, vmin=None, vmax=None, title=None, state_col='state'):
    import geopandas
    geo_usa = geopandas.read_file("./cb_2018_us_state_500k")
    geo_merge=geo_usa.merge(df, left_on='NAME', right_on=state_col)
    
    _, continental_ax = plt.subplots(figsize=(20, 10))
    alaska_ax = continental_ax.inset_axes([-128,22,16,8], transform=continental_ax.transData)
    hawaii_ax = continental_ax.inset_axes([-110,22.8,8,5], transform=continental_ax.transData)

    continental_ax.set_xlim(-130, -64)
    continental_ax.set_ylim(22, 53)

    alaska_ax.set_ylim(51, 72)
    alaska_ax.set_xlim(-180, -127)

    hawaii_ax.set_ylim(18.8, 22.5)
    hawaii_ax.set_xlim(-160, -154.6)

    if vmin==None or vmax==None:
        vmin, vmax = df[col_to_plot].agg(['min', 'max']) # share the same colorbar
    geo_merge.plot(column=col_to_plot, ax=continental_ax, vmin=vmin, vmax=vmax, legend=True, cmap='coolwarm')
    geo_merge.plot(column=col_to_plot, ax=alaska_ax, vmin=vmin, vmax=vmax, cmap='coolwarm')
    geo_merge.plot(column=col_to_plot, ax=hawaii_ax, vmin=vmin, vmax=vmax, cmap='coolwarm')

    for _, row in geo_merge.iterrows():
        x = row['geometry'].centroid.coords[0][0]
        y = row['geometry'].centroid.coords[0][1]
        x_displacement = 0
        y_displacement = 0
        xytext = None
        arrows = None
        if row['NAME']=="Alaska":
            x = -150
            y = 65
            xytext=(x,y)
        elif row['NAME']=="Hawaii":
            x = -157
            y = 20.5
            xytext=(x,y)
        elif row['NAME']=="Maryland":
            xytext = (x+4.5, y+0.5)
            arrows = dict(arrowstyle="-")
        elif row['NAME']=="District of Columbia":
            xytext = (x+4.5, y-1)
            arrows = dict(arrowstyle="-")
        elif row['NAME']=="Delaware":
            xytext =  (x+4.5, y+0.05)
            arrows = dict(arrowstyle="-")
        elif row['NAME']=="Rhode Island":
            xytext =  (x+5, y-0.1)
            arrows = dict(arrowstyle="-")
        elif row['NAME']=="Connecticut":
            xytext =  (x+4, y-1.5)
            arrows = dict(arrowstyle="-")
        elif row['NAME'] in ['Mississippi', 'West Virginia', 'New Hampshire']:
            y_displacement = -0.35

        
        alaska_ax.annotate(
            text=row['NAME'],
            xy=(x+x_displacement, y+y_displacement),
            xytext=xytext,
            arrowprops=arrows,
            ha='center',
            fontsize=8
        )
        hawaii_ax.annotate(
            text=row['NAME'],
            xy=(x+x_displacement, y+y_displacement),
            xytext=xytext,
            arrowprops=arrows,
            ha='center',
            fontsize=8
        )
        continental_ax.annotate(
            text=row['NAME'],
            xy=(x+x_displacement, y+y_displacement),
            xytext=xytext,
            arrowprops=arrows,
            ha='center',
            fontsize=8
        )
    plt.title(title,fontsize=16)
    for ax in [continental_ax, alaska_ax, hawaii_ax]:
        ax.set_yticks([])
        ax.set_xticks([])
    plt.show()
    plt.tight_layout()
    

# %%
vmin, vmax = poverty_data['povertyPercentage'].agg(['min', 'max'])
plot_map(poverty_data[poverty_data['year']==2011], 'povertyPercentage', vmin=vmin, vmax=vmax, title='Poverty percentage by state in 2011')

# %%
plot_map(poverty_data[poverty_data['year']==2012], 'povertyPercentage', vmin=vmin, vmax=vmax, title='Poverty percentage by state in 2012')

# %%
plot_map(poverty_data[poverty_data['year']==2013], 'povertyPercentage', vmin=vmin, vmax=vmax, title='Poverty percentage by state in 2013')

# %%
poverty_data.boxplot(column='povertyPercentage', by='state', figsize=(20, 10), rot=90)

# %%
incidents_data['date'] = pd.to_datetime(incidents_data['date'], format="%Y/%m/%d")
incidents_data['year'] = incidents_data['date'].dt.year
# join incidents and poverty data
incidents_poverty_data = incidents_data.merge(poverty_data, on=['state', 'year'], how='left')
incidents_poverty_data = incidents_poverty_data.drop(columns=['year'])
incidents_poverty_data.head()

# %%
# check if the joining operation was successful
incidents_poverty_data[
    (incidents_poverty_data['povertyPercentage'].isnull()) &
    (incidents_poverty_data['date'].dt.year<=poverty_data['year'].max())
    ].size==0


