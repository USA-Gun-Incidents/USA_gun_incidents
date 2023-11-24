# %%
import pandas as pd
import numpy as np

# %%
def compute_entropy_indicator(df, fixed_cols, var_cols):
    occ = df.groupby(fixed_cols)[var_cols].value_counts().reset_index(name='occ')
    tot = df.groupby(fixed_cols).size().reset_index(name='total')
    probs = occ.merge(tot, how='left', on=fixed_cols)

    label = 'entropy'
    for attr in var_cols:
        label += '_' + attr
    label += '_fixing'
    for attr in fixed_cols:
        label += '_' + attr

    probs[label] = -np.log2(probs['occ']/probs['total']) # 0/0 never happens
    probs.drop(columns=['occ', 'total'], inplace=True)
    
    df = df.merge(probs, how='left', on=fixed_cols+var_cols)

    return df

# %%
df = pd.DataFrame(data={
    'state': ['California','California','California','California','California','California','California','California','California',
              'Florida','Florida','Florida','Florida','Florida','Florida','Florida','Florida','Florida'],
    'semester': [1,1,1,1,2,2,2,2,2,
                 1,1,1,1,2,2,2,2,2],
    'congd': [1, 1, 1, 1, 1, 1, 1, 1, 1,
              2, 2, 2, 2, 2, 2, 2, 2, 2],
    'year': [2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012, 2012,
             2013, 2013, 2013, 2013, 2013, 2013, 2013, 2013, 2013],
    'day': [1, 1, 1, 1, 4, 4, 31, 31, 31, 
            1, 1, 1, 1, 4, 4, 31, 31, 31],
    'month': [1, 1, 1, 1, 7, 7, 10, 12, 12,
              1, 1, 1, 1, 7, 7, 10, 12, 12],
})
df

# %%
fixed_cols=['state', 'semester', 'congd', 'year']
var_cols=['day', 'month']
occ = df.groupby(fixed_cols)[var_cols].value_counts().reset_index(name='occ')
occ

# %%
tot = df.groupby(fixed_cols).size().reset_index(name='total')
tot

# %%
probs = occ.merge(tot, how='left', on=fixed_cols)
label = 'entropy'
for attr in var_cols:
    label += '_' + attr
label += '_fixing'
for attr in fixed_cols:
    label += '_' + attr

probs[label] = -np.log2(probs['occ']/probs['total'])
probs

# %%
probs.drop(columns=['occ', 'total'], inplace=True)
df = df.merge(probs, how='left', on=fixed_cols+var_cols)
df


