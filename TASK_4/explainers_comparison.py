# %%
import pandas as pd

# %%
# TODO:
# - aggiungere le metriche calcolate sulle feature importance globali (intrinsiche al classificatore)

# %%
# faithfulness on selected records (post-hoc explainers)

lime_faithfulness_selected_df = pd.read_csv('../data/explanation_results/lime_faithfulness_selected_records.csv', index_col=0)
shap_faithfulness_selected_df= pd.read_csv('../data/explanation_results/shap_faithfulness_selected_records.csv', index_col=0)

lime_faithfulness_selected_df.columns = pd.MultiIndex.from_product([lime_faithfulness_selected_df.columns, ['lime']])
shap_faithfulness_selected_df.columns = pd.MultiIndex.from_product([shap_faithfulness_selected_df.columns, ['shap']])
lime_faithfulness_selected_df.join(shap_faithfulness_selected_df).sort_index(level=0, axis=1)

# %%
# faithfulness on selected records (ebm)

ebm_metrics_selected_df = pd.read_csv('../data/explanation_results/ebm_metrics_selected_records.csv', index_col=0)
ebm_metrics_selected_df[['faithfulness']].rename(columns={'faithfulness': 'ExplainableBoostingMachineClassifier'})

# %%
# monotonicity on selected records (post-hoc explainers)

lime_monotonicity_selected_df = pd.read_csv('../data/explanation_results/lime_monotonicity_selected_records.csv', index_col=0)
shap_monotonicity_selected_df = pd.read_csv('../data/explanation_results/shap_monotonicity_selected_records.csv', index_col=0)

lime_monotonicity_selected_df.columns = pd.MultiIndex.from_product([lime_monotonicity_selected_df.columns, ['lime']])
shap_monotonicity_selected_df.columns = pd.MultiIndex.from_product([shap_monotonicity_selected_df.columns, ['shap']])
lime_monotonicity_selected_df.join(shap_monotonicity_selected_df).sort_index(level=0, axis=1)

# %%
# monotonicity (ebm)

ebm_metrics_selected_df[['monotonicity']].rename(columns={'monotonicity': 'ExplainableBoostingMachineClassifier'})

# %%
# metrics on random records (post-hoc explainers)

lime_metrics_random_df = pd.read_csv('../data/explanation_results/lime_metrics_random_records.csv', index_col=0)
shap_metrics_random_df = pd.read_csv('../data/explanation_results/shap_metrics_random_records.csv', index_col=0)

lime_metrics_random_df.columns = pd.MultiIndex.from_product([lime_metrics_random_df.columns, ['lime']])
shap_metrics_random_df.columns = pd.MultiIndex.from_product([shap_metrics_random_df.columns, ['shap']])
lime_metrics_random_df.join(shap_metrics_random_df).sort_index(level=0, axis=1)

# %%
# metrics on random records (ebm)

ebm_metrics_ransom_df = pd.read_csv('../data/explanation_results/ebm_metrics_random_records.csv', index_col=0)
ebm_metrics_ransom_df


