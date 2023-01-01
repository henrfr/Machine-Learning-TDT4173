import h2o
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


h2o.init()


all_data = pd.read_csv("all_with_stores_pop.csv")
all_data.set_index(["dataset", "range_index"], inplace=True)
all_data['in_mall'] = all_data['mall_name'].notna()
all_data['in_chain'] = all_data['chain_name'].notna()
all_data['mall_name'] = all_data['mall_name'].fillna("None")
all_data['as'] = all_data['store_name'].str.contains(r"\b(AS)\b", case=False, regex=True)
all_data['chain_name'] = all_data['chain_name'].fillna("None")
all_data['busstop_id'] = all_data['busstop_id'].map(str)
all_data['lv1'] = all_data['lv1'].map(str) + "cat"
all_data['lv2'] = all_data['lv2'].map(str) + "cat"
all_data['lv3'] = all_data['lv3'].map(str) + "cat"
all_data['lv4'] = all_data['lv4'].map(str) + "cat"
all_data.drop(columns=[
    'store_name',
    'address',
    'importance_level',
    'busstop_id', 
    'other_stores_50', 
    'buss_stops_300', 
    'municipality_name', 
    'lv1', 
    'lat', 
    'couple_children_6_to_17_years', 
    'couple_without_children_x', 
    'single_parent_children_0_to_5_years', 
    'singles_x', 
    'singles_y', 
    'couple_without_children_y', 
    'couple_with_children', 
    'district_age_0-14_distribution', 
    'district_age_65-90_distribution', 
    'grunnkrets_population', 
    'municipality_density', 
    'all_households', 
    'lv2_population_district_div_count_stores', 
    'lv1_population_municipality_div_count_stores', 
    'lv2_population_municipality_div_count_stores', 
    'in_mall', 
    'lv3_population_district_div_count_stores', 
    'district_name', 
    'num_of_buss_stops_closer_that_1000_to_busstop', 
    'municipality_age_0-14_distribution', 
    'municipality_age_35-64_distribution', 
    'municipality_age_65-90_distribution', 

    ], inplace=True)

data_with_label = all_data.loc[["train"]]

data_with_label.set_index('store_id', inplace=True)
data_without_label = all_data.loc[['test']]
data_without_label.set_index('store_id', inplace=True)
data_without_label.drop(columns=["revenue"], inplace=True)

X, y = data_with_label.loc[:, data_with_label.columns != 'revenue'], data_with_label['revenue']

y_log = np.log1p(y)


temp = X.merge(y_log, left_index=True, right_index=True)
hf = h2o.H2OFrame(temp)

categorical_features = X.select_dtypes(include=[np.object0]).columns.to_list()


hf[categorical_features] = hf[categorical_features].asfactor()

y = "revenue"
x = hf.columns
x.remove(y)
aml = H2OAutoML(max_models=300, seed=1)
aml.train(x = x, y = y, training_frame=hf)
test = all_data.loc["test"]
test.drop(columns="revenue", inplace=True)
test.set_index('store_id', inplace=True)

hf_test = h2o.H2OFrame(test)
hf_test[categorical_features] = hf_test[categorical_features].asfactor()
preds = aml.predict(hf_test)
preds = preds.as_data_frame()
preds.index = test.index
preds.index.name = "id"
preds.rename(columns={"predict": "predicted"}, inplace=True)
preds["predicted"] = np.expm1(preds["predicted"])
preds.to_csv("2022-11-01-H2O_300models-drop-columns.csv")
lb = aml.leaderboard
for (i, name) in enumerate(lb.head(rows=lb.nrows).as_data_frame()['model_id'][:102]):
    print(h2o.get_model(name))
    preds = h2o.get_model(name).predict(hf_test)
    preds = preds.as_data_frame()
    preds.index = test.index
    preds.index.name = "id"
    preds.rename(columns={"predict": "predicted"}, inplace=True)
    preds["predicted"] = np.expm1(preds["predicted"])
    preds.to_csv(f"2022-11-01-H2O_300models-drop-columns_{i}.csv")
