from sklearn.pipeline import Pipeline
from tpot import TPOTRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

def rmsle(y_true, y_pred):
    """
    Computes the Root Mean Squared Logarithmic Error 
    
    Args:
        y_true (np.array): n-dimensional vector of ground-truth values 
        y_pred (np.array): n-dimensional vecotr of predicted values 
    
    Returns:
        A scalar float with the rmsle value 
    
    Note: You can alternatively use sklearn and just do: 
        `sklearn.metrics.mean_squared_log_error(y_true, y_pred) ** 0.5`
    """
    y_pred[y_pred < 0] = 0
    assert (y_true >= 0).all(), 'Received negative y_true values'
    assert (y_pred >= 0).all(), 'Received negative y_pred values'
    assert y_true.shape == y_pred.shape, 'y_true and y_pred have different shapes'
    y_true_log1p = np.log1p(y_true)  # log(1 + y_true)
    y_pred_log1p = np.log1p(y_pred)  # log(1 + y_pred)
    return np.sqrt(np.mean(np.square(y_pred_log1p - y_true_log1p)))



all_data = pd.read_csv("../../own_data/all_merged.csv")
all_data_large = pd.read_csv("../../own_data/grunnkrets_norway_large.csv")
all_data_large[all_data_large['grunnkrets_density']>50000]['grunnkrets_density'] = all_data_large['grunnkrets_density'].mean()

# print(all_data_large.select_dtypes(include=['float64']).columns)
# all_data_large[all_data_large.select_dtypes(include=['float64']).columns] = all_data_large[all_data_large.select_dtypes(include=['float64']).columns].applymap(np.int64)
# print(all_data_large['grunnkrets_age_0-0'])
ages = [0,3,7,13,18,25,31,41,54,65,78,91]
# ages = [0,5,12,18,30,45,65,91]
ages = [0,15, 35,65,91]


# ages = list(range(92))
cols = []
for name in ["district_age", "grunnkrets_age", "municipality_age"]:
    for j in range(1, len(ages)):  
        age_from = ages[j-1]
        age_to = ages[j]
        col_name = f"{name}_{age_from}-{age_to-1}"
        cols.append(col_name)
        col_name = col_name + "_distribution"
        cols.append(col_name)

        # col_mean, col_std = all_data_large[col_name].mean(), all_data_large[col_name].std()
        
        # all_data_large[all_data_large[col_name]> col_mean + 3*col_std] = all_data_large[all_data_large[col_name]> col_mean + 3*col_std].apply(lambda x: col_mean +np.random.uniform(-1,1)* col_std)

# print(all_data_large['district_age_0-2'])
cols.extend(['grunnkrets_id', 
        'grunnkrets_population',
        'district_population', 
        'municipality_population', 
        'district_area', 
        'municipality_area', 
        'municipality_density', 
        'district_density', 
        "district_pop_number", 
        "municipality_pop_number"
])
all_data_large = all_data_large[cols]
all_data.set_index(["dataset", "range_index"], inplace=True)
all_data['copy_index'] = all_data.index
# print(all_data.sort_values('store_id'))
all_data = all_data.merge(all_data_large, how="left", on='grunnkrets_id')
all_data.index = pd.MultiIndex.from_tuples(
    all_data['copy_index'], 
    names=['dataset', 'range_index'])
all_data.drop(columns='copy_index', inplace=True)
# print(all_data.sort_values('store_id'))
all_data.drop(columns=['age_'+str(i) for i in range(91)], inplace=True)
all_data.drop(columns=['side_placement'], inplace=True)

# print(all_data[all_data['municipality_age_65-90']>60000]['grunnkrets_id'])
# for name in ["district_age", "municipality_age"]:
#     for j in range(1, len(ages)):  
#         age_from = ages[j-1]
#         age_to = ages[j]
#         all_data[f"{name}_{age_from}-{age_to-1}"] = all_data[f"{name}_{age_from}-{age_to-1}"].apply(lambda x: np.log10(x))


# for i in range(91):
#     all_data[f"age_{i}"] = all_data[f"grunnkrets_age_{i}-{i}"]
# all_data.drop(columns=[f'grunnkrets_age_{i}-{i}' for i in range(91)], inplace=True)
# all_data = all_data[sorted(list(all_data.columns.to_numpy()))]

# age_dist = pd.read_csv("../../data/grunnkrets_age_distribution.csv")
# age_dist.drop(age_dist[age_dist.year != 2016].index, inplace=True)
# age_dist.drop(columns='year', inplace=True)
# all_data = all_data.merge(age_dist, on="grunnkrets_id").set_index(["dataset", "range_index"])
# #all_data.drop(columns=['store_name', 'address', 'lat', 'lon', 'busstop_id', 'importance_level', 'stopplace_type', 'grunnkrets_id'], inplace=True)
# all_data.drop(columns=['grunnkrets_age_0-14', 'grunnkrets_age_15-64', 'grunnkrets_age_65-90', 'grunnkrets_population', 'grunnkrets_pop_number', 'grunnkrets_density', 
#         'district_population', 'municipality_population', 'district_area', 'municipality_area','district_age_0-14', 'district_age_15-64', 'district_age_65-90', 
#         'municipality_age_0-14', 'municipality_age_15-64', 'municipality_age_65-90', 'district_pop_number', 'municipality_pop_number', 'grunnkrets_density', 'district_density', 'municipality_density'], inplace=True)
# print(all_data['grunnkrets_age_2-2'])

all_data['in_mall'] = all_data['mall_name'].notna()
all_data['in_chain'] = all_data['chain_name'].notna()
# all_data['stopplace_type'] = all_data['stopplace_type'].fillna("Mangler type")
all_data['mall_name'] = all_data['mall_name'].fillna("None")
# all_data['address'] = all_data['address'].fillna("None")
# all_data['stopplace_type'] = all_data['stopplace_type'].fillna("None")
# all_data['stopplace_type'] = all_data['stopplace_type'].fillna("None")
all_data['as'] = all_data['store_name'].str.contains(r"\b(AS)\b", case=False, regex=True)
all_data['chain_name'] = all_data['chain_name'].fillna("None")
all_data['busstop_id'] = all_data['busstop_id'].map(str)
all_data['lv1'] = all_data['lv1'].map(str)
all_data['lv2'] = all_data['lv2'].map(str)
all_data['lv3'] = all_data['lv3'].map(str)
all_data['lv4'] = all_data['lv4'].map(str)
all_data.drop(columns=[
    'store_name', 
    # 'stopplace_type', 
    'address', 
    # "importance_level", 
    "mall_name", 
    # "busstop_id", 
    # "municipality_name", 
    # "lv1", 
    # "lv2", 
    # "lv3", 
    # "grunnkrets_id", 
    # "lat", 
    # "lon", 
    # "area_km2", 
    # 'other_stores_50', 
    # 'buss_stops_1000', 'couple_children_0_to_5_years', 'couple_children_18_or_above',
    # 'couple_children_6_to_17_years', 'couple_without_children_x',
    # 'single_parent_children_0_to_5_years',
    # 'single_parent_children_18_or_above',
    # 'single_parent_children_6_to_17_years', 'singles_x', 'singles_y', 'couple_without_children_y', 'couple_with_children',
    # 'other_households', 
    # 'single_parent_with_children',
    # 'other_stores_1000', 
    # 'other_stores_250',
    # 'in_mall',
    #    'in_chain',
    # 'buss_stops_300'

    ], inplace=True)

data_with_label = all_data.loc[["train"]]
data_with_label.set_index('store_id', inplace=True)
X, y = data_with_label.loc[:, data_with_label.columns != 'revenue'], data_with_label['revenue']
log_features = ['other_stores_1000', 'other_stores_100', 'buss_stops_300',
       'distance_closest_busstop',
       'all_households', 'num_of_buss_stops_close',
       'district_population',
       'district_area', 'district_density',
       'other_stores_50', 'buss_stops_1000', 'area_km2',
       'other_stores_250',]
# for col in log_features:
#     X[col] = np.log1p(X[col]) 
# print(X.shape)
data_train, data_test = train_test_split(data_with_label, test_size=0.2, random_state=43)
data_train=  data_train[data_train.revenue != 0]
# print(data_train.shape)
# data_train.drop(data_train[data_train['municipality_name'] == "Oslo"][-80:].index, inplace=True)
# print(data_train.shape)
CAT_SIZE = .1
X_train, y_train = data_train.loc[:, data_train.columns != 'revenue'], data_train['revenue']
X_test, y_test = data_test.loc[:, data_test.columns != 'revenue'], data_test['revenue']
y_train_scaled = np.log1p(y_train)
# y_train_scaled = y_train_scaled//CAT_SIZE
# y_train_df['category'] = y_train_df['revenue']//CAT_SIZE
# y_train_df['mean'] = y_train_df.groupby('category')['revenue'].transform('mean')
# y_train_scaled = y_train_df['category']
# y_train_scaled_mean_category = 
y_test_scaled = np.log1p(y_test)
# y_test_scaled = y_test_scaled//CAT_SIZE
y_scaled = np.log1p(y)


categorical_features = X.select_dtypes(include=[np.object0]).columns
categorical_transformer = OneHotEncoder(handle_unknown="ignore")


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="passthrough",
)
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("tpot", TPOTRegressor(verbosity=2, config_dict="TPOT sparse", n_jobs=-1, periodic_checkpoint_folder="checkpoints_tpot"))
])
pipeline.fit(X_train, y_train_scaled)
y_tpot = pipeline.predict(X_test)
print(rmsle(y_test, np.expm1(y_tpot)))
try:
    joblib.dump(pipeline, 'pipeline.pkl')
except:
    pass
pipeline.steps[1][1].export('tpot_exported_pipeline.py')


