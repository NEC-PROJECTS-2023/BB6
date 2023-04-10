import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

housing_labels = pd.read_csv('labels.csv')
housing_prepared = pd.read_csv('housing_prepared.csv')

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

pickle.dump(forest_reg,open('model.pkl','wb'))








