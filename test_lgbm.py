import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from copy import deepcopy

params = {
'objective': 'regression',
'verbose': -1,
'num_leaves': 3
}

X = np.random.rand(100,2)
Y = np.ravel(np.random.rand(100,1))
lgbm = lgb.train(params, lgb.Dataset(X,label=Y),num_boost_round=1)

f = open('test_pickle.pkl','wb')
pickle.dump(lgbm,f)
f.close()

print(lgbm.params)

## Deep copy will missing params
new_model = deepcopy(lgbm)
print(new_model.params)


## Load from file is fine
import pickle
f = open('test_pickle.pkl','rb')
m2 = pickle.load(f)
f.close()

print(lgbm.params)
