# AutoMF

AutoMF is a Python package for automatic model fusion that performs automatic model selection and weighted averaging using CCA (Canonical Correlation Analysis).

## Usage

```python
from automf import AutoMF

pipelines = [  
    [('RF', RandomForestRegressor(random_state=42))],  
    [('GBDT', GradientBoostingRegressor(random_state=42)),  
     ('XGB', XGBRegressor(random_state=42)),  
     ('LGBM', LGBMRegressor(random_state=42))],  
    [('MLP', MLPRegressor(hidden_layer_sizes=(2000,), max_iter=1000, random_state=42))],  
    [('LR', LinearRegression())]  
]

auto_mf = AutoMF(pipelines)
auto_mf.fit(X_train, y_train, X_test, y_test)
r2, mae, mse, rmse, rrmse = auto_mf.evaluate(y_test)



