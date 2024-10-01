import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import CCA

class AutoMF:
    def __init__(self, pipelines):
        self.pipelines = pipelines
        self.best_models = []
        self.weights = None

    def metric_library(self, y_true, y_pred, metric):  
        if metric == 'mae':  
            return mean_absolute_error(y_true, y_pred)  
        elif metric == 'mse':  
            return mean_squared_error(y_true, y_pred)  
        elif metric == 'rmse':  
            return np.sqrt(mean_squared_error(y_true, y_pred))  
        elif metric == 'r2':  
            return r2_score(y_true, y_pred)  

    def fit(self, X_train, y_train, X_test, y_test):
        # 选择最佳模型
        for pipeline in self.pipelines:  
            best_score = float('inf')  
            best_model = None  

            for name, model in pipeline:  
                model.fit(X_train, y_train)  
                y_pred = model.predict(X_test)  
                mae = self.metric_library(y_test, y_pred, metric='mae')  

                if mae < best_score:  
                    best_score = mae  
                    best_model = (name, model)  

            self.best_models.append((best_model, best_score))  

        # 使用最佳模型进行预测
        predictions = []  
        
        for model_tuple, _ in self.best_models:  
            name, model = model_tuple  
            model.fit(X_train, y_train)  
            y_pred = model.predict(X_test)  
            predictions.append(y_pred.reshape(-1, 1))  

        # 重新生成预测结果
        y_pred_all = np.hstack(predictions)
        y_true_all = y_test.reshape(-1, 1)

        # 使用 CCA 分析得到权重
        cca = CCA(n_components=1)
        cca.fit(y_pred_all, y_true_all)
        self.weights = np.abs(cca.coef_)
        self.weights /= self.weights.sum()

        # 加权平均
        self.result = np.average(y_pred_all, axis=1, weights=self.weights.flatten())  

    def evaluate(self, y_test):
        r2 = self.metric_library(y_test, self.result, metric='r2')
        mae = self.metric_library(y_test, self.result, metric='mae')
        mse = self.metric_library(y_test, self.result, metric='mse')
        rmse = self.metric_library(y_test, self.result, metric='rmse')
        rrmse = rmse / np.mean(y_test)

        return r2, mae, mse, rmse, rrmse

    def get_best_models_scores_weights(self):
        best_models_scores_weights = [
            (model_tuple, score, weight) 
            for (model_tuple, score), weight in zip(self.best_models, self.weights.flatten())
        ]
        return best_models_scores_weights

# 使用示例
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

print(f"AutoMF: R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, RRMSE: {rrmse:.4f}")

# 输出每个pipeline的最佳模型、得分和权重
best_models_scores_weights = auto_mf.get_best_models_scores_weights()
for (name, _), score, weight in best_models_scores_weights:
    print(f"Best model: {name}, Score (MAE): {score:.4f}, Weight: {weight:.4f}")
