import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Spliting training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], 
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regressor": LinearRegression(),
                "K-Neighbours Regressor": KNeighborsRegressor(),
                "XG-Boost Regressor": XGBRegressor(),
                "Ada Boost Regressor": AdaBoostRegressor(),
                "Cat Boosting Regressor": CatBoostRegressor()
            }

            params = {
                "Random Forest": {
                    "n_estimators": [50, 100],
                    # "max_depth": [None, 10, 20],
                    # "criterion": ["squared_error", "absolute_error"]
                },
                "Decision Tree": {
                    # "max_depth": [5, 10, 20],
                    "criterion": ["squared_error", "absolute_error"]
                },
                "Gradient Boosting": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                    # "loss": ["squared_error", "absolute_error"]
                },
                "Linear Regressor": {},  # No tuning required
                "K-Neighbours Regressor": {
                    "n_neighbors": [3, 5, 7]
                },
                "XG-Boost Regressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                    # "objective": ["reg:squarederror"]
                },
                "Ada Boost Regressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                    "loss": ["linear", "square"]
                },
                "Cat Boosting Regressor": {
                    "iterations": [50, 100],
                    "depth": [4, 6],
                    "learning_rate": [0.01, 0.1],
                    # "loss_function": ["RMSE", "MAE"],
                    # "verbose": [0]  # suppress CatBoost output
                }
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,params=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        

            
        except Exception as e:
            raise CustomException(e, sys)