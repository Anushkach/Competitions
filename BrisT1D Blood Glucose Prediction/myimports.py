
print(f"\n---> Commencing Imports")

from gc import collect
from warnings import filterwarnings
filterwarnings('ignore')
from IPython.display import display_html, clear_output
clear_output()
import os, sys, logging, re, joblib, ctypes, shutil
from copy import deepcopy

# General library imports
from os import path, walk, getpid
from psutil import Process
from collections import Counter
from itertools import product
import ctypes
libc=ctypes.CDLL("libc.so.6")

from IPython.display import display_html, clear_output
from pprint import pprint
from functools import partial
from copy import deepcopy
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from numpy.typing import ArrayLike, NDArray
import polars as pl
import polars.selectors as cs
from polars.testing import assert_frame_equal

import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Style, init
from tqdm.notebook import tqdm

# Importing model and pipeline specifics
from category_encoders import OrdinalEncoder, OneHotEncoder

# Pipeline specifics
from sklearn.preprocessing import (RobustScaler,
MinMaxScaler,
StandardScaler,
FunctionTransformer as FT,
PowerTransformer,
)
from sklearn.impute import SimpleImputer as SI
from sklearn.model_selection import (
    RepeatedStratifiedKFold as RSKF,
    StratifiedKFold as SKF,
    KFold,
    GroupKFold as GKF,
    RepeatedKFold as RKF,
    PredefinedSplit as PDS,
    cross_val_score,
    cross_val_predict
)
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import VarianceThreshold as VT
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import (BaseEstimator,TransformerMixin, RegressorMixin, clone)
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import Ridge
from sklearn.metrics import ( 
    root_mean_squared_error as rmse,
    mean_absolute_error as mae
    )
from sklearn.ensemble import (
    RandomForestRegressor as RFR,
    ExtraTreesRegressor as ETR,
    HistGradientBoostingRegressor as HGBR)

# Importing model packages
import xgboost as xgb, lightgbm as lgb
from xgboost import XGBRegressor as XGBR, QuantileDMatrix
from lightgbm import log_evaluation, early_stopping, LGBMRegressor as LGBMR
from catboost import CatBoostRegressor as CBR, Pool

# Importing ensemble and tuning packages
import optuna
from optuna import Trial, trial, create_study
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler, CmaEsSampler
optuna.logging.disable_default_handler()

# setting rc parameters in seaborn for plots and graphs
sns.set_theme(
    {
        "axes.facecolor":"#ffffff",
        "figure.facecolor":"#ffffff",
        "axes.edgecolor": "#000000",
        "grid.color":"#ffffff",
        "font.family":["Cambria"],
        "axes.labelcolor":"#000000",
        "xtick.color":"#000000",
        "ytick.color":"#000000",
        "grid.linewidth":0.75,
        "grid.linestyle":"--",
        "axes.titlecolor":"#0099e6",
        "axes.titlesize":8.5,
        "axes.labelweight":"bold",
        "legend.fontsize":7.0,
        "legend.title_fontsize":7.0,
        "font.size":7.5,
        "xtick.labelsize":7.5,
        "ytick.labelsize":7.5
    }
        )

# color printing
def PrintColor(text:str,color=Fore.BLUE,style=Style.BRIGHT):
    print(style+color+text+Style.RESET_ALL)

# Checking package versions
import xgboost as xgb, lightgbm as lgb, catboost as cb, sklearn as sk, pandas as pd, polars as pl
print(f"---> XGBoost = {xgb.__version__} | LightGBM = {lgb.__version__} | CatBoost = {cb.__version__}")
print(f"---> Sklearn = {sk.__version__} | Pandas = {pd.__version__} | Polars = {pl.__version__}")
 
class MyLogger:
    """
    This class helps to supress logs in lightgbm and optuna
    """
    def init(self,logging_lbl:str):
        self.logger=logging.getLogger(logging_lbl)
        self.logger.setLevel(logging.ERROR)

    def info(self,message):
        pass

    def warning(self,message):
        pass

    def error(self,message):
        self.logger.error(message)

# Customizing logging for xgboost

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logger=logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
formatter=logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler=logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)

file_handler=logging.FileHandler(f"xgb_optimize.log")
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

class XGBLogging(xgb.callback.TrainingCallback):
    """
    This class is design for custom logging in xgboost
    This is to be used inside XGBoost callback
    """
    def __init__(self,epoch_log_interval=100):
        self.epoch_log_interval=epoch_log_interval

    def after_interaction(
            self,model,epoch:int,evals_log:xgb.callback.TrainingCallback.EvalsLog
    ):
        if self.epoch_log_interval<=0:
            pass

        elif (epoch%self.epoch_log_interval==0):
            for data,metric in evals_log.items():
                for metric_name, log in metric.items():
                    score=log[-1][0] if isinstance(log[-1], tuple) else log[-1]
                    logger.info(f"XGBLogging apoch {epoch} dataset {data} {metric_name} {score}")

        return False





