import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from bikesharing_model.config.core import config
from bikesharing_model.processing.features import WeekdayImputer
from bikesharing_model.processing.features import Mapper
from bikesharing_model.processing.features import OutlierHandler
from bikesharing_model.processing.features import WeekdayOneHotEncoder
from bikesharing_model.processing.features import ColumnDropper
from bikesharing_model.processing.features import WeathersitImputer

bikesharing_pipeline = Pipeline([
    ('weekday_imputer', WeekdayImputer()),  # Add your preprocessing classes
    ('weather_imputer', WeathersitImputer()),
    ("map_year", Mapper(config.model_config.year_var, config.model_config.yr_mappings)),
    ("map_month", Mapper(config.model_config.month_var, config.model_config.mnth_mappings)),
    ("map_season", Mapper(config.model_config.season_var, config.model_config.season_mappings)),
    ("map_weathersit", Mapper(config.model_config.weathersit_var, config.model_config.weathersit_mappings)),
    ("map_holiday", Mapper(config.model_config.holiday_var, config.model_config.holiday_mappings) ),
    ("map_working_day", Mapper(config.model_config.workingday_var, config.model_config.workingday_mappings)),
    ("map_hr", Mapper(config.model_config.hr_var, config.model_config.hr_mappings)),
    ('numeric_outlier_handler', OutlierHandler(config.model_config.numerical_fields)),
    ('weekday_onehot_encoder', WeekdayOneHotEncoder(config.model_config.weekday_one_hot)),
    ('column_dropper', ColumnDropper(config.model_config.unused_fields)),
    ('model', RandomForestRegressor(n_estimators=config.model_config.n_estimators, max_depth=config.model_config.max_depth, random_state=config.model_config.random_state)) 
])  

#print(bikesharing_pipeline)