import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from bikesharing_model import __version__ as _version
from bikesharing_model.config.core import config
from bikesharing_model.pipeline import bikesharing_pipeline
from bikesharing_model.processing.data_manager import load_pipeline
from bikesharing_model.processing.data_manager import pre_pipeline_preparation
from bikesharing_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
bikesharing_pipeline= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))

    results = {"predictions": None, "version": _version, "errors": errors}
    
    if errors:
        return results
    
    predictions = bikesharing_pipeline.predict(validated_data)
    results = {"predictions": predictions,"version": _version, "errors": errors}
    print(results)
    if not errors:

        predictions = bikesharing_pipeline.predict(validated_data)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        #print(results)

    return results

if __name__ == "__main__":

    #data_in={'PassengerId':[79],'Pclass':[2],'Name':["Caldwell, Master. Alden Gates"],'Sex':['male'],'Age':[0.83],
                #'SibSp':[0],'Parch':[2],'Ticket':['248738'],'Cabin':[np.nan,],'Embarked':['S'],'Fare':[29]}
    
    data_in={'dteday':['2012-11-29'],'season':['winter'],'hr':['9pm'],'holiday':['No'],'weekday':['Thu'],'workingday':['Yes'],
    'weathersit':['Clear'],'temp':[4.220000000000001],'atemp':[1.9982000000000006],
    'hum':[65.0],'windspeed':[8.5],'casual':[7],'registered':[201]}

    

    make_prediction(input_data=data_in)
