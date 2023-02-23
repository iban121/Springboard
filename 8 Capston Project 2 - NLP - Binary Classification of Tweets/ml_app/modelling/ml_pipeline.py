#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from ml_app.utils.config import LR_ML_ARTIFACT_PATH
from ml_app.utils.ml_model_management import load_model_artifacts
from ml_app.data_processing.pre_processor import clean_tweets
from ml_app.modelling.ml_interface import make_lr_model_predictions


# In[ ]:


def load_lr_ml_artifacts(path = LR_ML_ARTIFACT_PATH):
    
    ml_artifacts = load_model_artifacts(path = path
                                       )
    return ml_artifacts


def run_lr_ml_pipeline(request_data, ml_artifacts):
    
    request_data = clean_tweets(tweets = request_data)
    
    pred_response = make_lr_model_predictions(request_df = request_data, 
                                             ml_model-artifacts = ml_artifacts)
    
    return pred_response 


# In[ ]:




