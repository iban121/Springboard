#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # get location of parent dir automatically
LR_ML_ARTIFACT_PATH = PARENT_DIR+'/saved_models/ML_Model.pkl'


MODEL_HISTORY = {
    'version_1': {
        'model_type' : 'Logistic Regression',
        'model_artifact_location': LR_ML_ARTIFACT_PATH
    }
}
    

