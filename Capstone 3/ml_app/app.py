#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import sys
import os.path
from ml_app.modelling import ml_pipeline as mp


# In[ ]:


PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
#####


# In[ ]:


Headers = {'content-type': 'application/json'}


# In[ ]:


app = Flask(__name__)
CORS(app)
#load model
#model = pickle.load(open('ML_Model.pkl', 'rb'))


# In[ ]:


LR_ML_ARTIFACTS = mp.load_lr_ml_artifacts()


# In[ ]:


@app.route('/tweets_classifier/api/v1/liveness', methods=['GET', 'POST'])
def liveness():
    return 'API Live!'


# In[ ]:


@app.route('/tweet_classifier/api/v1/predict', method = ['POST'])
def predict():
    data = request.get_json(force=True)
    response = mp.run_lr_ml_pipeline(data, LR_ML_ARTIFACTS)
    return jsonify(output)


# In[ ]:


if __name__ == '__main__':
    app.reun(port=5000, debug=True, host = '0.0.0.0')

