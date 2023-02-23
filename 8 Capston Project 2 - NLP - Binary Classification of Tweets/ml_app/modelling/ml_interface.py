#!/usr/bin/env python
# coding: utf-8

# In[1]:


def make_lr_model_predictions(request_df, ml_model_artifacts):
    ml_model = ml_model_artifacts['lr_model']
    
    predictions = ml_model.predict(request_df)
    
    return {'predicted_classes': list(predictions)}


# In[ ]:




