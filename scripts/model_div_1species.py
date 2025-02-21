#!/usr/bin/env python

"""
This script uses deep learning to model relationships between 2 input features (diversity measures) and 1 output feature (species)
It performs a supervised binary classification
Best model is saved on disk for future use
Dependency: common_dl.py -> ClassificationSpecies
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os

from common_dl import ClassificationSpecies # import class for finetuning

# 1. Read appropriate data as dataframe

data = pd.read_csv('../data/dataset1_het_froh_1species.csv')

print('data looks like:\n', data.head())



# 2. Plot features for quality control

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(20, 10))
data.hist(ax=ax, bins=50, color='black', alpha=0.7)
plt.show()
fig.savefig("../output/Quality_Control_Features_Single_Species", dpi=500)



# 3. Hot encode output feature

from sklearn.preprocessing import OneHotEncoder

hot_encoder=OneHotEncoder(sparse_output=False)

species_encoded=hot_encoder.fit_transform(np.array(data.iloc[:, -1]).reshape(-1, 1))

data[['species_encoded_1', 'species_encoded_2']]=species_encoded # number of categories determined the number of new features added, here 2



# 4. Define training, validation and test sets

from sklearn.model_selection import train_test_split # load utility for data splitting into different sets

data_train_valid, data_test = train_test_split(data, test_size=0.1, random_state=2025)

data_train, data_valid = train_test_split(data_train_valid, test_size=0.1, random_state=2025)

print('data train looks like:\n', data_train.head())



# 5. Define input and output features for each set

X_train, X_valid, X_test = data_train.iloc[:, :2], data_valid.iloc[:, :2], data_test.iloc[:, :2] # take only the first 2 features -> heterozygosity and FROH

y_train, y_valid, y_test = data_train.iloc[:, 3:], data_valid.iloc[:, 3:], data_test.iloc[:, 3:] # take the remaining features omitting features with names

print('X train looks like:\n', X_train.head())

print('y train looks like:\n', y_train.head())



# 6. Proceed to fine-tuning in search for best model

import keras_tuner as kt
from pathlib import Path
from time import strftime

tf.keras.utils.set_random_seed(2025) # set random seed for tensorflow utilities for reproducibility

if os.path.exists('single_species_modeling/best_model.keras'):
    
    print('The model has already been trained and saved on disk!')
    
    best_model=tf.keras.models.load_model('single_species_modeling/best_model.keras')
    
elif os.path.exists('single_species_modeling/best_checkpoint.keras'):
    
    print('The model has already been trained and saved on disk!')
    
    best_model=tf.keras.models.load_model('single_species_modeling/best_checkpoint.keras')

else:
    
    hyperband_tuner=kt.Hyperband(ClassificationSpecies(), objective='val_accuracy', seed=2025, max_epochs=10, factor=2, hyperband_iterations=2, overwrite=True, directory='single_species_modeling', project_name='hyperband')
        
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('single_species_modeling/best_checkpoint.keras', save_best_only=True)

    early_stopping_cb=tf.keras.callbacks.EarlyStopping(patience=2) # callback to prevent overfitting
    
    tensorboard_cb=tf.keras.callbacks.TensorBoard(Path(hyperband_tuner.project_dir)/'tensorflow'/strftime("run_%Y_%m_%d_%H_%M_%S")) # callback for tensorboard visualization
    
    hyperband_tuner.search(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid), callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb])
    
    top3_models=hyperband_tuner.get_best_models(num_models=3)
    
    best_model=top3_models[0] # select the best model
        
    best_model.save('single_species_modeling/best_model.keras') # save it




# 7. Train the best model for longer

#best_model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10)
