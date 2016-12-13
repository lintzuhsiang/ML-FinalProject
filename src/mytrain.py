# For NTU 2016 Fall Machine Learning Class
# Final Project: OutBrain Click Prediction
# Usage: mytrain.py ../data/ ../workspace/model

import argparse
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

""" parser """
parser = argparse.ArgumentParser()
parser.add_argument( "dataDirectory", help='directory that contains training/test data' )
parser.add_argument( "outputFile", help='path of the output file' )
parser.add_argument( "--printInfo", help='true to print info', default=False )
args = parser.parse_args()

""" read train data """
train_data_reduced = pd.read_csv( args.dataDirectory + 'train_combined_features.csv', delimiter=',', \
																	dtype={'clicked':bool, 'timestamp':np.int32, 'platform':np.int8, \
																	'display_source_id':np.int32, 'display_publisher_id':np.int32, \
																	'campaign_id':np.int32, 'advertiser_id':np.int32, \
																	'ad_source_id':np.int32, 'ad_publisher_id':np.int32})

""" preprocess train data """
y_train = train_data_reduced['clicked']
x_train_expand_feature_vec = train_data_reduced['timestamp']
col_name_list = [ 'platform', 'display_source_id', 'display_publisher_id', 'campaign_id', 'advertiser_id', \
									'ad_source_id', 'ad_publisher_id' ]			
for col_name in col_name_list:
	to_append = pd.get_dummies( train_data_reduced[col_name], prefix=col_name, sparse=True )
	x_train_expand_feature_vec = pd.concat( (x_train_expand_feature_vec, to_append), axis=1 )

"""" %%% CAN BE IMPROVED %%% """
# encode the sparse data
# extract useful features only

""" logistic regression """
model = Sequential()
model.add( Dense( x_train_expand_feature_vec.shape[1], input_dim=x_train_expand_feature_vec.shape[1], \
						init='normal', activation='relu' ))
model.add( Dense( 1, init='normal', activation='sigmoid' ) )
model.compile( loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy'] )
	
X = x_train_expand_feature_vec.values
Y = y_train.values.reshape(-1,1)

"""" %%% UNDER CONSTRUCTION %%% """
# EarlyStopping of SaveBestOnly to avoid over-fitting
# Need to save the trained model for future use
# Deeper network can be used
model.fit( X, Y, validation_split=0.33, nb_epoch=150, batch_size=100 )




# recycle bin
"""
def baseline_model():
	model = Sequential()
	model.add( Dense( x_train_expand_feature_vec.shape[1], input_dim=x_train_expand_feature_vec.shape[1], \
							init='normal', activation='relu' ))
	model.add( Dense( 1, init='normal', activation='sigmoid' ) )
	model.compile( loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy'] )
	return model
estimator = KerasRegressor( build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0 )
results = cross_val_score( estimator, x_train_expand_feature_vec, y_train, cv=KFold( n_splits=10 ) )
print( "Results: %.2f (%.2f) MSE" % ( results.mean(), results.std()) )
"""