# For NTU 2016 Fall Machine Learning Class
# Final Project: OutBrain Click Prediction

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def save_sparse_matrix(filename, x):
    x_coo = x.tocoo()
    row = x_coo.row
    col = x_coo.col
    data = x_coo.data
    shape = x_coo.shape
    np.savez(filename, row=row, col=col, data=data, shape=shape)

def load_sparse_matrix(filename):
    y = np.load(filename)
    z = sparse.coo_matrix((y['data'], (y['row'], y['col'])), shape=y['shape'])
    return z

""" parser """
parser = argparse.ArgumentParser()
parser.add_argument( "dataDirectory", help='directory that contains training/test data' )
parser.add_argument( "outputFile", help='path of the output file' )
parser.add_argument( "--printInfo", help='true to print info', default=False )
args = parser.parse_args()

""" read train data feature vector """
train_feature_vec = pd.read_csv( args.dataDirectory + 'train_combined_features.csv', delimiter=',', \
																	dtype={'display_id':np.int32, 'ad_id':np.int32, 'clicked':bool, \
																	'uuid':np.int64, 'display_document_id':np.int32, 'timestamp':np.int32, \
																	'platform':np.int8, 'display_source_id':np.int32, 'display_publisher_id':np.int32, \
																	'ad_document_id':np.int32, 'campaign_id':np.int32, 'advertiser_id':np.int32, \
																	'ad_source_id':np.int32, 'ad_publisher_id':np.int32})

train_feature_vec_reduced = train_feature_vec.iloc[:, [0,3,6,7,8,9,11,12,13,14]]
# now we have: 0-input index, 1-clicked, 2-timestamp, 3-platform,  4-display_source_id, 5-display_publisher_id, 
#                     6-campaign_id, 7-advertiser_id, 8-ad_source_id, 9-ad_publisher_id


"""
(Trial A)
train_feature_vec_reduced = train_feature_vec.values[:, [0,3,6,7,8,9,11,12,13,14]]
train_feature_vec_reduced_col_names = train_feature_vec.columns.values[[0,3,6,7,8,9,11,12,13,14]].astype(str)


hot_encoder = OneHotEncoder( n_values='auto', categorical_features='all', dtype='numpy.float64', sparse=True )
train_cat_features_expand = hot_encoder.fit_transform( train_feature_vec_reduced[3:10] )
save_sparse_matrix( args.outputFile, train_cat_features_expand )
hot_encoder.feature_indices_

(Trial B)
compressed_feature = train_feature_vec[['clicked', 'timestamp']]
col_name_list = [ 'platform', 'display_source_id', 'display_publisher_id', 'campaign_id', 'advertiser_id', \
													'ad_source_id', 'ad_publisher_id' ]
for col_name in col_name_list:
	to_append = pd.get_dummies(train_data_feature_vec[col_name], prefix=col_name, sparse=True)
	compressed_feature = pd.concat( (compressed_feature, to_append), axis=1 )

compressed_feature.to_csv( args.outputFile, sep=',' )

%%% do not use sparse blocks, use dense panda dataframe first, then change it into scipy sparse %%%

f = open("expand_categorical_10000_column_names.csv","w")
w = csv.writer(f)
w.writerow(list(compressed_feature.columns.values))
f.close()
"""