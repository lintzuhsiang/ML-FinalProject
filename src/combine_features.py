# For NTU 2016 Fall Machine Learning Class
# Final Project: OutBrain Click Prediction
# Usage: combine_features.py ../data/ ../workspace/ --mode both --max 
# Note: pass [--mode train] or [--mode test] to do on train/test data only, default is both

import argparse
import numpy as np
import pandas as pd


""" parser """
parser = argparse.ArgumentParser()
parser.add_argument( "dataDirectory", help='directory that contains training/test data' )
parser.add_argument( "workingDirectory", help='directory that contains working data' )
parser.add_argument( "--mode", help='execute on train data, test data or both', default='both', \
												choices=['both', 'train', 'test'] 
parser.add_argument( "--max", help='maximum number of lines to read' )
parser.add_argument( "--printInfo", help='true to print info', default=False )
args = parser.parse_args()

""" determine dataset (train or test or both) """
if( args.mode=='both' )
	DataSet = ['train', 'test']
elif( args.mode=='train' )
	DataSet = ['train']
else # args.mode=='test'
	DataSet = ['test']

for data_set in DataSet:
	""" read train data """
	if( data_set=='train' )	
		if( args.printInfo ):
			print "Extract_features.py: reading input data - clicks_train.csv"
		data = pd.read_csv( args.dataDirectory + 'clicks_train.csv', delimiter=',', nrows=args.max, \
													dtype={'display_id':np.int32, 'ad_id':np.int32, 'clicked':bool})

	if( data_set=='test' )
		if( args.printInfo ):
			print "Extract_features.py: reading input data - clicks_test.csv"
		data = pd.read_csv( args.dataDirectory + 'clicks_test.csv', delimiter=',', nrows=args.max, \
													dtype={'display_id':np.int32, 'ad_id':np.int32})													

	""" concatenate events features into train/test data """
	# read events data
	if( args.printInfo ):
		print "Extract_features.py: reading input data - events.csv"
	#events_data = np.genfromtxt( args.dataDirectory + 'events.csv', dtype='int64', delimiter=',', \
	#															skip_header=1, usecols =[0,1,2,3,4], converters = {1: lambda s: int(s, 16)})
	events_data = pd.read_csv( args.dataDirectory + 'events.csv', delimiter=',', usecols =[0,1,2,3,4], na_values='\N', \
															converters = {1: lambda s: int(s, 16)}, dtype={'display_id':np.int32, \
															'uuid':np.int64, 'document_id':np.int32, 'timestamp':np.int32, 'platform':str})
									
	if( args.printInfo ):
		print "Extract_features.py: processing input data - events.csv"					
	events_data.fillna(-1, inplace=True)
	events_data['platform'] = events_data['platform'].astype('int8')

	# search for the corresponding rows in events_data by display_id
	in_array = np.in1d( data[:]['display_id'], events_data[:]['display_id'] )
	idx_found = np.searchsorted( events_data[:]['display_id'], data[:]['display_id'] )
	idx_in_limit = ( idx_found != len(events_data) )
	idx_found = idx_found * np.logical_and(idx_in_limit, in_array) + (-1)*( np.logical_not(np.logical_and(idx_in_limit, in_array)) )
	idx_found_bool = np.concatenate(( ( idx_found > -1 ).reshape(-1,1), ( idx_found > -1 ).reshape(-1,1)), axis=1)
	idx_found_bool = np.concatenate(( idx_found_bool, idx_found_bool), axis=1)
	to_append = events_data.iloc[idx_found, 1:5] * (idx_found_bool) + np.array([-1,-1,-1,-1])*(1-(idx_found_bool))
	to_append.reset_index( drop=True, inplace=True )
	to_append.rename(columns={'document_id': 'display_document_id'}, inplace=True)

	# concatenate the corresponding rows into train data
	data = pd.concat( (data, to_append), axis=1 )


	""" concatenate documents_meta features into train/test data (display documents) """
	# read documents_meta data
	if( args.printInfo ):
		print "Extract_features.py: reading input data - documents_meta.csv"	
	# documents_meta_data = np.genfromtxt( args.dataDirectory + 'documents_meta.csv', dtype='int64', skip_header=1, delimiter=',')
	documents_meta_data = pd.read_csv( args.dataDirectory + 'documents_meta.csv', delimiter=',', usecols =[0,1,2], \
																			dtype={'document_id':np.int32})																	
	documents_meta_data.fillna(-1, inplace=True)
	documents_meta_data['source_id'] = documents_meta_data['source_id'].astype('int32')
	documents_meta_data['publisher_id'] = documents_meta_data['publisher_id'].astype('int32')

	if( args.printInfo ):
		print "Extract_features.py: processing input data - documents_meta.csv (for display documents)"	
	# search for the corresponding row in events_data by display_id
	documents_meta_data.sort_values( ['document_id'], inplace=True )
	in_array = np.in1d( data[:]['display_document_id'], documents_meta_data[:]['document_id'] )
	idx_found = np.searchsorted( documents_meta_data[:]['document_id'], data[:]['display_document_id'] )
	idx_in_limit = ( idx_found != len(documents_meta_data) )
	idx_found = idx_found * np.logical_and(idx_in_limit, in_array) + (-1)*( np.logical_not(np.logical_and(idx_in_limit, in_array)) )
	idx_found_bool = np.concatenate(( ( idx_found > -1 ).reshape(-1,1), ( idx_found > -1 ).reshape(-1,1) ), axis=1)
	to_append = documents_meta_data.iloc[idx_found, 1:3] * (idx_found_bool) + np.array([-1,-1])*(1-(idx_found_bool))
	to_append.reset_index( drop=True, inplace=True )
	to_append.rename(columns={'source_id': 'display_source_id', 'publisher_id': 'display_publisher_id'}, inplace=True)

	# concatenate the corresponding rows into train data
	data = pd.concat( (data, to_append), axis=1 )


	""" concatenate promoted_content features into train/test data """
	if( args.printInfo ):
		print "Extract_features.py: reading promoted_content.csv"															
	# promoted_content_data = np.genfromtxt( args.dataDirectory + 'promoted_content.csv', dtype='int64', skip_header=1, delimiter=',')
	promoted_content_data = pd.read_csv( args.dataDirectory + 'promoted_content.csv', delimiter=',', \
																				dtype={'ad_id':np.int32, 'document_id':np.int32,  \
																				'campaign_id':np.int32, 'advertiser_id':np.int32})

	# search for the corresponding rows in promoted_content_data by display_id
	in_array = np.in1d( data[:]['ad_id'], promoted_content_data[:]['ad_id'] )
	idx_found = np.searchsorted( promoted_content_data[:]['ad_id'], data[:]['ad_id'] )
	idx_in_limit = ( idx_found != len(promoted_content_data) )
	idx_found = idx_found * np.logical_and(idx_in_limit, in_array) + (-1)*( np.logical_not(np.logical_and(idx_in_limit, in_array)) )
	idx_found_bool = np.concatenate(( ( idx_found > -1 ).reshape(-1,1), ( idx_found > -1 ).reshape(-1,1)), axis=1)
	idx_found_bool = np.concatenate(( idx_found_bool, ( idx_found > -1 ).reshape(-1,1) ), axis=1)
	to_append = promoted_content_data.iloc[idx_found, 1:4] * (idx_found_bool) + np.array([-1,-1,-1])*(1-(idx_found_bool))
	to_append.reset_index( drop=True, inplace=True )
	to_append.rename(columns={'document_id': 'ad_document_id'}, inplace=True)

	# concatenate the corresponding rows into train/test data
	data = pd.concat( (data, to_append), axis=1 )


	""" concatenate documents_meta features into train/test data (ad documents) """
	if( args.printInfo ):
		print "Extract_features.py: processing input data - documents_meta.csv (for ad documents)"	

	# search for the corresponding row in events_data by display_id
	# documents_meta_data.sort_values( ['document_id'], inplace=True )
	in_array = np.in1d( data[:]['ad_document_id'], documents_meta_data[:]['document_id'] )
	idx_found = np.searchsorted( documents_meta_data[:]['document_id'], data[:]['ad_document_id'] )
	idx_in_limit = ( idx_found != len(documents_meta_data) )
	idx_found = idx_found * np.logical_and(idx_in_limit, in_array) + (-1)*( np.logical_not(np.logical_and(idx_in_limit, in_array)) )
	idx_found_bool = np.concatenate(( ( idx_found > -1 ).reshape(-1,1), ( idx_found > -1 ).reshape(-1,1) ), axis=1)
	to_append = documents_meta_data.iloc[idx_found, 1:3] * (idx_found_bool) + np.array([-1,-1])*(1-(idx_found_bool))
	to_append.reset_index( drop=True, inplace=True )
	to_append.rename(columns={'source_id': 'ad_source_id', 'publisher_id': 'ad_publisher_id'}, inplace=True)

	# concatenate the corresponding rows into train data
	data = pd.concat( (data, to_append), axis=1 )


	""" write data to a csv file """
	if( args.printInfo ):
		print "Extract_features.py: writing output to a csv file"
	data.to_csv( args.workingDirectory + data_set + '_combined_features_reduced.csv', sep=',' )
	
	if( data_set=='train')
		data_reduced = data.iloc[:, [3,6,7,8,9,11,12,13,14]]
	else # data_set=='test'
		data_reduced = data.iloc[:, [3,6,7,8,9,11,12,13,14]]
	data_reduced.to_csv( args.workingDirectory + data_set + '_combined_features_reduced.csv', sep=',' )