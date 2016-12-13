# For NTU 2016 Fall Machine Learning Class
# Final Project: OutBrain Click Prediction
# Usage: extract_features.py ../data/ ../workspace/all_features.

import argparse
import numpy as np
import pandas as pd
from bisect import bisect_left

# parser
parser = argparse.ArgumentParser()
parser.add_argument( "dataDirectory", help='directory that contains training/test data' )
parser.add_argument( "outputFile", help='path of the output file' )
parser.add_argument( "--printInfo", help='true to printInfo', default=False )
args = parser.parse_args()

# read data
if( args.printInfo ):
	print "Extract_features.py: reading input data - events.csv"
events_data = np.genfromtxt( args.dataDirectory + 'events.csv', dtype='int64', delimiter=',', \
															skip_header=1, usecols =[0,1,2,3,4], converters = {1: lambda s: int(s, 16)})

if( args.printInfo ):
	print "Extract_features.py: reading promoted_content.csv"															
promoted_content_data = np.genfromtxt( args.dataDirectory + 'promoted_content.csv', dtype='int64', skip_header=1, delimiter=',')

if( args.printInfo ):
	print "Extract_features.py: processing promoted_content.csv"	
promoted_content_data = promoted_content_data[promoted_content_data[:,1].argsort()]
in_array = np.in1d( events_data[:,2], promoted_content_data[:,1] )
idx_found = np.searchsorted( promoted_content_data[:,1], events_data[:,2] )
idx_in_limit = ( idx_found != promoted_content_data.shape[0] )
idx_found = idx_found * np.logical_and(idx_in_limit, in_array) + (-1)*( np.logical_not(np.logical_and(idx_in_limit, in_array)) )
idx_found_bool = np.concatenate(( ( idx_found > -1 ).reshape(-1,1), ( idx_found > -1 ).reshape(-1,1) ), axis=1)
to_append_1 = promoted_content_data[idx_found, 2:4] * (idx_found_bool) + np.array([-1,-1])*(1-(idx_found_bool))

if( args.printInfo ):
	print "Extract_features.py: reading documents_meta.csv"	
documents_meta_data = np.genfromtxt( args.dataDirectory + 'documents_meta.csv', dtype='int64', skip_header=1, delimiter=',')

if( args.printInfo ):
	print "Extract_features.py: processing documents_meta.csv"	
documents_meta_data = documents_meta_data[documents_meta_data[:,0].argsort()]
in_array = np.in1d( events_data[:,2], documents_meta_data[:,0] )
idx_found = np.searchsorted( documents_meta_data[:,0], events_data[:,2] )
idx_in_limit = ( idx_found != documents_meta_data.shape[0] )
idx_found = idx_found * np.logical_and(idx_in_limit, in_array) + (-1)*( np.logical_not(np.logical_and(idx_in_limit, in_array)) )
idx_found_bool = np.concatenate(( ( idx_found > -1 ).reshape(-1,1), ( idx_found > -1 ).reshape(-1,1), ( idx_found > -1 ).reshape(-1,1) ), axis=1)
to_append_2 = documents_meta_data[idx_found, 1:4] * (idx_found_bool) + np.array([-1,-1,-1])*(1-(idx_found_bool))

output_to_file = np.concatenate( (events_data, to_append_1, to_append_2), axis=1 )  

# write data
if( args.printInfo ):
	print "Extract_features.py: writing output to a csv file"
column_names = "display_id, uuid, document_id, timestamp, platform, campaign_id, advertiser_id, source_id, publisher_id, publish_time"
np.savetxt( args.outputFile, output_to_file, delimiter=',', header=column_names, comments='', fmt='%i' )