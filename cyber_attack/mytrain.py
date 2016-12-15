import argparse
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import minmax_scale

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

"""
train_data = pd.read_csv( 'train', delimiter=',', header = None, dtype={0:np.int32, 1:str, 2:str, 3:str, 4:np.int32, \
													5:np.int32, 6:bool, 7:np.int8, 8:np.int8, 9:np.int8, 10:np.int8, 11:bool, 12:np.int16, \
													13:bool, 14:np.int8, 15:np.int16, 16:np.int8, 17:np.int8, 18:np.int8, 19:bool, 20: bool \
													21:bool, 22:np.int16, 23:np.int16, 31:np.int16, 41:str})
test_data = pd.read_csv( 'test.in', delimiter=',', header = None, dtype={0:np.int32, 1:str, 2:str, 3:str, 4:np.int32, \
													5:np.int32, 6:bool, 7:np.int8, 8:np.int8, 9:np.int8, 10:np.int8, 11:bool, 12:np.int16, \
													13:bool, 14:np.int8, 15:np.int16, 16:np.int8, 17:np.int8, 18:np.int8, 19:bool, 20: bool \
													21:bool, 22:np.int16, 23:np.int16, 31:np.int16, 32:np.int16})
"""

""" Numercial Data Preprocessing """
print('Reading Numercial Data')
train_data_numercial = pd.read_csv( 'train', delimiter=',', header=None, dtype=float, \
																	usecols=([0]+list(range(4,19))+list(range(20,41))) )
test_data_numercial = pd.read_csv( 'test.in', delimiter=',', header=None, dtype=float, \
																	usecols=([0]+list(range(4,19))+list(range(20,41))) )

print('Normalizing Numercial Data')
all_numercial_normalized = np.concatenate( ( train_data_numercial.values, test_data_numercial.values ), axis=0 )
all_numercial_normalized = minmax_scale( all_numercial_normalized, feature_range=(0, 1), copy=True, axis=0 )

train_data_numercial[:] = all_numercial_normalized[0:len(train_data_numercial), :]
test_data_numercial[:] = all_numercial_normalized[len(train_data_numercial):, :]

""" Categorical Data Preprocessing """
print('Reading Categorical Data')
train_data_categorical = pd.read_csv( 'train', delimiter=',', header=None, usecols=[1,2,3], dtype=str, \
																			names = [ 'protocol_type', 'service', 'flag'] )
test_data_categorical = pd.read_csv( 'test.in', delimiter=',', header=None, usecols=[1,2,3], dtype=str, \
																			names = [ 'protocol_type', 'service', 'flag'] )

print('Expanding Categorical Data')
all_categorical = pd.concat( (train_data_categorical, test_data_categorical), axis=0 )

train_data_expand = train_data_numercial
test_data_expand = test_data_numercial

for col_name in all_categorical.columns.values:
	to_append = pd.get_dummies( all_categorical[col_name], prefix=col_name, sparse=False ).astype(bool)
	train_data_expand = pd.concat( (train_data_expand, to_append[0:len(train_data_expand)]), axis=1 )
	test_data_expand = pd.concat( (test_data_expand, to_append[len(train_data_expand):]), axis=1 )

""" Label Data """
print('Reading Label Data')
attack_type_dict = {'normal':0}
attack_type_names = [ 'normal', 'dos', 'u2r', 'r2l', 'probe' ]
with open( 'training_attack_types.txt', 'r' ) as file:
	for line in file.readlines():
		attack_type_dict[line.split(' ')[0]] = attack_type_names.index(line.strip().split(' ')[1])

train_labels = pd.read_csv( 'train', delimiter=',', header=None, usecols=[41], names=['labels'], \
														converters={41: lambda s: attack_type_dict[s[:-1]]} )
train_labels = pd.get_dummies( train_labels['labels'], prefix='labels', sparse=False ).astype(bool)


""" Training Stage """
print('Training Model')
model = Sequential()
model.add( Dense( train_data_expand.shape[1], input_dim=train_data_expand.shape[1], init='normal', activation='relu' ) )
model.add( Dense( 20, init='normal', activation='relu' ) )
model.add( Dense( 5, init='normal', activation='sigmoid' ) )
model.compile( loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy'] )

shuffled_X_Y = np.concatenate( ( train_data_expand.values, train_labels.values.reshape(-1,1) ), axis=1 )
shuffled_X_Y = np.random.shuffle( shuffled_X_Y )

print('Training Stage')
callsback = ModelCheckpoint( 'mymodel.h5', monitor='val_loss', save_best_only=True, mode='auto')
model.fit( shuffled_X_Y[:-5, :], shuffled_X_Y[-5:, :].reshape(-1,1), nb_epoch=100, batch_size=100, \
						validation_split=0.33, callbacks=[checkpoint] )

""" Testing Stage """
print('Testing Stage')
model.load( 'mymodel.h5' )
predictions = model.predict_classes( test_data_expand )
write_to_file = np.concatenate( (np.arange(len(predictions))+1), predictions.reshape(-1,1)), axis=1 )
np.savetxt( 'predictions.csv', write_to_file, delimiter=",", header="id,label", comments='' ,fmt='%i' )
print("Saved predictions to disk")