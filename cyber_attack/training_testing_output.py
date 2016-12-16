
# coding: utf-8

# In[1]:

import argparse
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import minmax_scale

from keras.wrappers.scikit_learn import KerasRegressor


# In[2]:

""" Numercial Data Preprocessing """
print('Reading Numercial Data')
train_data_numercial = pd.read_csv('train', delimiter=',', header=None, dtype=float,                                    usecols=([0]+list(range(4,19))+list(range(20,41))))
test_data_numercial = pd.read_csv('test.in', delimiter=',', header=None, dtype=float,                                   usecols=([0]+list(range(4,19))+list(range(20,41))))
print('Normalizing Numercial Data')
all_numercial_normalized = np.concatenate((train_data_numercial.values, test_data_numercial.values ), axis=0)
all_numercial_normalized = minmax_scale(all_numercial_normalized, feature_range=(0, 1), copy=True, axis=0)

train_data_numercial[:] = all_numercial_normalized[0:len(train_data_numercial), :]
test_data_numercial[:] = all_numercial_normalized[len(train_data_numercial):, :]


# In[3]:

""" Categorical Data Preprocessing """
print('Reading Categorical Data')
train_data_categorical = pd.read_csv('train', delimiter=',', header=None, usecols=[1,2,3], dtype=str,                                      names = [ 'protocol_type', 'service', 'flag'] )
test_data_categorical = pd.read_csv('test.in', delimiter=',', header=None, usecols=[1,2,3], dtype=str,                                     names = [ 'protocol_type', 'service', 'flag'] )

print('Expanding Categorical Data')
all_categorical = pd.concat( (train_data_categorical, test_data_categorical), axis=0 )

train_data_expand = train_data_numercial
test_data_expand = test_data_numercial

for col_name in all_categorical.columns.values:
	to_append = pd.get_dummies( all_categorical[col_name], prefix=col_name, sparse=False ).astype(bool)
	train_data_expand = pd.concat( (train_data_expand, to_append[0:len(train_data_expand)]), axis=1 )
	test_data_expand = pd.concat( (test_data_expand, to_append[len(train_data_expand):]), axis=1 )


# In[4]:

""" Label Data """
print('Reading Label Data')
attack_type_dict = {'normal':0}
attack_type_names = [ 'normal', 'dos', 'u2r', 'r2l', 'probe' ]
with open( 'training_attack_types.txt', 'r' ) as file:
	for line in file.readlines():
		attack_type_dict[line.split(' ')[0]] = attack_type_names.index(line.strip().split(' ')[1])

train_labels = pd.read_csv('train', delimiter=',', header=None, usecols=[41], names=['labels'],                            converters={41: lambda s: attack_type_dict[s[:-1]]})
train_labels_expand = pd.get_dummies( train_labels['labels'], prefix='labels', sparse=False ).astype(bool)


# In[6]:

""" Training Stage """
print('Training Model')
model = Sequential()
model.add( Dense( train_data_expand.shape[1], input_dim=train_data_expand.shape[1], init='normal', activation='relu' ) )
model.add( Dense( 20, init='normal', activation='relu' ) )
model.add( Dense( 5, init='normal', activation='sigmoid' ) )
model.compile( loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy'] )


# In[8]:

shuffled_index = np.random.shuffle(np.arange(4408587))

shuffled_X = train_data_expand.as_matrix()[shuffled_index].reshape(4408587, 122)
shuffled_Y = train_labels_expand.as_matrix()[shuffled_index].reshape(4408587, 5)

checkpoint = ModelCheckpoint('mymodel.h5', monitor='val_loss', save_best_only=True, mode='auto')
model.fit(shuffled_X, shuffled_Y, nb_epoch=10, batch_size=5000,           validation_split=0.25, callbacks=[checkpoint] )


# In[9]:

""" Testing Stage """
print('Testing Stage')
#model.load( 'mymodel.h5' )
X_test = test_data_expand.as_matrix()
predictions = model.predict_classes(X_test)


# In[15]:

write_to_file = np.column_stack(((np.arange(len(predictions))+1), predictions))
np.savetxt('predictions.csv', write_to_file, delimiter=",", header="id,label", comments='' ,fmt='%i')
print('Saved predictions to disk')


# In[ ]:



