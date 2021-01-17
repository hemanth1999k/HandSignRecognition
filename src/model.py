import tensorflow as tf
import numpy as np
import keras
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
import kerasncp as kncp
import numpy as np
import matplotlib.pyplot as plt
import h5py

class Model:
	
	def __init__(self,outputs,load_name=None):
		if load_name != None:
			self.model = keras.models.load(load_name)		
		else:
			self.model = Sequential()
			sample_shape = (30,128,128,1);
			self.model.add(Conv3D(32,kernel_size=(3,3,3),activation ='relu', kernel_initializer='he_uniform',input_shape=sample_shape))
			self.model.add(MaxPooling3D(pool_size=(2, 2, 2)))
			self.model.add(BatchNormalization(center=True, scale=True))
			self.model.add(Dropout(0.5))
			self.model.add(Conv3D(64, kernel_size=(3,3, 3), activation='relu', kernel_initializer='he_uniform'))
			self.model.add(MaxPooling3D(pool_size=(2, 2, 2)))
			self.model.add(BatchNormalization(center=True, scale=True))
			self.model.add(Dropout(0.5))
			self.model.add(Flatten())
			self.model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
			self.model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
			self.model.add(Dense(outputs, activation='softmax'))
			self.model.compile(loss='sparse_categorical_crossentropy',
						optimizer = keras.optimizers.Adam(lr=0.001),
						metrics=['accuracy'])
		self.model.summary() 
	
	def train(self,X_train,targets_train,batch,epochs,validation_split):
		history = self.model.fit(X_train, targets_train,
            batch_size=batch,
            epochs=epochs,
            verbose=1,
            validation_split=validation_split)
		print(history.history.keys())
		return [history.history['accuracy'],history.history['loss'],
				history.history['val_accuracy'],history.history['val_loss']]

	def save(self,name):
		self.model.save(name)

class ModelNCP:
	def __init__(self,outputs,load_name=None):
		if load_name != None:
			self.model = keras.models.load(load_name)		
		else:
			self.model = Sequential()
			sample_shape = (30,128,128,1);
					

		self.model.summary() 
	
	def train(self,X_train,targets_train,batch,epochs,validation_split):
		history = self.model.fit(X_train, targets_train,
            batch_size=batch,
            epochs=epochs,
            verbose=1,
            validation_split=validation_split)
		print(history.history.keys())
		return [history.history['accuracy'],history.history['loss'],
				history.history['val_accuracy'],history.history['val_loss']]

	def save(self,name):
		self.model.save(name)	
		
if __name__ == '__main__':
	m = Model()
