import numpy as np, os, sys
srcpath = os.path.join(os.getcwd(), '..')
sys.path.append(srcpath)
import h5py
import pandas as pd
from pdb import set_trace as keyboard
import traceback
import argparse

import tensorflow as tf
from tensorflow import keras as tfk
from src.model_zoo import deep_bind
from src.layers import NearestNeighborConv1D, PairwiseConv1D
from src.regularizers import NearestNeighborKernelRegularizer, PairwiseKernelRegularizer
custom_objects_dict = {'NearestNeighborConv1D':NearestNeighborConv1D , 
					'NearestNeighborKernelRegularizer':NearestNeighborKernelRegularizer,
					'PairwiseConv1D':PairwiseConv1D , 
					'PairwiseKernelRegularizer':PairwiseKernelRegularizer}

DATADIR = os.path.join(os.getenv('HOME'), 'projects', 'higher_order_convolutions', 'data', 'deepbind_encode_chipseq')
RESULTSDIR = os.path.join(os.getcwd(), '..', 'results', 'deepbind_encode_chipseq')
MODELTYPE = deep_bind
EPOCHS = 50
BASELR = 1e-3
BATCHSIZE = 128


def get_tfids(datadir):
	ignore = ['__pycache__', '__init__.py', '.directory']
	tfids = [f for f in os.listdir(datadir) \
					if f not in ignore and os.path.isdir(os.path.join(datadir, f))]
	return tfids

def get_compile_options():
	# define a loss function and optimizer 
	lossfn = tfk.losses.binary_crossentropy
	optimizer = tfk.optimizers.Adam(learning_rate=BASELR)

	# define a list of metrics
	aupr = tfk.metrics.AUC(curve='PR', name='AUPR')
	auroc = tfk.metrics.AUC(curve='ROC', name='AUROC')
	acc = tfk.metrics.BinaryAccuracy(name='ACC')
	modelmetrics = [acc, auroc, aupr]
	
	compile_options = {'loss':lossfn, 'optimizer':optimizer, 'metrics':modelmetrics}
	return compile_options

def get_callbacks(early_stopping=True, historysavepath=None, modelsavedir=None, monitor='val_AUPR'):
	"""
	Define all the required callbacks here. 
	"""
	callbacks = [
			tfk.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.1,patience=10, 
												min_lr=1e-7,mode='max',verbose=0)
				]
	if early_stopping:
		callbacks.append(tfk.callbacks.EarlyStopping(monitor=monitor, patience=20, verbose=1, mode='max'))
	if historysavepath:
		callbacks.append(tfk.callbacks.CSVLogger(historysavepath, append=True))
	if modelsavedir:
		filename = 'best_model.hdf5'
		filepath = os.path.join(modelsavedir, filename)
		checkpoint_callback = tfk.callbacks.ModelCheckpoint(filepath=filepath,monitor=monitor, 
										mode='max', verbose=0, 
										save_best_only=True, 
										save_weights_only=False,)
		callbacks.append(checkpoint_callback)
	return callbacks

def test_model(model, test_data, resultdir, return_df=False):
	"""
	Evaluate the model with the given test data and dump the 
	results into a csv file.
	"""
	x_test, y_test = test_data
	values = model.evaluate(x_test, y_test, verbose=False)
	names = model.metrics_names
	index = np.arange(1,1+len(names))
	df = pd.DataFrame(data={'Metric':names, 'Value':values}, index=index)
	df.to_csv(os.path.join(resultdir, 'test_metrics.csv'))
	if return_df:
		return df

def train_model(train_data, conv_type='regular', resultdir=None,
				validation_split=0.2, batch_size=128, epochs=20, 
				extra_params={},
				early_stopping=True,
				monitor='val_AUPR', 
				test_data=None,
				return_model_and_history=False,
				save_final_model=True):
	"""
	the extra params dictionary is where you pass the L1 regularization parameter (if needed) for 
	the nearest neighbor interaction terms. 

	If the `return_model_and_history` arg is set to true, this function will return the final 
	state of the trained model and the history object returned by the model.fit function. 
	"""

	# instantiate the model
	x_train, y_train = train_data
	L, A = x_train.shape[1:]
	if not extra_params:
		model = MODELTYPE.get_model(L=L, A=A, conv_type=conv_type)
	else:
		model = MODELTYPE.get_model(L=L, A=A, conv_type=conv_type, **extra_params)

	# compile the model
	compile_options = get_compile_options()
	model.compile(**compile_options)

	# fit the model
	historysavepath = os.path.join(resultdir, 'history.csv')
	callbacks = get_callbacks(early_stopping, historysavepath, resultdir, monitor)
	hist=model.fit(x_train, y_train, validation_split=validation_split, 
					batch_size=batch_size, epochs=epochs,
					callbacks=callbacks,)
	if save_final_model:
		model.save(os.path.join(resultdir, 'final_model.hdf5'))

	# compute test metrics if test data is provided
	if test_data:
		bestmodelpath = os.path.join(resultdir, 'best_model.hdf5')
		with tfk.utils.CustomObjectScope(custom_objects_dict):
			try:
				best_model = tfk.models.load_model(bestmodelpath)
			except:
				model.load_weights(bestmodelpath)
				best_model = model
		test_model(best_model, test_data, resultdir)

	if return_model_and_history:
		return model, hist


def run_l1factor_experiment(train_data, l1_factors, resultdir, test_data=None):
	# set up the directory for regular conv results 
	#train the regular model
	print("")
	print("Training model with standard convolutions...")
	results_dir = os.path.join(resultdir, 'regular_conv_results')
	if not os.path.exists(results_dir):
		os.makedirs(results_dir)
	train_model(train_data=train_data, conv_type='regular', 
				resultdir=results_dir, epochs=EPOCHS, test_data=test_data, batch_size=BATCHSIZE)

	# loop over the list of l1 factors for the nearest neighbor and pairwise models 
	assert len(l1_factors) > 0, 'Must pass a list of l1 factors to test.'
	for i, l1 in enumerate(l1_factors):
		print("")
		print("Training model with pairwise convolutions with l1 factor {:.0e}".format(l1))
		results_dir = os.path.join(resultdir, 'pairwise_conv_results', 'l1={:.0e}'.format(l1))
		if not os.path.exists(results_dir):
			os.makedirs(results_dir)
		extra_params = {'pairwise_regularizer_type':'l1','pairwise_regularizer_const':l1}
		train_model(train_data=train_data, conv_type='pairwise', resultdir=results_dir, 
			epochs=EPOCHS, extra_params=extra_params, test_data=test_data, batch_size=BATCHSIZE)

		print("")
		print("Training model with nearest neighbor convolutions with l1 factor {:.0e}".format(l1))
		results_dir = os.path.join(resultdir, 'nearest_neighbor_conv_results', 'l1={:.0e}'.format(l1))
		if not os.path.exists(results_dir):
			os.makedirs(results_dir)
		extra_params = {'nn_regularizer_type':'l1','nn_regularizer_const':l1}
		train_model(train_data=train_data, conv_type='nearest_neighbor', resultdir=results_dir, 
			epochs=EPOCHS, extra_params=extra_params, test_data=test_data, batch_size=BATCHSIZE)


def run_l2factor_experiment(train_data, l2_factors, resultdir, test_data=None):
	# loop over the list of l1 factors for the nearest neighbor and pairwise models 
	assert len(l2_factors) > 0, 'Must pass a list of l2 factors to test.'
	for i, l2 in enumerate(l2_factors):
		print("")
		print("Training model with pairwise convolutions with l2 factor {:.0e}".format(l2))
		results_dir = os.path.join(resultdir, 'pairwise_conv_results', 'l2={:.0e}'.format(l2))
		if not os.path.exists(results_dir):
			os.makedirs(results_dir)
		extra_params = {'pairwise_regularizer_type':'l2','pairwise_regularizer_const':l2}
		train_model(train_data=train_data, conv_type='pairwise', resultdir=results_dir, 
			epochs=EPOCHS, extra_params=extra_params, test_data=test_data, batch_size=BATCHSIZE)

		print("")
		print("Training model with nearest neighbor convolutions with l2 factor {:.0e}".format(l2))
		results_dir = os.path.join(resultdir, 'nearest_neighbor_conv_results', 'l2={:.0e}'.format(l2))
		if not os.path.exists(results_dir):
			os.makedirs(results_dir)
		extra_params = {'nn_regularizer_type':'l2','nn_regularizer_const':l2}
		train_model(train_data=train_data, conv_type='nearest_neighbor', resultdir=results_dir, 
			epochs=EPOCHS, extra_params=extra_params, test_data=test_data, batch_size=BATCHSIZE)

def train_on_tfid(tfid):
	try:
		print("*************************")
		print("Loading data for TF: %s ..."%tfid)
		print("")
		data = h5py.File(os.path.join(DATADIR, tfid, 'data.h5'), 'r')
		x_train = data['X_train'][:]
		y_train = data['Y_train'][:]
		x_test = data['X_test'][:]
		y_test = data['Y_test'][:]
		if y_train.ndim == 1:
			y_train = y_train[:, None]
		if y_test.ndim == 1:
			y_test = y_test[:, None]
		data.close()

		# create a directory to store results 
		resultdir = os.path.join(RESULTSDIR, tfid)
		if not os.path.exists(resultdir):
			os.makedirs(resultdir)

		# run the experiment on the current TF
		l1_factors = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
		run_l1factor_experiment(train_data=(x_train, y_train), 
								l1_factors=l1_factors, 
								resultdir=resultdir,
								test_data=(x_test, y_test))

		# l2_factors = [1e-2, 1e-3, 1e-4, 1e-5]
		# run_l2factor_experiment(train_data=(x_train, y_train), 
		# 						l2_factors=l2_factors, 
		# 						resultdir=resultdir,
		# 						test_data=(x_test, y_test))

		print("")
		print("Finished experiment on current TF.")
		print("*************************************")
	except Exception as e:
		print(traceback.format_exc())
		print("")
		print("Exception occurred.")
		print("Skipping TF: %s"%tfid)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--tfid",default="ARID3A_K562_ARID3A_-sc-8821-_Stanford", help="TFID", type=str)
	args = parser.parse_args()
	tfid = args.tfid

	# get all TFs 
	available_tfids = get_tfids(datadir=DATADIR)
	assert tfid in available_tfids

	# run the experiment on current tfid 
	train_on_tfid(tfid)

if __name__ == '__main__':
	main()