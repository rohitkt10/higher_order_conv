import numpy as np, os
import pandas as pd
import h5py
from pdb import set_trace as keyboard

ALPHABET = {'A':0, 'C':1, 'G':2, 'T':3}
DATADIR = os.path.join(os.getcwd(), '..', 'data', 'deepbind_encode_chipseq')
#DATADIR = os.path.join(os.getcwd(), '..', '..','..','genomics_datasets', 'deepbind_encode_chip_seq')



def get_tfids(datadir):
	ignore = ['__pycache__', '__init__.py', '.directory']
	tfids = [f for f in os.listdir(datadir) \
					if f not in ignore and os.path.isdir(os.path.join(datadir, f))]
	return tfids

def _extract_file(filepath, verbose=True):
	"""
	Extract data from a single file (training or test data file for any given TF).
	"""
	data = open(filepath, 'r')
	lines = data.readlines()
	num_lines = len(lines)
	seq_idxs = np.arange(1, num_lines, 2)
	label_idxs = np.arange(0, num_lines, 2)
	seqs = []
	labels = []
	for i,(seq_idx, label_idx) in enumerate(zip(seq_idxs, label_idxs)):
		seq = pd.Series(list(lines[seq_idx].strip()))
		seq = np.eye(4, dtype=np.float32)[seq.map(ALPHABET).values]
		seqs.append(seq)
		labels.append(np.float32(lines[label_idx].strip()[-1]))
		if verbose:
			if (i+1)%1000 == 0:
				print("Processing sequence # %d"%(i+1))
	seqs = np.array(seqs)
	labels = np.array(labels)
	data.close()
	return seqs, labels

## utility function to extract training and test data 
def extract_tf_data(tfid, datadir=DATADIR, verbose=True):
	"""
	Extract training and test data for a single tf
	"""
	train_data_path = os.path.join(datadir, tfid, 'train.fa')
	test_data_path = os.path.join(datadir, tfid, 'test.fa')
	print("Extracting training data...")
	x_train, y_train = _extract_file(train_data_path, verbose=verbose)
	print("Extracting test data...")
	x_test, y_test = _extract_file(test_data_path, verbose=verbose)
	return (x_train, y_train), (x_test, y_test)

def main():
	# set the datadir
	datadir = DATADIR

	# get all the transcription factor ids
	tfids = get_tfids(datadir=datadir)
	for tfid in tfids:
		print("")
		print("Extracting TF: %s ..."%tfid)
		print("")
		(x_train, y_train), (x_test, y_test) = extract_tf_data(tfid, datadir=datadir)

		# save the data into a h5 file 
		hdir = os.path.join(datadir, tfid)
		hfile = h5py.File(os.path.join(hdir, 'data.h5'), 'w')
		dset = hfile.create_dataset('X_train', data=x_train, compression = 'gzip')
		dset = hfile.create_dataset('Y_train', data=y_train, compression = 'gzip')
		dset = hfile.create_dataset('X_test', data=x_test, compression = 'gzip')
		dset = hfile.create_dataset('Y_test', data=y_test, compression = 'gzip')
		hfile.close()

		print("")
		print("Finished TF: %s ..."%tfid)
		print("")

if __name__ == '__main__':
	main()