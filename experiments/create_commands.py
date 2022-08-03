import os

DATADIR = os.path.join(os.getcwd(), '..', 'data', 'deepbind_encode_chipseq')
RESULTSDIR = os.path.join(os.getcwd(), '..', 'results', 'deepbind_encode_chipseq')

def get_tfids(datadir):
	ignore = ['__pycache__', '__init__.py', '.directory']
	tfids = [f for f in os.listdir(datadir) \
					if f not in ignore and os.path.isdir(os.path.join(datadir, f))]
	return tfids


def main():
	tfids = get_tfids(DATADIR)
	cmds = ["python train_encode_deepbind_chipseq_nn.py --tfid \"%s\"\n"%tfid for tfid in tfids]
	with open("encode_chipseq_deepbind_nn.txt", "w") as f:
		f.writelines(cmds)

if __name__ == '__main__':
	main()