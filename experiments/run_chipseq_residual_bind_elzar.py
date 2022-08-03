import os
import subprocess

def get_tfids():
	datadir = os.path.join(os.getcwd(), '..', 'data', 'deepbind_encode_chipseq')
	ignore = ['__pycache__', '__init__.py', '.directory']
	tfids = [f for f in os.listdir(datadir) \
					if f not in ignore and os.path.isdir(os.path.join(datadir, f))]
	return tfids

def main():
	tfids = get_tfids()
	for tfid in tfids:
		cmd = ["qsub", "-N", "\"residual_bind_%s\""%tfid, "job.sh", "\"%s\""%tfid]
		subprocess.call(cmd, shell=True)
		break

if __name__ == '__main__':
	main()