import pickle


def save_pkl(filename, save_object):
	with open(filename,'wb') as f:
	    pickle.dump(save_object, f)

def load_pkl(filename):
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file
