import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
from tqdm import tqdm
import hashlib
import PIL

def hashfn(x: list):
    if type(x[0]) == PIL.Image.Image:
        samples_hash = hashlib.sha1(np.stack([img.resize((32, 32)) for img in tqdm(x)])).hexdigest()
    else:
        if type(x[0]) is not str:
            samples_hash = hashlib.sha1(np.array(x)).hexdigest()
        else:
            samples_hash = hashlib.sha256(str(x).encode()).hexdigest()
    return samples_hash

def load(filename):
    try:
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
    except:
        raise Exception('File not found')
    return obj

def save(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)