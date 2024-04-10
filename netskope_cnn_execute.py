#please read the 'environment_setup.md' file in the environment folder 
import pandas as pd
import numpy as np
import scipy as sp
import sys
import re
import pickle 
import random
from copy import deepcopy
import seaborn as sns
import gensim             
import torchtext
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch import optim
torch.manual_seed(10)
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

from IPython import get_ipython
ipython = get_ipython()

if '__IPYTHON__' in globals():
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 2')
    ipython.magic('eload_ext autoreload')


print('Version information')

print('python: {}'.format(sys.version))
print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('scipy: {}'.format(sp.__version__))
print('sns: {}'.format(sns.__version__))
print('gensim: {}'.format(gensim.__version__))        
print('torchtext: {}'.format(torchtext.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('torch: {}'.format(torch.__version__))

n = len(sys.argv)
if n == 3:
    print("Total arguments passed:", n-1)
if n != 3: 
    print("Require two arguments, the name of the input file that contains hisorical records and the classification problem, either fr or level. Both in string format. ")

from data_preprocessing import preprocess

pre = preprocess(sys.argv[1])
historical_loaded = pre.load_data() 
preprocessed_df = pre.preprocess(historical_loaded)
historical_remapped = pre.to_csv(preprocessed_df)

from model_embed_and_train import model_preprocess
model_embed = model_preprocess(historical_remapped[1:1000], class_prob = sys.argv[2])
model_embed.train_test_split()
model_embed.create_word_2_vec_model()
model_embed.create_cnn_model()
model_embed.train_cnn()

#model_embed.save_model()
model_embed.testing_and_predictions()
