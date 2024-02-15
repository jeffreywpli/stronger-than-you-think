import argparse
import contextlib
import gc
import json
import logging
import multiprocessing
import os
import signal
import sys
from pprint import pprint

import inflection
import numpy as np
import torch
from collections import OrderedDict
from joblib import Parallel, delayed

import pipelines
import wrench.endmodel as endmodel
import wrench.labelmodel as labelmodel
from wrench._logging import LoggingHandler
from wrench.dataset import load_dataset
from wrench.evaluation import AverageMeter
import warnings

from util import *

# datasets = ['youtube', 'trec', 'cdr', 'chemprot', 'imdb', 'yelp', 'agnews', 'sms', 'semeval', 'amazon-high-card', 'banking-high-card']
# for data in datasets:
#     train_data, valid_data, test_data = load_dataset("./weak_datasets", data)
#     # train_data, valid_data, test_data = load_dataset("./weak_datasets", data,  extract_feature=True,
#     #                                                extract_fn="bert",
#     #                                                cache_name="bert", model_name="bert-base-cased")
#     result = train_data.summary()
#     with open("./summary.txt", "a") as outfile:
#         outfile.write(data + "\n")
#         outfile.write(str(result) + "\n\n") 
#     torch.cuda.empty_cache()

sizecount = ['semeval', 'agnews']

for data in sizecount:
    train_data, valid_data, test_data = load_dataset("./weak_datasets", data)
    with open("./summary.txt", "a") as outfile:
        # outfile.write(data + " test\n")
        # outfile.write(str(result) + "\n\n") 
        outfile.write(data + "\n")
        outfile.write("train: {}, val: {}, test: {} \n".format(len(train_data), len(valid_data), len(test_data)) )
