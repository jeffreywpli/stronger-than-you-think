import logging

import numpy as np
import torch
from snorkel.utils import probs_to_preds

from wrench._logging import LoggingHandler
from wrench.search import grid_search
from wrench.endmodel import Cosine
from wrench.endmodel import MLPModel
from wrench.labelmodel import FlyingSquid
from util import get_filename, has_saturated
import wrench.labelmodel as labelmodel
import wrench.endmodel as endmodel
import json
import os
import random


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)

target_dict = {
    'f1_binary': ['sms', 'census', 'spouse', 'cdr', 'basketball', 'tennis', 'commercial'],
    'acc': ['semeval', 'chemprot', 'agnews', 'imdb', 'trec', 'yelp', 'youtube'],
}

data_to_target = {data: metric for metric, datasets in target_dict.items() for data in datasets}

# see wrench dataset basedataset.py,
# the following function is a modified version of the sample method in BaseDataset
def custom_stratified_sample(dataset, input_var, return_dataset=True):
    """
    Perform stratified sampling on a dataset and return either a subset dataset or indices.
    
    :param dataset: An instance of the dataset class you're working with.
    :param n_samples_per_class: The number of samples to draw from each class.
    :param return_dataset: Boolean flag to return a dataset subset if True, or indices if False.
    :return: Depending on return_dataset, either a new dataset instance or a list of indices.
    """

    unique_labels = np.unique(dataset.labels)
    sampled_indices = []

    if not isinstance(input_var, float):
        n_samples_per_class = input_var // dataset.n_class 
    for label in unique_labels:
        class_indices = np.where(dataset.labels == label)[0]
        # get the number of samples to draw from this class, if n_samples_per_class is a float then percent is 1 by default TODO (check my logic is correct here)
        
        # if a float is passed, in get the percent of that label, and at least one.
        if isinstance(input_var, float):
            n_samples_per_class = np.min(1, int(len(class_indices) * input_var))

        if len(class_indices) < n_samples_per_class:
            # If class size is smaller, allow replacement to meet the sample size requirement
            sampled_indices.extend(np.random.choice(class_indices, n_samples_per_class, replace=True))
        else:
            sampled_indices.extend(np.random.choice(class_indices, n_samples_per_class, replace=False))
    if return_dataset:
        # Use the collected indices to create a subset of the dataset
        return dataset.create_subset(sampled_indices)
    else:
        # Return the list of sampled indices directly
        return sampled_indices
    

def train_weak(label_model, end_model, train_data, val_data, test_data, seed,
               target, lm_search_space, em_search_space, n_repeats_lm, n_repeats_em, n_trials,
               n_steps, patience, evaluation_step, stratified, hard_label, fix_hyperparam, fix_steps, bb, max_tokens, indep_var, device="cuda", *args,
               **kwargs):
    """

    :param label_model_class:
    :param end_model_class:
    :param train_data:
    :param covered_train_data:
    :param val_data:
    :param target:
    :param search_space:
    :param device:
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if indep_var is not None and stratified:
        val_data = custom_stratified_sample(val_data, indep_var)
    elif indep_var is not None:
        val_data = val_data.sample(indep_var) # indep var = val size percentage
    

    label_model_class = getattr(labelmodel, label_model)
    end_model_class = getattr(endmodel, end_model)

    """ label model hyperparameter search, training and weak label prediction """
    # label model hyperparameter grid
    if label_model_class != FlyingSquid:
        label_model_searched_params = grid_search(label_model_class(), dataset_train=train_data,
                                                  dataset_valid=val_data,
                                                  metric=target, direction='auto',
                                                  search_space=lm_search_space,
                                                  n_repeats=n_repeats_lm, n_trials=n_trials)

        # fits label model
        label_model = label_model_class(**label_model_searched_params)
    else:
        label_model = label_model_class()

    label_model.fit(dataset_train=train_data, dataset_valid=val_data)

    # obtains weak labels from label model
    covered_train_data = train_data.get_covered_subset()

    weak_labels = label_model.predict_proba(covered_train_data)

    if hard_label:
        weak_labels = probs_to_preds(weak_labels)

    """ end model hyperparameter search and training """

    if end_model_class == Cosine:
        end_model_search = end_model_class(backbone=bb, max_tokens=max_tokens)
        em_search_space["backbone"] = [bb]
    elif end_model_class == MLPModel:
        end_model_search = end_model_class()
    else:
        end_model_search = end_model_class(max_tokens=max_tokens) 
    
    if not fix_hyperparam:
        end_model_searched_params = grid_search(end_model_search, dataset_train=covered_train_data, y_train=weak_labels,
                                                dataset_valid=val_data, metric=target, direction='auto',
                                                search_space=em_search_space,
                                                n_repeats=n_repeats_em, n_trials=n_trials, parallel=False,
                                                n_steps=n_steps,
                                                patience=patience, evaluation_step=evaluation_step, device=device)

        
    else:
        # If using fixed parameter size, randomly pick one from each hyperparameter from 
        # the end_model_search_space. If the fixed search space have list size to 1,
        # the result is deterministic.
        end_model_searched_params = dict()
        for key in em_search_space.keys():
            end_model_searched_params[key] = random.choice(em_search_space[key])
    
    end_model = end_model_class(**end_model_searched_params)

    if fix_steps:
        end_model.fit(dataset_train=covered_train_data, dataset_valid=val_data, y_train=weak_labels,
                  evaluation_step=evaluation_step, patience=-1, metric=target, device=device, n_steps=fix_steps)
    else:
        end_model.fit(dataset_train=covered_train_data, dataset_valid=val_data, y_train=weak_labels,
                  evaluation_step=evaluation_step, patience=patience, metric=target, device=device,
                  n_steps=n_steps)

    lm_val_score = label_model.test(val_data, target)
    lm_test_score = label_model.test(test_data, target)

    em_val_score = end_model.test(val_data, target)
    em_test_score = end_model.test(test_data, target)

    return {"em_test": em_test_score, "lm_test": lm_test_score,
            "em_val": em_val_score, "lm_val": lm_val_score}


def train_strong(end_model, train_data, val_data, test_data, train_val_split, seed,
                 target, em_search_space, n_repeats_em, n_trials, n_steps, patience, evaluation_step, stratified, fix_hyperparam, fix_steps, bb,
                 max_tokens, indep_var, experiment_flag, device="cuda", *args, **kwargs):
    """ 
        if training data is not given, 
        partitions the validation data into a training subset and a new (smaller) validation subset
    """

    random.seed(seed)
    np.random.seed(seed)

    # Check if stratfied, and perform different sampling
    # in case of training data is None, we use validation data as training data
    if experiment_flag:
        return
    else:
        if train_data is None: # TODO
        # note indep_var is a percentage or a int, (It shouldn't be a float if stratified is True though)
            if indep_var is not None and stratified:
                val_data = custom_stratified_sample(val_data, indep_var)
            elif indep_var is not None:
                val_data = val_data.sample(indep_var) # indep var = val size percentage
            # if using fixed hyperparam size and step size, 
            #  no need to further split validation into train and val data.
            if not fix_hyperparam and not fixed_steps:
                train_data, val_data = val_data.create_split(val_data.sample(train_val_split, return_dataset=False))
                val_data.n_class = val_data.n_class   
        else:
            if indep_var is not None and stratified: 
                train_data = custom_stratified_sample(train, indep_var)
            elif indep_var is not None:
                train_data = train_data.sample(indep_var)  # indep var = train size percentage 
    
    end_model_class = getattr(endmodel, end_model)

    """ end model hyperparameter search and training """
    if end_model_class == Cosine:
        end_model_search = end_model_class(backbone=bb, max_tokens=max_tokens)
        em_search_space["backbone"] = [bb]
    elif end_model_class == MLPModel:
        end_model_search = end_model_class()
    else:
        end_model_search = end_model_class(max_tokens=max_tokens) 

    if not fix_hyperparam:
        end_model_searched_params = grid_search(end_model_search, dataset_train=train_data,
                                                y_train=np.array(train_data.labels),
                                                dataset_valid=val_data, metric=target, direction='auto',
                                                search_space=em_search_space, n_repeats=n_repeats_em, n_trials=n_trials,
                                                parallel=False, n_steps=n_steps, patience=patience,
                                                evaluation_step=evaluation_step, device=device)
    else:
        # If using fixed parameter size, randomly pick one from each hyperparameter from 
        # the end_model_search_space. If the fixed search space have list size to 1,
        # the result is deterministic.
        end_model_searched_params = dict()
        for key in em_search_space.keys():
            end_model_searched_params[key] = random.choice(em_search_space[key])

    end_model = end_model_class(**end_model_searched_params)

    if fix_steps:
        end_model.fit(dataset_train=val_data,  y_train=np.array(val_data.labels),
                evaluation_step=evaluation_step, patience=patience, metric=target, device=device, n_steps=fix_steps)
    else:
        end_model.fit(dataset_train=train_data, dataset_valid=val_data,  y_train=np.array(train_data.labels),
                  evaluation_step=evaluation_step, patience=patience, metric=target, device=device, n_steps=n_steps)

    em_val_score = end_model.test(val_data, target)
    em_test_score = end_model.test(test_data, target)
    return {"em_test": em_test_score, "em_val": em_val_score}





def fine_tune_on_val(label_model, end_model, train_data, val_data, test_data, train_val_split,
               target, lm_search_space, em_search_space, n_repeats_lm, n_repeats_em, n_trials, seed,
               n_steps, patience, evaluation_step, stratified, hard_label, fix_hyperparam, fix_steps, bb, max_tokens, indep_var, model_path, device="cuda", *args,
               **kwargs):
    """

    :param label_model_class:
    :param end_model_class:
    :param train_data:
    :param covered_train_data:
    :param val_data:
    :param target:
    :param search_space:
    :param device:
    :return:
    """

    random.seed(seed)
    np.random.seed(seed)
    
    # note indep_var is a percentage or a int, (It shouldn't be a float if stratified is True though)
    # if is int
    if indep_var is not None and stratified:
        val_data = custom_stratified_sample(val_data, indep_var)
    elif indep_var is not None:
        val_data = val_data.sample(indep_var) # indep var = val size percentage
    
    label_model_class = getattr(labelmodel, label_model)
    end_model_class = getattr(endmodel, end_model)

    """ label model hyperparameter search, training and weak label prediction """
    # label model hyperparameter grid
    if label_model_class != FlyingSquid:
        label_model_searched_params = grid_search(label_model_class(), dataset_train=train_data,
                                                  dataset_valid=val_data,
                                                  metric=target, direction='auto',
                                                  search_space=lm_search_space,
                                                  n_repeats=n_repeats_lm, n_trials=n_trials)

        # fits label model
        label_model = label_model_class(**label_model_searched_params)
    else:
        label_model = label_model_class()

    label_model.fit(dataset_train=train_data, dataset_valid=val_data)

    # obtains weak labels from label model
    covered_train_data = train_data.get_covered_subset()

    weak_labels = label_model.predict_proba(covered_train_data)

    if hard_label:
        weak_labels = probs_to_preds(weak_labels)

    """ end model hyperparameter search and training """

    if end_model_class == Cosine:
        end_model_search = end_model_class(backbone=bb, max_tokens=max_tokens)
        em_search_space["backbone"] = [bb]
    else:
        end_model_search = end_model_class(max_tokens=max_tokens) 

    if not fix_hyperparam:
        end_model_searched_params = grid_search(end_model_search, dataset_train=covered_train_data, y_train=weak_labels,
                                                dataset_valid=val_data, metric=target, direction='auto',
                                                search_space=em_search_space,
                                                n_repeats=n_repeats_em, n_trials=n_trials, parallel=False,
                                                n_steps=n_steps,
                                                patience=patience, evaluation_step=evaluation_step, device=device)
    else:
        end_model_searched_params = dict()
        for key in em_search_space.keys():
            end_model_searched_params[key] = random.choice(em_search_space[key])

    end_model = end_model_class(**end_model_searched_params)
    
    end_model.fit(dataset_train=covered_train_data, dataset_valid=val_data, y_train=weak_labels,
                  evaluation_step=evaluation_step, patience=patience, metric=target, device=device,
                  n_steps=n_steps, to_save = model_path)


    lm_val_score = label_model.test(val_data, target)
    lm_test_score = label_model.test(test_data, target)

    em_val_score = end_model.test(val_data, target)
    em_test_score = end_model.test(test_data, target)


    #  futher fine-tuining on the validation set with clean label.
    #  if fixed step size, no need to resplit val set for early stopping.
    if not fix_steps:
        val_train_data, val_val_data = val_data.create_split(val_data.sample(train_val_split, return_dataset=False))
        val_val_data.n_class = val_data.n_class

        end_model.fit(dataset_train=val_train_data, dataset_valid=val_val_data,  y_train=np.array(val_train_data.labels),
                    evaluation_step=evaluation_step, patience=patience, metric=target, device=device, n_steps=n_steps, pretrained_model = model_path)
    else:
        end_model.fit(dataset_train=val_data,  y_train=np.array(val_data.labels),
                    evaluation_step=evaluation_step, patience=-1, metric=target, device=device, n_steps=fix_steps, pretrained_model = model_path)
                    
    val_em_val_score = end_model.test(val_data, target)
    val_em_test_score = end_model.test(test_data, target)

   
    return {"em_test": em_test_score, "lm_test": lm_test_score,
            "em_val": em_val_score, "lm_val": lm_val_score,
            "tuned_em_test": val_em_test_score, "tuned_em_val": val_em_val_score}