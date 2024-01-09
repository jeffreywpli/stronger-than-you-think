import logging

import numpy as np
import torch
from snorkel.utils import probs_to_preds

from wrench._logging import LoggingHandler
from wrench.search import grid_search
from wrench.endmodel import Cosine
from wrench.labelmodel import FlyingSquid
from util import get_filename, has_saturated
import wrench.labelmodel as labelmodel
import wrench.endmodel as endmodel
import json
import os


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


def train_weak(label_model, end_model, train_data, val_data, test_data,
               target, lm_search_space, em_search_space, n_repeats_lm, n_repeats_em, n_trials,
               n_steps, patience, evaluation_step, hard_label, bb, max_tokens, indep_var, device="cuda", *args,
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

    if indep_var is not None:
        val_data = val_data.sample(indep_var)  # indep var = val size percentage

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

    end_model_searched_params = grid_search(end_model_search, dataset_train=covered_train_data, y_train=weak_labels,
                                            dataset_valid=val_data, metric=target, direction='auto',
                                            search_space=em_search_space,
                                            n_repeats=n_repeats_em, n_trials=n_trials, parallel=False,
                                            n_steps=n_steps,
                                            patience=patience, evaluation_step=evaluation_step, device=device)

    end_model = end_model_class(**end_model_searched_params)
    end_model.fit(dataset_train=covered_train_data, dataset_valid=val_data, y_train=weak_labels,
                  evaluation_step=evaluation_step, patience=patience, metric=target, device=device,
                  n_steps=n_steps)

    lm_val_score = label_model.test(val_data, target)
    lm_test_score = label_model.test(test_data, target)

    em_val_score = end_model.test(val_data, target)
    em_test_score = end_model.test(test_data, target)

    return {"em_test": em_test_score, "lm_test": lm_test_score,
            "em_val": em_val_score, "lm_val": lm_val_score}


def train_strong(end_model, train_data, val_data, test_data, train_val_split,
                 target, em_search_space, n_repeats_em, n_trials, n_steps, patience, evaluation_step, bb,
                 max_tokens, indep_var, device="cuda", *args, **kwargs):

    if train_data is None:
        """ 
        if training data is not given, 
        partitions the validation data into a training subset and a new (smaller) validation subset
        """

        train_data, val_data = val_data.create_split(val_data.sample(train_val_split, return_dataset=False))
        val_data.n_class = val_data.n_class

    if indep_var is not None:
        train_data = train_data.sample(indep_var)  # indep var = train size percentage

    end_model_class = getattr(endmodel, end_model)

    """ end model hyperparameter search and training """
    if end_model_class == Cosine:
        end_model_search = end_model_class(backbone=bb, max_tokens=max_tokens)
        em_search_space["backbone"] = [bb]
    else:
        end_model_search = end_model_class(max_tokens=max_tokens) 

    end_model_searched_params = grid_search(end_model_search, dataset_train=train_data,
                                            y_train=np.array(train_data.labels),
                                            dataset_valid=val_data, metric=target, direction='auto',
                                            search_space=em_search_space, n_repeats=n_repeats_em, n_trials=n_trials,
                                            parallel=False, n_steps=n_steps, patience=patience,
                                            evaluation_step=evaluation_step, device=device)

    end_model = end_model_class(**end_model_searched_params)
    end_model.fit(dataset_train=train_data, dataset_valid=val_data,  y_train=np.array(train_data.labels),
                  evaluation_step=evaluation_step, patience=patience, metric=target, device=device, n_steps=n_steps)

    em_val_score = end_model.test(val_data, target)
    em_test_score = end_model.test(test_data, target)
    return {"em_test": em_test_score, "em_val": em_val_score}





def fine_tune_on_val(label_model, end_model, train_data, val_data, test_data, train_val_split,
               target, lm_search_space, em_search_space, n_repeats_lm, n_repeats_em, n_trials,
               n_steps, patience, evaluation_step, hard_label, bb, max_tokens, indep_var, model_path, device="cuda", *args,
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

    if indep_var is not None:
        val_data = val_data.sample(indep_var)  # indep var = val size percentage

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

    end_model_searched_params = grid_search(end_model_search, dataset_train=covered_train_data, y_train=weak_labels,
                                            dataset_valid=val_data, metric=target, direction='auto',
                                            search_space=em_search_space,
                                            n_repeats=n_repeats_em, n_trials=n_trials, parallel=False,
                                            n_steps=n_steps,
                                            patience=patience, evaluation_step=evaluation_step, device=device)

    end_model = end_model_class(**end_model_searched_params)

    end_model.fit(dataset_train=covered_train_data, dataset_valid=val_data, y_train=weak_labels,
                  evaluation_step=evaluation_step, patience=patience, metric=target, device=device,
                  n_steps=n_steps, to_save = model_path)


    lm_val_score = label_model.test(val_data, target)
    lm_test_score = label_model.test(test_data, target)

    em_val_score = end_model.test(val_data, target)
    em_test_score = end_model.test(test_data, target)

    val_train_data, val_val_data = val_data.create_split(val_data.sample(train_val_split, return_dataset=False))
    val_val_data.n_class = val_data.n_class

    end_model.fit(dataset_train=val_train_data, dataset_valid=val_val_data,  y_train=np.array(val_train_data.labels),
                  evaluation_step=evaluation_step, patience=patience, metric=target, device=device, n_steps=n_steps, pretrained_model = model_path)

    val_em_val_score = end_model.test(val_data, target)
    val_em_test_score = end_model.test(test_data, target)

   
    return {"em_test": em_test_score, "lm_test": lm_test_score,
            "em_val": em_val_score, "lm_val": lm_val_score,
            "tuned_em_test": val_em_test_score, "tuned_em_val": val_em_val_score}