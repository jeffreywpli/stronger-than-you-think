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

torch.backends.cudnn.deterministic = True

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logger = logging.getLogger(__name__)

target_dict = {
    'f1_binary': ['sms', 'census', 'spouse', 'cdr', 'basketball', 'tennis', 'commercial'],
    'acc': ['semeval', 'chemprot', 'agnews', 'imdb', 'trec', 'yelp', 'youtube', 'amazon-high-card', 
    'banking-high-card', 'news-category', 'amazon31', 'banking77', 'massive', 'dbpedia-219', 'massive_lowcard', 'dbpedia'],
}

token_dict = {
    "agnews": 128,
    "imdb": 256,
    "yelp": 512,
    "trec": 64,
    "chemprot": 400,
    "youtube": 512,
    "spouse" : 525
}

data_to_target = {data: metric for metric, datasets in target_dict.items() for data in datasets}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An pipeline manager for the limit of weak supervision")

    parser.add_argument("-d", "--data", help="Dataset to run experiments on (default: youtube)", default="youtube")

    parser.add_argument("-p", "--pipeline", help="Pipeline to run (default: end-to-end)", default="end-to-end")

    parser.add_argument("-r", "--root", help="Directory root storing all the datasets (default: ./)",
                        default="../../../weak_datasets")

    parser.add_argument("-j", "--jobs", help="Number of jobs to distribute experiments across (default: 1)", type=int,
                        default=1)

    parser.add_argument("-lm", "--label-model", help="Label model to use (default: MajorityVoting)",
                        default='MajorityVoting')

    parser.add_argument("-em", "--end-model", help="End model to use (default: MLPModel)", default='MLPModel')

    parser.add_argument("-s", "--steps", help="Maximum number of steps for each experiment (default: 100000)", type=int,
                        default=100000)

    parser.add_argument("-nr", "--num-runs", help="Number of runs for each experiment (default: 5)", type=int,
                        default=5)

    parser.add_argument("-gs", "--grid-size",
                        help="Maximum size of the grid search for each experiment (default: 1000)",
                        type=int, default=1000)

    parser.add_argument("-pa", "--patience", help="End model patience (default: 100)", type=int, default=100)

    parser.add_argument("-es", "--evaluation-step", help="Number of steps between evaluations (default: 10)",
                        type=int, default=10)

    parser.add_argument("-ef", "--extract_fn", help="Extraction function used for dataset", default="bert")

    parser.add_argument("-exm", "--extract_model", help="Model used for feature extraction", default="bert-base-cased")

    parser.add_argument("-nrlm", "--num_repeat_lm", help="Number of trial repeats used in the label model grid search"
                                                         "(default: 1)", type=int, default=1)

    parser.add_argument("-nrem", "--num_repeat_em", help="Number of trial repeats used in the end model grid search"
                                                         "(default: 1)", type=int, default=1)

    parser.add_argument("--device", help="Device to run discriminative model on (default: cuda)", default='cuda')

    parser.add_argument("-hl", "--hard-label", help="use hard label for label model (default: False)",
                        action='store_true')

    parser.add_argument("-bb", "--backbone", help="backbone for the end model (default: BERT)", default="BERT")

    parser.add_argument("-emn", "--end-model-name", help="name for specific end model type eg. (Roberta for BERT)", default=None)

    parser.add_argument("-sat", "--saturate",
                        help="results with respect to which to measure the saturation point (default: oracle)",
                        default="oracle")

    parser.add_argument("-m", "--max-iter", help="Maximum number of training iterations to obtain the saturation point",
                        default=5)

    parser.add_argument("--train-val-split", type=float, default=0.8)

    parser.add_argument("--debug", help="Enable debug mode.", action="store_true")

    # handles CTRL^C signal interruption to terminate the program
    signal.signal(signal.SIGINT, signal_handler)

    args = parser.parse_args()

    if args.debug:
        jobs = 1
    else:
        jobs = args.jobs
        
    print(args.data, args.pipeline, args.label_model, args.end_model, args.end_model_name,args.backbone, args.hard_label)
    
    if args.end_model_name is not None:
        filename = get_filename(args.data, args.pipeline, args.label_model, args.end_model + "_" + args.end_model_name, args.backbone, args.hard_label)
    else:
        filename = get_filename(args.data, args.pipeline, args.label_model, args.end_model, args.backbone, args.hard_label)
        
    model_path = f"./models/{filename}.pt"

    # automatically selects max tokens for the given dataset - change token_dict as necessary
    max_tokens = token_dict[args.data] if args.data in token_dict else 512
    
    print("max tokens: {}".format(max_tokens))

    lm_search_space = json.load(open("model_search_space/{}.json".format(args.label_model)))

    if args.end_model_name is not None:
        em_search_space = json.load(open("model_search_space/{}.json".format(args.end_model+"_"+ args.end_model_name)))
    else:
        em_search_space = json.load(open("model_search_space/{}.json".format(args.end_model)))

    # warns the user if they're parallelizing experiments across more CPUs than available, which can lead to slowdowns
    if jobs > multiprocessing.cpu_count():
        raise Warning("Attempting to run more parallel experiments than CPUs available. "
                      "Consider increasing the number of CPUs!")
    # # warns the user if they are attempting to train the BERT classification model the default MLP configuration
    if args.end_model == 'BertClassifierModel' or 'Cosine':
        if args.steps != 10000:
            warnings.warn("Recommended number of steps for BertClassifierModel is 10000.")

        if args.patience < 20:
            warnings.warn("Recommended patience for BertClassifierModel is <20.")

    if args.label_model == "MajorityVoting" and args.hard_label == False:
        warnings.warn("Using soft label for MajorityVoting")

    target = data_to_target[args.data]
    if args.pipeline == "val-as-train" and args.label_model != "MajorityVoting":
        raise NotImplementedError("val-as-train pipeline is only implemented for majority voting label model.")

    #### Load dataset
    train_data, val_data, test_data = load_dataset(args.root, args.data, extract_feature=True,
                                                   extract_fn=args.extract_fn,
                                                   cache_name=args.extract_fn, model_name=args.extract_model)

    #### Initialize label model and end-model

    num_classes = train_data.n_class
    min, max = 0, 0

    results = OrderedDict()
    
    if args.pipeline == "oracle":
        pipeline = pipelines.train_strong
        indep_vars = [None]
        max_iter = 1
    elif args.pipeline == "end-to-end":
        pipeline = pipelines.train_weak
        indep_vars = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]  # % of validation used
        max_iter = len(indep_vars)
    elif args.pipeline == "val-as-train":
        pipeline = pipelines.train_strong
        train_data = None
        indep_vars = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]  # % of validation used
        max_iter = len(indep_vars)
    elif args.pipeline == "train-as-train":
        pipeline = pipelines.train_strong
        indep_vars = np.array([1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]) * len(val_data)  # % of strong samples used
        max_iter = len(indep_vars)
    elif args.pipeline == "saturation":
        pipeline = pipelines.train_strong
        min = 0
        max = 1
        indep_vars = [(max + min) / 2]
        max_iter = args.max_iter

        saturate_filename = get_filename(args.data, args.saturate, args.label_model, args.end_model, args.backbone,
                                         args.hard_label)

        with open("./results/{}/{}.json".format(args.data, saturate_filename), "r") as file:
            oracle_results = json.load(file)["em_test"][target]
    elif args.pipeline == "fine-tune-on-val":
        pipeline = pipelines.fine_tune_on_val
        indep_vars = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]  # % of validation used
        max_iter = len(indep_vars)
    else:
        raise NotImplementedError

    ix = 0
    while ix < len(indep_vars) and ix < args.max_iter:

        indep_var = indep_vars[ix]
        print("independent variable: {}".format(indep_var))

        ix += 1

        em_meter_val = AverageMeter(names=[target])
        em_meter_test = AverageMeter(names=[target])

        lm_meter_val = AverageMeter(names=[target])
        lm_meter_test = AverageMeter(names=[target])

        tuned_em_meter_val = AverageMeter(names=[target])
        tuned_em_meter_test = AverageMeter(names=[target])

        #loop over the number of runs
        result = Parallel(n_jobs=jobs, backend="loky")(delayed(
            pipeline_loader)(
            pipeline=pipeline,
            data=args.data,
            label_model=args.label_model,
            end_model=args.end_model,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            train_val_split=args.train_val_split,
            target=target,
            lm_search_space=lm_search_space,
            em_search_space=em_search_space,
            n_repeats_lm=args.num_repeat_lm,
            n_repeats_em=args.num_repeat_em,
            n_trials=args.grid_size,
            n_steps=args.steps,
            patience=args.patience,
            evaluation_step=args.evaluation_step,
            device=args.device,
            hard_label=args.hard_label,
            bb=args.backbone,
            max_tokens=max_tokens,
            indep_var=indep_var,
            seed=run_id,
            model_path=model_path
        ) for run_id in range(1, args.num_runs + 1))

        for run in result:
            # only updates the label model scores if we're working with a weak supervision pipeline
            if pipeline == pipelines.train_weak or pipeline == pipelines.fine_tune_on_val:
                lm_meter_val.update(**{target: run["lm_val"]})
                lm_meter_test.update(**{target: run["lm_test"]})

            if pipeline == pipelines.fine_tune_on_val:
                tuned_em_meter_val.update(**{target: run["tuned_em_val"]})
                tuned_em_meter_test.update(**{target: run["tuned_em_test"]})
            
            em_meter_val.update(**{target: run["em_val"]})
            em_meter_test.update(**{target: run["em_test"]})

        results[indep_var] = {
            "em_test": em_meter_test.get_results(),
            "em_val": em_meter_val.get_results(),
        }

        if pipeline == pipelines.train_weak or pipelines.fine_tune_on_val:
            results[indep_var]["lm_val"] = lm_meter_val.get_results(),
            results[indep_var]["lm_test"] = lm_meter_test.get_results()
        
        if pipeline == pipelines.fine_tune_on_val:
            results[indep_var]["tuned_em_val"] = tuned_em_meter_val.get_results(),
            results[indep_var]["tuned_em_test"] = tuned_em_meter_test.get_results()
        

        if args.pipeline == "saturation":
            if has_saturated(*results[indep_var]["em_test"][target], *oracle_results, args.num_runs):
                max = indep_var
            else:
                min = indep_var

            indep_vars.append((max + min) / 2)

        torch.cuda.empty_cache()
        gc.collect()

    if indep_vars == [None]:
        results = results[None]

    pprint(results)

    if not os.path.exists("./results"):
        os.mkdir("./results")

    count = 1
    while (1):
        if count == 1:
            num = ""
        else:
            num = "_(" + str(count) + ")"
        if not os.path.exists("./results/{}/{}{}.json".format(args.data, filename, num)):
            filename += num 
            break
        count += 1
    if not os.path.exists("./results/{}".format(args.data)):
        os.mkdir("./results/{}".format(args.data))

    with open("./results/{}/{}.json".format(args.data, filename), "w") as outfile:
        outfile.write(json.dumps(results, indent=4, default=str))