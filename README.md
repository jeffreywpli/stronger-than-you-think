# stronger-than-you-think

This project focuses on idenitifying real-life datasets that weak supervision excells at. It is based on the [`WRENCH`](https://github.com/JieyuZ2/wrench). Some of the pipeline design are consistent with paper [Weaker Than You Think: A Critical Look at Weakly Supervised Learning](https://arxiv.org/pdf/2305.17442.pdf).


## Installation

```
conda env create -f environment.yml
source activate stronger-than-uthink
```

```
pip install git+https://github.com/openai/CLIP.git
```

## Structure of the project

### Pipeline
The **`val_size_experiment.py`** file handles the user argument of the experiments and calls corresponding pipelines in **`pipelines.py`** to run the experiments.

The pipeline currently has the following experiments:

[1] `oracle`

It uses all the training data of the dataset with ground truth labels and trains a supervised model on it.

[2] `end-to-end`

It uses all the labeled validation data and unlabeled training data. It first trains a weakly supervised model to create weak labels for all the training data, and aggregates to get strong labels that get piped into a fully supervised end model.

[3] `val-as-train`

It takes a range of numbers/proportions of the labeled validation data and trains a supervised model just on those subsampled validation data.

[4] `train-as-train`

It takes a range of numbers/proportions of the training data (with ground truth), and trains a supervised model just on those subsampled validation data.

[5] `saturation`

These experiments require `orcale` experiment of the same dataset and supervised model to be run first.

Then it performs a binary search on the number/proportion of labelled data required to match the oracle performance.

[6] `fine-tune-on-val`

This experiment is based on the `end-to-end` experiment, and performs an additional step of fine-tuning on the trained supervised model with the labeled validation dataset.


### Search space

The **`model_search_space`** folder stores hyperparameter search space for each model (both weak and supervised) in `json` file. It also stores parameters for variation of model implemented in   `WRENCH` (see RoBERTa vs. BERT as an example)


## How to run experiments
[1] 

Put the dataset in the `WRENCH` format. 

Note that additional modifications to `WRENCH` dataset files in order to correctly import the additional dataset.

[2] 

Add/Modify the `json` file in the **`model_search_space`** folder with corresponding parameters.

[3]

```
python val_size_experiment.py
```
### Major arguments

`--data`, `--pipeline`, `--label-model`, `--end-model`, `--end-model-name`
`--num-runs`, `--fix-hyperparam`, `--fix-steps`

### Examples of codes

[1]

Running the continous fine tuning experiment on ChemProt dataset, on RoBERTa model, with validaton dataset size of 50 per class, fixed step size of 6000, and hyperparameter searching for the first phase.

```
python3 val_size_experiment.py -p fine-tune-on-val -d chemprot -em BertClassifierModel -emn roberta -vnpc 50 -fixStep 6000
```

[2]
Running the supervised model on the clear validation data of AGNews dataset, on RoBERTa model, with validation dataset size of  5 per class, and fixed step size of 6000. There is no hyperparameter searching, and each hyperparameter is randomly chosen from the list in the hyperparameter search space called `BertClassifierModel_roberta.json`

```
python3 val_size_experiment.py -p val-as-train -d agnews -em BertClassifierModel -emn roberta -vnpc 5 -fixStep 6000 -fixHyper
```

### Note:
[1] If running grid search with Cosine as an end model with parallelism enabled, the following combination of hyperparameters may cause GPU memory issues:
{"optimizer_lr" : [1e-5], "optimizer_weight_decay" : [1e-4], "batch_size" : [32], "real_batch_size" : [8],  "teacher_update" : [100], "lambda" : [0.01] "thresh" : [0.2], "margin” : [1.0], "mu" : [1.0]}

