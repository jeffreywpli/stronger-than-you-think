# stronger-than-you-think


## Installation

```
conda env create -f environment.yml
source activate stronger-than-uthink
```

```
pip install git+https://github.com/openai/CLIP.git
```

# End_model_training



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
[1] Put the dataset in the `WRENCH` format. 

Note that additional modifications to `WRENCH` dataset files in order to correctly import the additional dataset.

[2] Add/Modify the `json` file in the **`model_search_space`** folder with corresponding parameters.

[3]

```
python val_size_experiment.py
```
### Major arguments

`--data`, `--pipeline`, `--label-model`, `--end-model`,
`--num-runs`

### Note:
[1] If running grid search with Cosine as an end model with parallelism enabled, the following combination of hyperparameters may cause GPU memory issues:
{"optimizer_lr" : [1e-5], "optimizer_weight_decay" : [1e-4], "batch_size" : [32], "real_batch_size" : [8],  "teacher_update" : [100], "lambda" : [0.01] "thresh" : [0.2], "margin” : [1.0], "mu" : [1.0]}

