# Limit of Weak Supervision

This project is based on the wrench library. Please visit [link](https://github.com/JieyuZ2/wrench) for more detailed instructions.

## Installation
[1] Install anaconda:
Instructions here: https://www.anaconda.com/download/

[2] Clone the repository:
```
git clone https://github.com/safranchik/limits-of-ws.git
cd limits-of-ws
```

[3] Create virtual environment:
```
conda env create -f environment.yml
source activate wrench
```

[4] Download the dataset:

The datasets can be downloaded via [this](https://drive.google.com/drive/folders/1v55IKG2JN9fMtKJWU48B_5_DcPWGnpTq?usp=sharing).

Move the datasets folder to the limits-of-ws directory.

## Using the pipeline

[1] Inside the limits-of-ws directory

'''
python3 pipeline.py
'''


### Note:
If running grid search with Cosine as an end model with parallelism enablabled, the following combination of hyperparameters may cause GPU memory issue:
{"optimizer_lr" : [1e-5], "optimizer_weight_decay" : [1e-4], "batch_size" : [32], "real_batch_size" : [8],  "teacher_update" : [100], "lambda" : [0.01] "thresh" : [0.2], "margin” : [1.0], "mu" : [1.0]}
