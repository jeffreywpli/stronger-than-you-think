import numpy as np
import snorkel
from snorkel.labeling import labeling_function
from snorkel.labeling import LabelingFunction
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter
import json
import pandas as pd
import label_improve

# def wrench_to_df(dataset):
#     indices = [int(i) for i in dataset.keys()]
#     text = [ dataset[str(i)]['data']['text'] for i in range(len(indices))]    
#     labels = [ dataset[str(i)]['label'] for i in range(len(indices))]    
#     data_dict = {'text': text, 'labels': labels}
#     df = pd.DataFrame(data=data_dict, index=indices)
#     return df
    
def wrench_to_df(dataset):
    indices = [int(i) for i in dataset.keys()]
    text = [dataset[i]['data']['text'] for i in dataset.keys()]    
    labels = [dataset[i]['label'] for i in dataset.keys()]    
    data_dict = {'text': text, 'labels': labels}
    df = pd.DataFrame(data=data_dict, index=indices)
    return df

def chemprot_to_df(dataset):
    indices = [int(i) for i in dataset.keys()]
    text = [dataset[i]['data']['text'] for i in dataset.keys()]    
    labels = [dataset[i]['label'] for i in dataset.keys()]    
    entity1 = [dataset[i]['data']['entity1'] for i in dataset.keys()]    
    entity2 = [dataset[i]['data']['entity2'] for i in dataset.keys()]    
    span1 = [dataset[i]['data']['span1'] for i in dataset.keys()]    
    span2 = [dataset[i]['data']['span2'] for i in dataset.keys()]    
    weak_labels = [dataset[i]['weak_labels'] for i in dataset.keys()]    
    data_dict = {'text': text, 'labels': labels, 'entity1': entity1, 'entity2': entity2, 'span1': span1, 'span2': span2, 'weak_labels': weak_labels}
    df = pd.DataFrame(data=data_dict, index=indices)
    return df

# Calculates coverage given a label matrix
def calc_coverage(L):
    return (L.max(axis=1) > -1).mean()

# Applies a set lfs (functions) to a dataset (in df form)
def apply_LFs(lfs, dataset):
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=dataset)
    return L_train

# Helper function for converting an individual keyowrd into an LF
def _keyword_LF(x, keyword=None, label=None):
    # The + allows for conjunctions of keyword LFs
    if "+" in keyword:
        keywords = keyword.split("+")
        return label if all([k in x.text.lower() for k in keywords]) else -1 
    else:
        return label if keyword in x.text.lower() else -1

# Allows us to convert from a keyword_dict {class: [keyword list]} to a set of LFs 
def keywords_to_LFs(keyword_dict):
    lfs = []
    for l, v in keyword_dict.items():
        for k in v:
            lfs.append(LabelingFunction(name=f"lf_{k}", f=_keyword_LF, resources={'keyword':k, 'label':l}))
    return lfs

def find_entity_indices(text, entity1, entity2):
    index1 = text.find(entity1)
    index2 = text.find(entity2)
    return index1, index2

def df_to_chemprot(df):
    dataset = {}
    for index, row in df.iterrows():
        data = {
            'text': row['text'],
            'entity1': row['entity1'],
            'entity2': row['entity2'],
            'span1': row['span1'],
            'span2': row['span2']
        }
        dataset[str(index)] = {
            'label': row['labels'],
            'data': data,
            'weak_labels': row['weak_labels']
        }
    return dataset

def save_dataset(dataset, path):
    with open(path, 'w') as file:
        json.dump(dataset, file)