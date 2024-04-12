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

ABSTAIN = -1

def find_entity_indices(text, entity1, entity2):
    index1 = text.find(entity1)
    index2 = text.find(entity2)
    return index1, index2

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
    data_dict = {'text': text, 'label': labels, 'entity1': entity1, 'entity2': entity2, 'span1': span1, 'span2': span2, 'weak_labels': weak_labels}
    df = pd.DataFrame(data=data_dict, index=indices)
    return df

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
            'label': row['label'],
            'data': data,
            'weak_labels': row['weak_labels']
        }
    return dataset

def save_dataset(dataset, path):
    with open(path, 'w') as file:
        json.dump(dataset, file)
        
def find_entity_word_indices(text, entity):
    words = text.split()
    entity_words = entity.split()
    indices = [i for i, word in enumerate(words) if any(entity_word in word for entity_word in entity_words)]
    return sum(indices) / len(indices) if indices else ABSTAIN

def chemprot_enhanced(df):
    if len(df) == 0:
        return df
    df = df.copy()
    df['entity1_index'] = df.apply(lambda row: calculate_accurate_word_index(row['span1'], row['text']), axis=1)
    df['entity2_index'] = df.apply(lambda row: calculate_accurate_word_index(row['span2'], row['text']), axis=1)
    return df


def calculate_accurate_word_index(span, text):
    words = text.split()
    start_index, end_index = None, None

    # Calculate the start and end positions of each word
    cumulative_pos = 0
    for index, word in enumerate(words):
        start_pos = cumulative_pos
        end_pos = cumulative_pos + len(word)

        # Check if the word or part of it is within the span
        if start_pos <= span[0] < end_pos:
            start_index = index
        if start_pos < span[1] <= end_pos:
            end_index = index

        cumulative_pos += len(word) + 1  # Add 1 for the space after each word

    if start_index is not None and end_index is not None:
        return (start_index + end_index) / 2
    return -1

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
        return label if all([k in x.text.lower() for k in keywords]) else ABSTAIN
    else:
        return label if keyword in x.text.lower() else ABSTAIN

# Allows us to convert from a keyword_dict {class: [keyword list]} to a set of LFs 
def keywords_to_LFs(keyword_dict):
    lfs = []
    for l, v in keyword_dict.items():
        for k in v:
            lfs.append(LabelingFunction(name=f"lf_{k}", f=_keyword_LF, resources={'keyword':k, 'label':l}))
    return lfs

def chemprot_df_with_new_lf(df, lfs):
    df = df.copy()
    #enhance df
    df = chemprot_enhanced(df)
    #apply new lfs
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df)
    # add the new lfs to the df's weaklabels column
    df['weak_labels'] = L_train.tolist()
    # delete the added two column from chemprot_enhanced:
    df = df.drop(columns=['entity1_index', 'entity2_index'])
    return df


def see_label_function(df, lfs):
    length = len(df)
    df = df.copy()
    df = chemprot_enhanced(df)
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df)
    df = df.drop(columns=['entity1_index', 'entity2_index'])
    # if the label function is -1 then drop that row of the df
    drop_indices = [i for i in range(len(L_train)) if (L_train[i] == -1)]
    df = df.drop(df.iloc[drop_indices].index)
    return df, len(df)


#  and df.iloc[i]['label']!=0
