import numpy as np
import snorkel
from snorkel.labeling import labeling_function
from snorkel.labeling import LabelingFunction
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter
import json
import pandas as pd
from googletrans import Translator
from tqdm import tqdm
from deep_translator import GoogleTranslator
import deepl
import concurrent.futures


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

def massive_to_df(dataset):
    indices = [int(i) for i in dataset.keys()]
    text = [ dataset[str(i)]['data']['text'] for i in range(len(indices))]    
    labels = [ dataset[str(i)]['label'] for i in range(len(indices))]    
    weak_labels = [dataset[i]['weak_labels'] for i in dataset.keys()]    
    data_dict = {'text': text, 'label': labels, 'weak_labels': weak_labels}
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

def df_to_massive(df):
    dataset = {}
    for index, row in df.iterrows():
        data = {
            'text': row['text']
        }
        dataset[str(index)] = {
            'label': row['label'],
            'data': data,
            'weak_labels': row['weak_labels']
        }
    return dataset

def save_dataset(dataset, path):
    with open(path, 'w') as file:
        json.dump(dataset, file, indent=4, ensure_ascii=False)
        
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

def massive_df_with_new_lf(df, lfs):
    df = df.copy()
    #apply new lfs
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df)
    # add the new lfs to the df's weaklabels column
    df['weak_labels'] = L_train.tolist()
    # delete the added two column from chemprot_enhanced:
    return df

def df_to_df_with_new_lf(df_target, df_source, source_lfs):
    df_target = df_target.copy()
    df_source = df_source.copy()
    applieer = PandasLFApplier(lfs=source_lfs)
    L_train = applieer.apply(df_source)
    df_target['weak_labels'] = L_train.tolist()
    return df_target

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


# utility

def df_with_label(df, label):
    df = df.copy()
    # check the weak_labels column, if label is in the list then preserve the row
    df = df[df['weak_labels'].apply(lambda x: label in x)]
    return df

# translation

def translate_text(text, dest_language):
    translator = Translator()
    translation = translator.translate(text, dest=dest_language)
    return translation.text

def translate_src(text, src_language):
    translator = Translator()
    translation = translator.translate(text, src=src_language)
    return translation.text

# translate df's text to english

# def translate_df_to_english(df, src_language):
#     tqdm.pandas()
#     df = df.copy()
#     # df['text'] = df['text'].apply(lambda x: translate_src(x,  src_language))
#     df['text'] = df['text'].progress_apply(lambda x: translate_src(x,  src_language))
#     return df

def translate_df_to_english(df, src_language, batch_size=25):
    df = df.copy()
    texts = df['text'].astype(str).tolist()  # Ensure all texts are strings
    translator = GoogleTranslator(source=src_language, target='en')
    
    translated_texts = []
    
    # Create a progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
        batch = texts[i:i + batch_size]
        try:
            translations = translator.translate_batch(batch)
            translated_texts.extend(translations)
        except Exception as e:
            print(f"Error: {e}. Some texts may not be translated.")
            translated_texts.extend([None] * len(batch))  # Fill with None if translation fails

    df['text'] = translated_texts
    return df


# def translate_df_to_english(df, src_language, batch_size=100):
#     df = df.copy()
#     texts = df['text'].tolist()
#     translator = Translator()
    
#     translated_texts = []
    
#     # Create a progress bar
#     for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
#         batch = texts[i:i + batch_size]
#         translations = translator.translate(batch, src=src_language, dest='en')
#         translated_texts.extend([translation.text for translation in translations])
    
#     df['text'] = translated_texts
#     return df


# translate df's text to another language

def translate_df(df, dest_language):
    df = df.copy()
    df['text'] = df['text'].apply(lambda x: translate_text(x, dest_language))
    return df


def deep_translate_df_to_english(df, auth_key, src_language, batch_size=50, max_workers=5):
    df = df.copy()
    texts = df['text'].astype(str).tolist()  # Ensure all texts are strings
    translator = deepl.Translator(auth_key)
    
    translated_texts = [None] * len(texts)

    def translate_batch(start_index, batch):
        try:
            translations = translator.translate_text(batch, source_lang=src_language, target_lang='EN-US')
            return (start_index, [translation.text for translation in translations])
        except Exception as e:
            print(f"Error: {e}. Some texts may not be translated.")
            return (start_index, [None] * len(batch))

    # Create a progress bar
    with tqdm(total=len(texts), desc="Translating") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(translate_batch, i, texts[i:i + batch_size]): i for i in range(0, len(texts), batch_size)}
            for future in concurrent.futures.as_completed(future_to_index):
                batch_index, translated_batch = future.result()
                translated_texts[batch_index:batch_index + len(translated_batch)] = translated_batch
                pbar.update(len(translated_batch))

    df['text'] = translated_texts
    return df


def deep_translate_df(df, auth_key, src_language='ZH', target_language='EN-US', batch_size=50, max_workers=5):
    df = df.copy()
    texts = df['text'].astype(str).tolist()  # Ensure all texts are strings
    translator = deepl.Translator(auth_key)
    
    translated_texts = [None] * len(texts)

    def translate_batch(start_index, batch):
        try:
            translations = translator.translate_text(batch, source_lang=src_language, target_lang=target_language)
            return (start_index, [translation.text for translation in translations])
        except Exception as e:
            print(f"Error: {e}. Some texts may not be translated.")
            return (start_index, [None] * len(batch))

    # Create a progress bar
    with tqdm(total=len(texts), desc="Translating") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(translate_batch, i, texts[i:i + batch_size]): i for i in range(0, len(texts), batch_size)}
            for future in concurrent.futures.as_completed(future_to_index):
                batch_index, translated_batch = future.result()
                translated_texts[batch_index:batch_index + len(translated_batch)] = translated_batch
                pbar.update(len(translated_batch))

    df['text'] = translated_texts
    return df














def analysis_LFs(lfs, df, class_size):
    L_dev = apply_LFs(lfs, df)
    print("Test Coverage:", calc_coverage(L_dev))
    lf_analysis = LFAnalysis(L_dev, lfs = lfs).lf_summary(Y = df.label.values)
    lf_analysis['Conflict Ratio'] = lf_analysis['Conflicts'] / lf_analysis['Coverage']
    majority_model = MajorityLabelVoter(class_size)
    preds_valid = majority_model.predict(L=L_dev)
    print("acuracy for the not abstains")
    print((preds_valid[preds_valid != -1] == df[preds_valid != -1].label.values).mean())
    print("acuracy for all")
    print((preds_valid == df.label.values).mean())
    return lf_analysis
    
def analysis_LFs_with_weak_labels(df, class_size):
    L_dev = np.array(df.weak_labels.tolist())
    print("Test Coverage:", calc_coverage(L_dev))
    majority_model = MajorityLabelVoter(class_size)
    preds_valid = majority_model.predict(L=L_dev)
    print("acuracy for the not abstains")
    print((preds_valid[preds_valid != -1] == df[preds_valid != -1].label.values).mean())
    print("acuracy for all")
    print((preds_valid == df.label.values).mean())