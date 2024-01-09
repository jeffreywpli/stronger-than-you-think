import logging
import torch
import numpy as np
import fire
import sklearn
from functools import partial
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import jaccard_score, accuracy_score
from fwrench.embeddings.vae_embedding import VAE2DEmbedding
from fwrench.embeddings.resnet_embedding import ResNet18Embedding

from wrench.dataset import load_dataset
from wrench.logging import LoggingHandler
from wrench.evaluation import f1_score_
from wrench.labelmodel import MajorityVoting, FlyingSquid, Snorkel
from wrench.endmodel import EndClassifierModel, MLPModel
from fwrench.lf_selectors import SnubaSelector, AutoSklearnSelector
from fwrench.embeddings import *
from fwrench.datasets import CIFAR10Dataset

import utils

def main(data_dir='CIFAR10_2500', 
        dataset_home='./datasets',
        even_odd=False,
        embedding='resnet18', # raw | pca | resnet18 | vae
        scoring_fn=None, # TODO
        lf_class_options='default', # default | comma separated list of lf classes to use in the selection procedure. Example: 'DecisionTreeClassifier,LogisticRegression'
        lf_selector='snuba', # snuba | interactive | goggles
        em_hard_labels=True, # Use hard labels in the end model
        n_labeled_points=100,
        seed=123, # TODO
        ):

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    logger = logging.getLogger(__name__)
    device = torch.device('cuda')

    train_data = CIFAR10Dataset('train', name='CIFAR10')
    valid_data = CIFAR10Dataset('valid', name='CIFAR10')
    test_data = CIFAR10Dataset('test', name='CIFAR10')

    data = data_dir
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, 
        extract_feature=True,
        dataset_type='NumericDataset')

    # Create subset of labeled dataset
    valid_data = valid_data.create_subset(np.arange(n_labeled_points))

    binary_mode = even_odd
    if even_odd:
        train_data = utils.convert_to_even_odd(train_data)
        valid_data = utils.convert_to_even_odd(valid_data)
        test_data = utils.convert_to_even_odd(test_data)
        #train_data = utils.convert_0_1(train_data)
        #valid_data = utils.convert_0_1(valid_data)
        #test_data = utils.convert_0_1(test_data)
    
    # TODO also hacky... normalize MNIST data because it comes unnormalized
    train_data = utils.normalize01(train_data)
    valid_data = utils.normalize01(valid_data)
    test_data = utils.normalize01(test_data)

    # Dimensionality reduction...
    if embedding == 'raw': 
        embedder = FlattenEmbedding() 
    elif embedding == 'pca':
        emb = PCA(n_components=100)
        embedder = SklearnEmbedding(emb)
    elif embedding == 'resnet18':
        embedder = ResNet18Embedding()
    elif embedding == 'vae':
        embedder = VAE2DEmbedding()
    else:
        raise NotImplementedError

    embedder.fit(train_data, valid_data, test_data)
    train_data_embed = embedder.transform(train_data)
    valid_data_embed = embedder.transform(valid_data)
    test_data_embed = embedder.transform(test_data)

    # Fit Snuba with multiple LF function classes and a custom scoring function
    if lf_class_options == 'default':
        lf_classes = [
            #partial(DecisionTreeClassifier, max_depth=1),
            LogisticRegression
            ]
    else:
        if not isinstance(lf_class_options, tuple):
            lf_class_options = [lf_class_options]
        lf_classes = []
        logger.info(lf_class_options)
        for lf_cls in lf_class_options:
            if lf_cls == 'DecisionTreeClassifier':
                lf_classes.append(partial(DecisionTreeClassifier, max_depth=1))
            elif lf_cls == 'LogisticRegression':
                lf_classes.append(LogisticRegression)
            else: 
                # If the lf class you need isn't implemented, add it here
                raise NotImplementedError
    logger.info(f'Using LF classes: {lf_classes}')

    scoring_fn = None #accuracy_score
    if lf_selector == 'snuba':
        snuba = SnubaSelector(lf_classes, scoring_fn=scoring_fn)
        # Use Snuba convention of assuming only validation set labels...
        snuba.fit(valid_data_embed, train_data_embed, 
            b=0.1 if not binary_mode else 0.5, # TODO
            #b=0.9,
            cardinality=1, iters=23)
        logger.info(snuba.hg.heuristic_stats())
        # NOTE that snuba uses different F1 score implementations in 
        # different places... 
        # In it uses average='weighted' for computing abstain thresholds
        # and average='micro' for pruning... 
        # Maybe we should try different choices in different places as well?
    else: 
        raise NotImplementedError

    train_weak_labels = snuba.predict(train_data_embed)
    train_data.weak_labels = train_weak_labels.tolist()
    valid_weak_labels = snuba.predict(valid_data_embed)
    valid_data.weak_labels = valid_weak_labels.tolist()
    test_weak_labels = snuba.predict(test_data_embed)
    test_data.weak_labels = test_weak_labels.tolist()

    # NOTE MV code sometimes crashes for unknown reasons
    # Get score from majority vote
    #label_model = MajorityVoting()
    #label_model.fit(
    #    dataset_train=train_data,
    #    dataset_valid=valid_data
    #)
    #logger.info(f'---Majority Vote eval---')
    #acc = label_model.test(train_data, 'acc')
    #logger.info(f'label model (MV) train acc:    {acc}')
    #acc = label_model.test(valid_data, 'acc')
    #logger.info(f'label model (MV) valid acc:    {acc}')
    #acc = label_model.test(test_data, 'acc')
    #logger.info(f'label model (MV) test acc:     {acc}')

    # Get score from Snorkel (afaik, this is the default Snuba LM)
    label_model = Snorkel()
    label_model.fit(
        dataset_train=train_data,
        dataset_valid=valid_data
    )

    # Train end model
    #### Filter out uncovered training data
    train_data_covered = train_data#.get_covered_subset()
    aggregated_hard_labels = label_model.predict(train_data_covered)
    aggregated_soft_labels = label_model.predict_proba(train_data_covered)


    # Get actual label model accuracy using hard labels
    utils.get_accuracy_coverage(train_data, label_model, logger, split='train')
    utils.get_accuracy_coverage(valid_data, label_model, logger, split='valid')
    utils.get_accuracy_coverage(test_data, label_model, logger, split='test')

    coverage = aggregated_soft_labels.shape[0] / len(train_data_embed.labels)
    logger.info(f'coverage = {coverage:.4f}')

    # Does it do better than just training on the validation labels?
    model = EndClassifierModel(
        batch_size=256,
        test_batch_size=512,
        n_steps=1_000, # Increase this to 100_000 if needed
        backbone='LENET',
        optimizer='SGD',
        optimizer_lr=1e-1,
        optimizer_weight_decay=0.0,
        binary_mode=binary_mode,
    )
    model.fit(
        dataset_train=train_data_covered,
        y_train=aggregated_hard_labels if em_hard_labels \
            else aggregated_soft_labels,
        dataset_valid=valid_data,
        evaluation_step=50,
        metric='acc',
        patience=1000,
        device=device
    )
    logger.info(f'---LeNet eval---')
    acc = model.test(test_data, 'acc')
    logger.info(f'end model (LeNet) test acc:    {acc}')

if __name__ == '__main__':
    fire.Fire(main)
