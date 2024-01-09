from ctypes import util
from fwrench.datasets.torchvision_dataset import CIFAR10Dataset, FashionMNISTDataset
import fwrench.utils as utils
import numpy as np
from fwrench.datasets import (
    MNISTDataset,
    FashionMNISTDataset,
    KMNISTDataset,
    SphericalDataset,
    CIFAR10Dataset,
    ECGTimeSeriesDataset,
    EmberDataset,
    NavierStokesDataset,
)
from wrench.dataset import load_dataset
from wrench.endmodel import EndClassifierModel
import os, shutil


def get_mnist(
    n_labeled_points, dataset_home, data_dir="MNIST_3000",
):

    train_data = MNISTDataset("train", name="MNIST")
    valid_data = MNISTDataset("valid", name="MNIST")
    test_data = MNISTDataset("test", name="MNIST")
    n_classes = 10

    data = data_dir
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, extract_feature=True, dataset_type="NumericDataset"
    )

    # Create subset of labeled dataset
    valid_data = valid_data.create_subset(np.arange(n_labeled_points))

    # TODO also hacky...
    # normalize data because it comes unnormalized apparently...
    train_data = utils.normalize01(train_data)
    valid_data = utils.normalize01(valid_data)
    test_data = utils.normalize01(test_data)

    # Create end model
    model = EndClassifierModel(
        batch_size=256,
        test_batch_size=512,
        n_steps=1_000,
        backbone="LENET",
        optimizer="SGD",
        optimizer_lr=1e-1,
        optimizer_weight_decay=0.0,
        binary_mode=False,
    )

    return train_data, valid_data, test_data, n_classes, model


def get_fashion_mnist(
    n_labeled_points, dataset_home, data_dir="FashionMNIST_3000",
):

    train_data = FashionMNISTDataset("train", name="FashionMNIST")
    valid_data = FashionMNISTDataset("valid", name="FashionMNIST")
    test_data = FashionMNISTDataset("test", name="FashionMNIST")
    n_classes = 10

    data = data_dir
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, extract_feature=True, dataset_type="NumericDataset"
    )

    # Create subset of labeled dataset
    valid_data = valid_data.create_subset(np.arange(n_labeled_points))

    # TODO also hacky...
    # normalize data because it comes unnormalized apparently...
    train_data = utils.normalize01(train_data)
    valid_data = utils.normalize01(valid_data)
    test_data = utils.normalize01(test_data)

    # Create end model
    model = EndClassifierModel(
        batch_size=256,
        test_batch_size=512,
        n_steps=1_000,
        backbone="LENET",
        optimizer="SGD",
        optimizer_lr=1e-1,
        optimizer_weight_decay=0.0,
        binary_mode=False,
    )

    return train_data, valid_data, test_data, n_classes, model


def get_kmnist(
    n_labeled_points, dataset_home, data_dir="KMNIST_3000",
):

    train_data = KMNISTDataset("train", name="KMNIST")
    valid_data = KMNISTDataset("valid", name="KMNIST")
    test_data = KMNISTDataset("test", name="KMNIST")
    n_classes = 10

    data = data_dir
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, extract_feature=True, dataset_type="NumericDataset"
    )

    # Create subset of labeled dataset
    valid_data = valid_data.create_subset(np.arange(n_labeled_points))

    # TODO also hacky...
    # normalize data because it comes unnormalized apparently...
    train_data = utils.normalize01(train_data)
    valid_data = utils.normalize01(valid_data)
    test_data = utils.normalize01(test_data)

    # Create end model
    model = EndClassifierModel(
        batch_size=256,
        test_batch_size=512,
        n_steps=1_000,
        backbone="LENET",
        optimizer="SGD",
        optimizer_lr=1e-1,
        optimizer_weight_decay=0.0,
        binary_mode=False,
    )

    return train_data, valid_data, test_data, n_classes, model


def get_cifar10(
    n_labeled_points, dataset_home, data_dir="CIFAR10_2500",
):

    train_data = CIFAR10Dataset("train", name="CIFAR10")
    valid_data = CIFAR10Dataset("valid", name="CIFAR10")
    test_data = CIFAR10Dataset("test", name="CIFAR10")
    n_classes = 10
    data = data_dir
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, extract_feature=True, dataset_type="NumericDataset"
    )

    # Create subset of labeled dataset
    valid_data = valid_data.create_subset(np.arange(n_labeled_points))

    # TODO also hacky...
    # normalize KMNIST data because it comes unnormalized apparently...
    train_data = utils.normalize01(train_data)
    valid_data = utils.normalize01(valid_data)
    test_data = utils.normalize01(test_data)

    # Create end model
    # TODO Change to ResNet or s2cnn
    model = EndClassifierModel(
        batch_size=256,
        test_batch_size=512,
        n_steps=1_000,
        backbone="LENET",
        optimizer="SGD",
        optimizer_lr=1e-1,
        optimizer_weight_decay=0.0,
        binary_mode=False,
    )

    return train_data, valid_data, test_data, n_classes, model


def get_spherical_mnist(
    n_labeled_points, dataset_home, data_dir="SphericalMNIST_3000",
):

    train_data = SphericalDataset("train", name="SphericalMNIST")
    valid_data = SphericalDataset("valid", name="SphericalMNIST")
    test_data = SphericalDataset("test", name="SphericalMNIST")
    n_classes = 10

    data = data_dir
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, extract_feature=True, dataset_type="NumericDataset"
    )

    # Create subset of labeled dataset
    valid_data = valid_data.create_subset(np.arange(n_labeled_points))

    # TODO also hacky...
    # normalize data because it comes unnormalized apparently...
    train_data = utils.normalize01(train_data)
    valid_data = utils.normalize01(valid_data)
    test_data = utils.normalize01(test_data)

    # Create end model
    # TODO Change to ResNet or s2cnn
    model = EndClassifierModel(
        batch_size=256,
        test_batch_size=512,
        n_steps=1_000,
        backbone="LENET",
        optimizer="SGD",
        optimizer_lr=1e-1,
        optimizer_weight_decay=0.0,
        binary_mode=False,
    )

    return train_data, valid_data, test_data, n_classes, model


def get_permuted_mnist(
    n_labeled_points, dataset_home, data_dir="MNIST_3000",
):

    train_data = MNISTDataset("train", name="MNIST")
    valid_data = MNISTDataset("valid", name="MNIST")
    test_data = MNISTDataset("test", name="MNIST")
    n_classes = 10

    data = data_dir
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, extract_feature=True, dataset_type="NumericDataset"
    )

    # Create subset of labeled dataset
    valid_data = valid_data.create_subset(np.arange(n_labeled_points))

    # TODO also hacky...
    # normalize data because it comes unnormalized apparently...
    train_data = utils.normalize01(train_data)
    valid_data = utils.normalize01(valid_data)
    test_data = utils.normalize01(test_data)

    # Permute the images
    train_data = utils.row_col_permute(train_data)
    valid_data = utils.row_col_permute(valid_data)
    test_data = utils.row_col_permute(test_data)

    # Create end model
    model = EndClassifierModel(
        batch_size=256,
        test_batch_size=512,
        n_steps=1_000,
        backbone="LENET",
        optimizer="SGD",
        optimizer_lr=1e-1,
        optimizer_weight_decay=0.0,
        binary_mode=False,
    )

    return train_data, valid_data, test_data, n_classes, model

def get_ecg(
    n_labeled_points, dataset_home, data_dir="ECG_14752",
):

    train_data = ECGTimeSeriesDataset("train", name="ECG")
    valid_data = ECGTimeSeriesDataset("valid", name="ECG")
    test_data = ECGTimeSeriesDataset("test", name="ECG")
    n_classes = 4

    data = data_dir
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, extract_feature=True, dataset_type="NumericDataset"
    )

    # Create subset of labeled dataset
    valid_data = valid_data.create_subset(np.arange(n_labeled_points))

    # Create end model
    model = EndClassifierModel(
        batch_size=256,
        test_batch_size=512,
        n_steps=1_000,
        backbone="LENET",
        optimizer="SGD",
        optimizer_lr=1e-1,
        optimizer_weight_decay=0.0,
        binary_mode=False,
    )

    return train_data, valid_data, test_data, n_classes, model

def get_ember_2017(
    # 600000 training / 200000 testing
    # sometimes work,sometimes get killed. Maybe only implement a subset?
    n_labeled_points, dataset_home, data_dir="ember_2017_15000",
):

    train_data = EmberDataset('train', name='ember_2017')
    valid_data = EmberDataset('valid', name='ember_2017')
    test_data = EmberDataset('test', name='ember_2017')

    # could remove the tar file after loading 
    # os.remove('ember_dataset_2017_2.tar.bz2')
    # shutil.rmtree('ember_2017_2')

    n_classes = 2

    data = data_dir
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, extract_feature=True, dataset_type="NumericDataset"
    )

    # Create subset of labeled dataset
    # train_data = train_data.create_subset(np.arange(3000))
    # test_data = test_data.create_subset(np.arange(1000))
    valid_data = valid_data.create_subset(np.arange(n_labeled_points))

    # Create end model
    model = EndClassifierModel(
        batch_size=256,
        test_batch_size=512,
        n_steps=1_000,
        backbone="LENET",
        optimizer="SGD",
        optimizer_lr=1e-1,
        optimizer_weight_decay=0.0,
        binary_mode=False,
    )

    return train_data, valid_data, test_data, n_classes, model

def get_navier_stokes(
    n_labeled_points, dataset_home, data_dir="NavierStokes_100",
):
    
    train_data = NavierStokesDataset("train", name="NavierStokes")
    valid_data = NavierStokesDataset("valid", name="NavierStokes")
    test_data = NavierStokesDataset("test", name="NavierStokes")
    n_classes = 2
    
    data = data_dir
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, extract_feature=True, dataset_type="NumericDataset"
    )
    
    # Create subset of labeled dataset
    valid_data = valid_data.create_subset(np.arange(n_labeled_points))

    # Create end model
    model = EndClassifierModel(
        batch_size=256,
        test_batch_size=512,
        n_steps=1_000,
        backbone="LENET",
        optimizer="SGD",
        optimizer_lr=1e-1,
        optimizer_weight_decay=0.0,
        binary_mode=False,
    )

    return train_data, valid_data, test_data, n_classes, model

def get_imdb(
    n_labeled_points, dataset_home,  extract_fn, data_dir="imdb",
):
    n_classes = 2
    data = data_dir
    extract_feature=(extract_fn != None)
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, extract_feature=extract_feature, extract_fn= extract_fn,
        cache_name=extract_fn, dataset_type="TextDataset"
    )

    valid_data = valid_data.create_subset(np.arange(n_labeled_points))
    if extract_fn is not None:
        train_data = utils.convert_text_to_feature(train_data)
        valid_data = utils.convert_text_to_feature(valid_data)
        test_data = utils.convert_text_to_feature(test_data)

    # Create end model
    model = EndClassifierModel(
        batch_size=256,
        test_batch_size=512,
        n_steps=1_000,
        backbone="LENET",
        optimizer="SGD",
        optimizer_lr=1e-1,
        optimizer_weight_decay=0.0,
        binary_mode=False,
    )

    return train_data, valid_data, test_data, n_classes, model

def get_yelp(
    n_labeled_points, dataset_home,  extract_fn, data_dir="yelp",
):
    n_classes = 2
    
    data = data_dir
    extract_feature=(extract_fn != None)
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, extract_feature=extract_feature, extract_fn= extract_fn,
        cache_name=extract_fn, dataset_type="TextDataset"
    )

    valid_data = valid_data.create_subset(np.arange(n_labeled_points))
    if extract_fn is not None:
        train_data = utils.convert_text_to_feature(train_data)
        valid_data = utils.convert_text_to_feature(valid_data)
        test_data = utils.convert_text_to_feature(test_data)

    # Create end model
    model = EndClassifierModel(
        batch_size=256,
        test_batch_size=512,
        n_steps=1_000,
        backbone="LENET",
        optimizer="SGD",
        optimizer_lr=1e-1,
        optimizer_weight_decay=0.0,
        binary_mode=False,
    )

    return train_data, valid_data, test_data, n_classes, model

def get_youtube(
    n_labeled_points, dataset_home,  extract_fn, data_dir="youtube",
):
    n_classes = 2
    
    data = data_dir
    extract_feature=(extract_fn != None)
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, extract_feature=extract_feature, extract_fn= extract_fn,
        cache_name=extract_fn, dataset_type="TextDataset"
    )
    valid_data = valid_data.create_subset(np.arange(n_labeled_points))
    
    if extract_fn is not None:
        train_data = utils.convert_text_to_feature(train_data)
        valid_data = utils.convert_text_to_feature(valid_data)
        test_data = utils.convert_text_to_feature(test_data)

    # Create end model
    model = EndClassifierModel(
        batch_size=256,
        test_batch_size=512,
        n_steps=1_000,
        backbone="LENET",
        optimizer="SGD",
        optimizer_lr=1e-1,
        optimizer_weight_decay=0.0,
        binary_mode=False,
    )

    return train_data, valid_data, test_data, n_classes, model
    