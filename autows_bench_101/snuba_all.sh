#!/bin/bash

# Snuba sweep over all datasets and embeddings
resdir=results/neurips2022/snuba_multiclass
mkdir -p ${resdir}

snuba_cardinality=1
n_labeled_points=100
lf_selector=snuba_multiclass
snuba_iterations=3
snuba_combo_samples=-1

for seed in 0
do 
    for emb in raw pca resnet18
    do 
        for dataset in mnist cifar10 spherical_mnist permuted_mnist ecg navier_stokes ember
        do

            savedir=${resdir}/${seed}/${emb}/${dataset}
            mkdir -p ${savedir}

            CUDA_VISIBLE_DEVICES=0 python -u fwrench/applications/pipeline.py \
                --lf_selector=${lf_selector} \
                --n_labeled_points=${n_labeled_points} \
                --snuba_cardinality=${snuba_cardinality} \
                --snuba_iterations=${snuba_iterations} \
                --snuba_combo_samples=${snuba_combo_samples} \
                --seed=${seed} \
                --embedding=${emb} \
                --dataset=${dataset} \
                |& tee -a ${savedir}/res_seed${seed}.log
        done
    done
done
