#!/bin/bash

# Set the GPU index
gpu_index="3"
export CUDA_VISIBLE_DEVICES=$gpu_index

# Array of parameterization types
parameterizations=("best" "exp" "direct" "softplus")

# Array of learning rates
learning_rates=(5e-0 5e-1 5e-2 5e-3 5e-4 5e-5 5e-6)

# Number of repetitions for each experiment
num_repetitions=3

# Loop over repetitions
for (( rep=1; rep<=num_repetitions; rep++ )); do
    # Loop over learning rates
    for lr in "${learning_rates[@]}"; do
        # Loop over parameterizations
        for param in "${parameterizations[@]}"; do
            echo "Running experiment with parameterization: $param, learning rate: $lr, repetition: $rep"
            python src/train.py \
                experiment=MNIST/ssm \
                task_name="MNIST_StableSSM_Run${rep}_LR${lr}_${param}" \
                trainer.devices=1 \
                model.net.parameterization="$param" \
                logger=many_loggers \
                model.optimizer.lr=$lr \
                seed=$rep \
                model.net.n_layers=4 \
                logger.wandb.project="StableSSM_MNIST_SSM_4Layers"
        done
    done
done
