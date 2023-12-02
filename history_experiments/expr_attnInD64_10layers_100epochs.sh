#!/bin/bash

# Set the GPU index
gpu_index="0,1,2,3"
export CUDA_VISIBLE_DEVICES=$gpu_index

# Array of parameterization types
# parameterizations=("best" "exp" "direct" "softplus")
parameterizations=("best" "exp" "softplus")

# Array of learning rates
# learning_rates=(5e-0 5e-1 5e-2 5e-3 5e-4 5e-5 5e-6)
learning_rates=(5e-1 5e-2 5e-3 5e-4)

# Number of repetitions for each experiment
num_repetitions=3
# num_repetitions=1

# Loop over repetitions
for (( rep=1; rep<=num_repetitions; rep++ )); do
    # Loop over learning rates
    for lr in "${learning_rates[@]}"; do
        # Loop over parameterizations
        for param in "${parameterizations[@]}"; do
            echo "Running experiment with parameterization: $param, learning rate: $lr, repetition: $rep"
            python src/train.py \
                experiment=Attn_by_SSM/ssm \
                task_name="Attn_StableSSM_Run${rep}_LR${lr}_${param}" \
                trainer.devices=4 \
                model.net.parameterization="$param" \
                logger=many_loggers \
                model.optimizer.lr="$lr" \
                seed=$rep \
                model.net.n_layers=10 \
                logger.wandb.project="StableSSM_Attn_SSM_10Layers_InD64" \
                data.input_dim=64 \
                trainer.max_epochs=100
        done
    done
done

# Loop over repetitions
for (( rep=1; rep<=num_repetitions; rep++ )); do
    # Loop over learning rates
    for lr in "${learning_rates[@]}"; do
        # Loop over parameterizations
        for param in "${parameterizations[@]}"; do
            echo "Running experiment with parameterization: $param, learning rate: $lr, repetition: $rep"
            python src/train.py \
                experiment=Attn_by_SSM/ssm \
                task_name="Attn_StableSSM_Run${rep}_LR${lr}_${param}" \
                trainer.devices=4 \
                model.net.parameterization="$param" \
                logger=many_loggers \
                model.optimizer.lr="$lr" \
                seed=$rep \
                model.net.n_layers=10 \
                logger.wandb.project="StableSSM_Attn_SSM_10Layers_InD64" \
                data.input_dim=64 \
                trainer.max_epochs=1000
        done
    done
done
