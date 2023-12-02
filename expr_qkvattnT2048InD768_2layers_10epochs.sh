#!/bin/bash

# Set the GPU index
gpu_index="2"
export CUDA_VISIBLE_DEVICES=$gpu_index

# Array of parameterization types
parameterizations=("best" "exp" "softplus" "direct")

# Array of learning rates
learning_rates=(5e-1 5e-2 5e-3 5e-4)

# Number of repetitions for each experiment
num_repetitions=2

# Loop over repetitions
for (( rep=1; rep<=num_repetitions; rep++ )); do
    # Loop over learning rates
    for lr in "${learning_rates[@]}"; do
        # Loop over parameterizations
        for param in "${parameterizations[@]}"; do
            echo "Running experiment with parameterization: $param, learning rate: $lr, repetition: $rep"
            python src/train.py \
                experiment=QKVAttn_by_SSM/ssm \
                task_name="QKVAttn_StableSSM_Run${rep}_LR${lr}_${param}" \
                trainer.devices=1 \
                model.net.parameterization="$param" \
                logger=many_loggers \
                model.optimizer.lr="$lr" \
                seed=$rep \
                model.net.n_layers=2 \
                logger.wandb.project="StableSSM_QKVAttn_SSM_2Layers_T2048_InD768_H3072" \
                trainer.max_epochs=10 \
                data.batch_size=4 \
                data.seq_length=2048 \
                data.input_dim=768 \
                model.net.rec1_size=3072 \
                data.train_val_test_split=[600,50,50]
        done
    done
done
