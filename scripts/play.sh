#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export ISTRAIN=""
export NUM_SWAPS=512

MODEL_NAME=env8
mkdir -p plogs/${MODEL_NAME}

for i in 00001 {01000..03000..1000}; # {01000..10000..1000};
do
SAVE_NAME=ckpt${i}

\time --verbose python -m baselines.run --alg=ppo2 \
                        --env=Mapping-v0 \
                        --num_timesteps=0 \
                        --log_path=logs/${MODEL_NAME}/eval-${NUM_SWAPS}/ckpt${i} \
			            --load_path=logs/${MODEL_NAME}/train/checkpoints/${i} \
                        --istrain=False \
                        --num_units=256 \
                        --num_layers=2 \
                        --network=rnet \
                        --play | tee plogs/${MODEL_NAME}/eval-${NUM_SWAPS}-ckpt${i}
done
