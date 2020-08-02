#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export ISTRAIN=1
export NUM_SWAPS=12
MODEL_NAME='env8'

# TODO: save path add train keyword
python -W ignore -m baselines.run --alg=ppo2 \
                        --env=Mapping-v0 \
                        --num_timesteps=1e8 \
			            --save_path=models/${MODEL_NAME} \
                        --log_path=logs/${MODEL_NAME}/train \
                        --save_interval=100 \
                        --log_interval=5 \
                        --num_units=256 \
                        --istrain=True \
                        --num_layers=3 \
                        --network=rnet \
                        --nsteps=128 \
                        --num_env=8
