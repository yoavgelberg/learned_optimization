#!/bin/bash

set -e

CUDA_VISIBLE_DEVICES=0 python -m learned_optimization.examples.simple_lopt_train --train_log_dir=logs/mlp_logs_v2/learnable_adam \
    --lopt adam --task mlp --learnable_hp True --step_mult 0.001 --alsologtostderr

CUDA_VISIBLE_DEVICES=0 python -m learned_optimization.examples.simple_lopt_train --train_log_dir=logs/mlp_logs_v2/mlp_nominalAdam_stepMult0.001_outMult0.001_chan4_layers2 \
    --lopt mlp_adam --task mlp --learnable_hp True --step_mult 0.001 --out_mult 0.001 --alsologtostderr

CUDA_VISIBLE_DEVICES=0 python -m learned_optimization.examples.simple_lopt_train --train_log_dir=logs/mlp_logs_v2/mlp_nominalAdam_stepMult0.001_outMult0.0001_chan4_layers2 \
    --lopt mlp_adam --task mlp --learnable_hp True --step_mult 0.001 --out_mult 0.0001 --alsologtostderr
