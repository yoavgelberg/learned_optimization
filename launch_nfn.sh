#!/bin/bash

set -e

CUDA_VISIBLE_DEVICES=1 python -m learned_optimization.examples.simple_lopt_train --train_log_dir=logs/mlp_logs_v2/nominalAdam_stepMult0.001_outMult0 \
    --lopt mlp_adam --task mlp --learnable_hp True --step_mult 0.001 --out_mult 0 --alsologtostderr

CUDA_VISIBLE_DEVICES=1 python -m learned_optimization.examples.simple_lopt_train --train_log_dir=logs/mlp_logs_v2/nfn_nominalAdam_stepMult0.001_outMult0.001 \
    --lopt nfnhybrid_adam --task mlp --pointwise True --learnable_hp True --step_mult 0.001 --out_mult 0.001 --alsologtostderr

CUDA_VISIBLE_DEVICES=1 python -m learned_optimization.examples.simple_lopt_train --train_log_dir=logs/mlp_logs_v2/nfn_nominalAdam_stepMult0.001_outMult0.0001 \
    --lopt nfnhybrid_adam --task mlp --pointwise True --learnable_hp True --step_mult 0.001 --out_mult 0.0001 --alsologtostderr
