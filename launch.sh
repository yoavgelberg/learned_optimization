#!/bin/bash

set -e

# python -m learned_optimization.examples.simple_lopt_train --train_log_dir=logs/mlp_logs/adam \
#     --lopt adam --task mlp --learnable_hp --alsologtostderr
python -m learned_optimization.examples.simple_lopt_train --train_log_dir=logs/mlp_logs/nfnhybrid_adam \
    --lopt nfnhybrid_adam --task mlp --pointwise True --learnable_hp True --alsologtostderr
