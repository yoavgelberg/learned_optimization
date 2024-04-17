#!/bin/bash

set -e

function TPU0() {
    TPU_VISIBLE_DEVICES=0 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8476 TPU_MESH_CONTROLLER_PORT=8476 "$@"
}
function TPU1() {
    TPU_VISIBLE_DEVICES=1 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8477 TPU_MESH_CONTROLLER_PORT=8477 "$@"
}
function TPU2() {
    TPU_VISIBLE_DEVICES=2 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8478 TPU_MESH_CONTROLLER_PORT=8478 "$@"
}
function TPU3() {
    TPU_VISIBLE_DEVICES=3 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8479 TPU_MESH_CONTROLLER_PORT=8479 "$@"
}


TPU0 python -m examples.eval_inner --num_evals 5 --task mlp --out_path /tmp/mlp_eval/nfn --lopt nfn --learnable_hp True --ckpt_path ~/lopt_mlp_logs/mlp_nfn_learnable_rep0/thetaT.pkl &
TPU2 python -m examples.eval_inner --num_evals 5 --task mlp --out_path /tmp/mlp_eval/mlp --lopt mlp --learnable_hp True --ckpt_path ~/lopt_mlp_logs/mlp_mlp_learnable_rep0/thetaT.pkl &
TPU3 python -m examples.eval_inner --num_evals 5 --task mlp --out_path /tmp/mlp_eval/sgdm --lopt sgdm --ckpt_path ~/lopt_mlp_logs/mlp_sgdm_rep0/thetaT.pkl