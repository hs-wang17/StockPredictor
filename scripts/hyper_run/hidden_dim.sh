#!/bin/bash

HIDDEN_DIM_VALUES=(32 64 128 256 512 1024)

for HIDDEN_DIM in "${HIDDEN_DIM_VALUES[@]}"; do
    echo "Running with HIDDEN_DIM=${HIDDEN_DIM}"
    HIDDEN_DIM=${HIDDEN_DIM} bash /home/user0/project/predictor/scripts/run_neural_network_parallel.sh
    echo "Finished HIDDEN_DIM=${HIDDEN_DIM}"
    echo ""
done