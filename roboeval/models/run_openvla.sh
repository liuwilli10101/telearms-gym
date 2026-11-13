#!/bin/bash

# Activate your conda environment first (optional)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate roboeval

# PROJECT_ROOT="${HOME}/RoboEval"
# REPO_PY_ROOT="${PROJECT_ROOT}"

# # Ensure Python can import the repo packages (demonstrations, roboeval, etc.)
# export PYTHONPATH="${REPO_PY_ROOT}:${PYTHONPATH}"

# # Run from the package root so relative imports resolve as expected
# cd "${REPO_PY_ROOT}" || {
#     echo "Failed to cd into ${REPO_PY_ROOT}" >&2
#     exit 1
# }

# List of (ckpt_path, env_name, instruction) tuples
declare -a ckpt_paths=(
    "/home/carter/RoboEval/thirdparty/openvla/openvla-7b+stack_two_blocks_delta+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
    "/home/carter/RoboEval/thirdparty/openvla/openvla-7b+push_block_to_target_delta+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
    "/home/carter/RoboEval/thirdparty/openvla/openvla-7b+rotate_letter_block_delta+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
    "/home/carter/RoboEval/thirdparty/openvla/openvla-7b+align_blocks_delta+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
)

declare -a env_names=(
    "Stack Two Blocks"
    "Push Block to Target"
    "Rotate Letter Block"
    "Align Blocks"
)

declare -a instructions=(
    "stack the blocks on top of each other"
    "place the blocks to the colored target patch"
    "rotate the letter block"
    "align the blocks in a straight line"
)

declare -a unnorm_keys=(
    "bridge_orig"
    "bridge_orig"
    "bridge_orig"
    "bridge_orig"
)

# Path to the Python script
SCRIPT_PATH="${HOME}/RoboEval/roboeval/models/eval_openvla.py"

# Number of entries
NUM_ENTRIES=${#ckpt_paths[@]}

# Ensure checkpoint directories exist (create placeholders if missing)
for ckpt in "${ckpt_paths[@]}"; do
    if [ ! -d "${ckpt}" ]; then
        echo "Creating checkpoint directory: ${ckpt}"
        mkdir -p "${ckpt}"
        echo "⚠️  Directory created empty. Download the OpenVLA model files into this folder before running evaluations."
    fi
done

for ((i=0; i<NUM_ENTRIES; i++)); do
    echo "Running evaluation for: ${env_names[$i]}"

    python "$SCRIPT_PATH" \
        --ckpt_path "${ckpt_paths[$i]}" \
        --env_name "${env_names[$i]}" \
        --instruction "${instructions[$i]}" \
    --unnorm_key "${unnorm_keys[$i]}" \
        --downsample_rate 25 \
        --num_episodes 2
done