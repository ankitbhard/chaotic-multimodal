#!/usr/bin/env bash
# HAM10000 multimodal classification benchmark — EC2 launcher
# Usage: bash experiments/ham10000/run_ec2.sh
set -euo pipefail

PYTHON="/opt/pytorch/bin/python3"
CODE_DIR="$HOME/chaotic-multimodal-v2"
DATA_DIR="$HOME/ham10000_data"
RESULTS_DIR="$CODE_DIR/results/ham10000_classification"
KAGGLE_DATASET="kmader/skin-cancer-mnist-ham10000"

# Auto-shutdown on exit (safety net)
trap 'echo "Script finished at $(date). Shutting down in 5 minutes..."; sudo shutdown +5' EXIT

cd "$CODE_DIR"

echo "=== HAM10000 Benchmark Runner ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

# Verify CUDA
if ! $PYTHON -c "import torch; assert torch.cuda.is_available(), 'No CUDA'; print(f'CUDA OK: {torch.cuda.get_device_name(0)}')"; then
    echo "ERROR: CUDA not available"
    exit 1
fi

# Install extra deps if needed
$PYTHON -c "import PIL" 2>/dev/null || pip install Pillow
$PYTHON -c "import pandas" 2>/dev/null || pip install pandas

# Download HAM10000 data if not present
if [ ! -f "$DATA_DIR/HAM10000_metadata.csv" ]; then
    echo "=== Downloading HAM10000 dataset ==="
    mkdir -p "$DATA_DIR"
    pip install kaggle 2>/dev/null || true
    kaggle datasets download -d "$KAGGLE_DATASET" -p "$DATA_DIR"
    cd "$DATA_DIR"
    unzip -o "*.zip"
    # Handle nested zips from Kaggle
    for z in HAM10000_images_part_1.zip HAM10000_images_part_2.zip; do
        [ -f "$z" ] && unzip -o "$z"
    done
    cd "$CODE_DIR"
    echo "Download complete. Images: $(find $DATA_DIR -name '*.jpg' | wc -l)"
fi

# Verify data
N_IMAGES=$(find "$DATA_DIR" -name 'ISIC_*.jpg' | wc -l)
echo "Found $N_IMAGES images in $DATA_DIR"
if [ "$N_IMAGES" -lt 10000 ]; then
    echo "WARNING: Expected ~10015 images, found $N_IMAGES"
fi

# Run tests first
echo "=== Running unit tests ==="
$PYTHON -m pytest experiments/ham10000/test_ham10000.py -v --tb=short 2>&1 | tail -20
echo "Tests passed."

# Smoke test (2 epochs, 1 seed, subset of methods)
echo "=== Smoke test (2 epochs) ==="
$PYTHON -m experiments.ham10000.train \
    --data-dir "$DATA_DIR" \
    --methods adamw ra_samo \
    --seeds 42 \
    --epochs 2 \
    --out "$RESULTS_DIR/smoke" \
    2>&1 | tail -20
echo "Smoke test passed."

# Full run: 7 methods x 3 seeds x 30 epochs
echo "=== Full benchmark run ==="
$PYTHON -m experiments.ham10000.train \
    --data-dir "$DATA_DIR" \
    --methods adamw sam adaptive_aux samo ra_samo image_only_adamw meta_only_adamw \
    --seeds 42 43 44 \
    --epochs 30 \
    --batch-size 64 \
    --lr 1e-4 \
    --out "$RESULTS_DIR" \
    --skip-existing \
    2>&1 | tee "$RESULTS_DIR/full_run.log"

echo "=== Benchmark complete ==="
echo "Results in: $RESULTS_DIR"

# Run analysis
echo "=== Running analysis ==="
$PYTHON -m experiments.ham10000.analyze --results-dir "$RESULTS_DIR"
echo "Analysis complete."
echo "Done at $(date)"
