#!/bin/bash

echo "=============================================="
echo "  Parkinson's Disease Detection"
echo "  Complete Dataset Training"
echo "=============================================="
echo ""
echo "This script will train using ALL available data:"
echo "  - GITDATA: 226 Parkinson's samples"
echo "  - FIGDATA: 81 samples (41 healthy + 40 Parkinson's)"
echo "  - TOTAL: 307 raw samples (~921 after augmentation)"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to start..."
sleep 5

echo ""
echo "Starting training with combined dataset..."
echo ""

python3 train_combined.py

echo ""
echo "=============================================="
echo "Training complete!"
echo "Check the results/ directory for outputs"
echo "=============================================="
