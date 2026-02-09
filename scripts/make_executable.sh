#!/bin/bash
# Setup script to make all shell scripts executable

echo "Making all shell scripts executable..."

chmod +x *.sh

echo ""
echo "Scripts ready:"
ls -1 *.sh | while read script; do
    echo "  ✓ $script"
done

echo ""
echo "All scripts are now executable!"
echo ""
echo "Usage:"
echo "  ./help.sh              - Show all available commands"
echo "  ./setup_runpod.sh      - Initial setup (install PyTorch, dependencies)"
echo "  ./run_training.sh      - Start/Resume training"
echo "  ./quick_test.sh        - Quick model test"
echo "  ./run_inference.sh     - Run single inference"
echo "  ./evaluate_model.sh    - Full evaluation suite"
echo "  ./compare_checkpoints.sh - Compare all checkpoints"
