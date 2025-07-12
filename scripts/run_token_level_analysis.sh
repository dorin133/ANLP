#!/bin/bash

# Token-Level Salience Analysis Script
# This script runs the complete token-level analysis pipeline on Qwen3 model data
# 
# Purpose: Generate heatmap visualizations showing salient tokens within each thought
# Input: Zip files in output4_new/ directory  
# Output: Visualizations in visualiztions_salient_thoughts/qwen3/token_level_analysis/
#
# Usage: ./run_token_level_analysis.sh [num_examples]
#   num_examples: Number of examples to process (default: 5)

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANLP_DIR="$(dirname "$SCRIPT_DIR")"
ANALYSIS_SCRIPT="$ANLP_DIR/auxiliaries/token_level_salience_analysis.py"
OUTPUT_DIR="$ANLP_DIR/visualiztions_salient_thoughts/qwen3/token_level_analysis"
TEMP_DIR="$ANLP_DIR/temp_extract"

# Default number of examples to process
NUM_EXAMPLES=${1:-5}

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"

# Print configuration
echo "=================================="
echo "Token-Level Salience Analysis"
echo "=================================="
echo "Analysis script: $ANALYSIS_SCRIPT"
echo "Output directory: $OUTPUT_DIR"
echo "Processing $NUM_EXAMPLES examples"
echo "=================================="

# Change to the auxiliaries directory to run the script
cd "$ANLP_DIR/auxiliaries"

# Run the analysis
python token_level_salience_analysis.py

# Check if output was generated
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A "$OUTPUT_DIR")" ]; then
    echo "✅ Analysis complete! Results saved to:"
    echo "   $OUTPUT_DIR"
    echo "   Generated $(find "$OUTPUT_DIR" -name "*.png" | wc -l) visualization files"
else
    echo "❌ No output generated. Check for errors above."
    exit 1
fi

# Clean up temporary files
if [ -d "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
    echo "🧹 Cleaned up temporary files"
fi

echo "=================================="
echo "Token-level analysis completed successfully!"
echo "==================================" 