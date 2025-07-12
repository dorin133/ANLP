#!/bin/bash

# Thought-Level Salience Analysis Script
# This script runs the complete thought-level analysis pipeline on Qwen3 model data
# 
# Purpose: Generate heatmap visualizations showing salient thoughts across layers and heads
# Input: Zip files in output4_new/ directory (60 examples total)
# Output: Visualizations in visualiztions_salient_thoughts/qwen3/thought_level_analysis/
#
# Usage: ./run_thought_level_analysis.sh

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANLP_DIR="$(dirname "$SCRIPT_DIR")"
ANALYSIS_SCRIPT="$ANLP_DIR/auxiliaries/thought_level_salience_analysis.py"
OUTPUT_DIR="$ANLP_DIR/visualiztions_salient_thoughts/qwen3/thought_level_analysis"
INPUT_DIR="$ANLP_DIR/output4_new"

# Print configuration
echo "=================================="
echo "Thought-Level Salience Analysis"
echo "=================================="
echo "Analysis script: $ANALYSIS_SCRIPT"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Processing all 60 examples (30 AIME + 30 math-algebra)"
echo "=================================="

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ Input directory not found: $INPUT_DIR"
    exit 1
fi

# Count zip files
ZIP_COUNT=$(find "$INPUT_DIR" -name "*.zip" | wc -l)
echo "Found $ZIP_COUNT zip files to process"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Change to the auxiliaries directory to run the script
cd "$ANLP_DIR/auxiliaries"

# Run the analysis
echo "Starting batch processing..."
python thought_level_salience_analysis.py

# Check if output was generated
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A "$OUTPUT_DIR")" ]; then
    echo "✅ Analysis complete! Results saved to:"
    echo "   $OUTPUT_DIR"
    
    # Count generated files
    AIME_COUNT=$(find "$OUTPUT_DIR" -name "aime_*" -type d | wc -l)
    MATH_COUNT=$(find "$OUTPUT_DIR" -name "math-algebra_*" -type d | wc -l)
    TOTAL_PLOTS=$(find "$OUTPUT_DIR" -name "*.png" | wc -l)
    
    echo "   Generated visualizations:"
    echo "   - AIME examples: $AIME_COUNT"
    echo "   - Math-algebra examples: $MATH_COUNT"
    echo "   - Total plots: $TOTAL_PLOTS"
else
    echo "❌ No output generated. Check for errors above."
    exit 1
fi

echo "=================================="
echo "Thought-level analysis completed successfully!"
echo "All 60 examples processed with 9 layers × 32 heads"
echo "==================================" 