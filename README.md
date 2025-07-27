# ANLP

### Directory Structure

```
array_logs/           # Legacy Qwen2.5 analysis results, weren't included in our final project report

output4_new/           # Contains all results produced for our final report analysis by running main.py

auxilaries/           # Legacy code of results visualization

visualiztions_salient_thoughts/
├── qwen2.5/           # Legacy Qwen2.5 analysis results
│   ├── *.png          # old results, weren't included in our final project report
└── qwen3/             # New Qwen3 analysis results  
    ├── token_level_analysis/     # Results of thought-level analysis of SalientCount heatmaps (like figure 1 in the final report) over all samples
    │   └── *.png
    └── thought_level_analysis/   # Results of token-level analysis of SalientCount (like figure 2 in the final report) over some representative samples (the amount over all samples is huge so we didn't upload all to github)
        ├── aime_*/salient_thoughts.png
        └── math-algebra_*/salient_thoughts.png
```

## Running Visualizations

Refer to [`utils/visualization_readme.md`](utils/visualization_readme.md) for instructions on generating visualizations.

## Reproducing Final Results

To recreate the experiments and figures in the final report:

- For **multi-node GPU execution**:
  ```bash
  scripts/run_parallel.sh
