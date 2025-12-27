# Transformer Experiment Results Summary

## Overall Results

| Configuration         | Position Encoding   | Normalization   | Optimizer   | Label Smoothing   |   Best Epoch |   Best Val BLEU |   Test BLEU |   Final Train Loss |   Final Val Loss |
|:----------------------|:--------------------|:----------------|:------------|:------------------|-------------:|----------------:|------------:|-------------------:|-----------------:|
| rel_rms_adam_smooth   | Relative            | RMSNorm         | Adam        | Smooth            |           11 |            9.69 |        9.5  |             1.0883 |           3.4751 |
| rel_ln_noam_nosmooth  | Relative            | LayerNorm       | Noam        | No Smooth         |           13 |            8.67 |        9.08 |             0.9405 |           4.4543 |
| rel_ln_noam_smooth    | Relative            | LayerNorm       | Noam        | Smooth            |           15 |            9.68 |        8.91 |             0.9334 |           3.5358 |
| abs_rms_noam_smooth   | Absolute            | RMSNorm         | Noam        | Smooth            |           11 |            8.85 |        8.67 |             1.213  |           3.5003 |
| rel_ln_adam_smooth    | Relative            | LayerNorm       | Adam        | Smooth            |           15 |            8.99 |        8.67 |             0.9167 |           3.6253 |
| abs_ln_noam_smooth    | Absolute            | LayerNorm       | Noam        | Smooth            |           10 |            8.55 |        8.41 |             1.2416 |           3.4822 |
| rel_rms_noam_nosmooth | Relative            | RMSNorm         | Noam        | No Smooth         |           11 |            9.52 |        8.37 |             1.058  |           4.2045 |
| abs_rms_adam_smooth   | Absolute            | RMSNorm         | Adam        | Smooth            |           13 |            8.42 |        8.36 |             1.1164 |           3.6378 |
| rel_rms_noam_smooth   | Relative            | RMSNorm         | Noam        | Smooth            |           10 |            9.82 |        8.06 |             1.1616 |           3.3606 |
| rel_rms_adam_nosmooth | Relative            | RMSNorm         | Adam        | No Smooth         |            8 |            8.49 |        7.95 |             1.2846 |           4.0306 |
| abs_ln_adam_smooth    | Absolute            | LayerNorm       | Adam        | Smooth            |           16 |            8.41 |        7.9  |             1.0056 |           3.7695 |
| abs_ln_noam_nosmooth  | Absolute            | LayerNorm       | Noam        | No Smooth         |           12 |            7.65 |        7.64 |             1.1415 |           4.331  |
| rel_ln_adam_nosmooth  | Relative            | LayerNorm       | Adam        | No Smooth         |           10 |            7.35 |        7.59 |             1.1961 |           4.3334 |
| abs_rms_adam_nosmooth | Absolute            | RMSNorm         | Adam        | No Smooth         |           13 |            7.2  |        7.22 |             1.0946 |           4.5083 |
| abs_rms_noam_nosmooth | Absolute            | RMSNorm         | Noam        | No Smooth         |           12 |            7.95 |        6.55 |             1.1133 |           4.3793 |
| abs_ln_adam_nosmooth  | Absolute            | LayerNorm       | Adam        | No Smooth         |           11 |            6.97 |        5.84 |             1.2495 |           4.408  |

## Top 5 Configurations

| Configuration        | Position Encoding   | Normalization   | Optimizer   | Label Smoothing   |   Best Epoch |   Best Val BLEU |   Test BLEU |   Final Train Loss |   Final Val Loss |
|:---------------------|:--------------------|:----------------|:------------|:------------------|-------------:|----------------:|------------:|-------------------:|-----------------:|
| rel_rms_adam_smooth  | Relative            | RMSNorm         | Adam        | Smooth            |           11 |            9.69 |        9.5  |             1.0883 |           3.4751 |
| rel_ln_noam_nosmooth | Relative            | LayerNorm       | Noam        | No Smooth         |           13 |            8.67 |        9.08 |             0.9405 |           4.4543 |
| rel_ln_noam_smooth   | Relative            | LayerNorm       | Noam        | Smooth            |           15 |            9.68 |        8.91 |             0.9334 |           3.5358 |
| abs_rms_noam_smooth  | Absolute            | RMSNorm         | Noam        | Smooth            |           11 |            8.85 |        8.67 |             1.213  |           3.5003 |
| rel_ln_adam_smooth   | Relative            | LayerNorm       | Adam        | Smooth            |           15 |            8.99 |        8.67 |             0.9167 |           3.6253 |

## Statistical Analysis

- **Best BLEU Score**: 9.50 (rel_rms_adam_smooth)
- **Worst BLEU Score**: 5.84
- **Average BLEU Score**: 8.04
- **Standard Deviation**: 0.94
