# HAM10000 Multimodal Classification Benchmark — Results

## Summary

Evaluation of RA-SAMO generalization from survival prediction to
dermoscopic image + tabular metadata 7-class lesion classification.

## Results Table

| Method | N | Macro-F1 | Balanced Acc | AUROC | Accuracy |
|--------|---|----------|-------------|-------|----------|
| Image-Only | 3 | 0.6981 +/- 0.0179 | 0.6698 +/- 0.0199 | 0.9573 +/- 0.0045 | 0.8270 +/- 0.0056 |
| Meta-Only | 3 | 0.1956 +/- 0.0082 | 0.2163 +/- 0.0164 | 0.7192 +/- 0.0547 | 0.6661 +/- 0.0284 |
| AdamW | 3 | 0.7091 +/- 0.0277 | 0.7043 +/- 0.0281 | 0.9598 +/- 0.0030 | 0.8284 +/- 0.0057 |
| SAM | 3 | 0.7043 +/- 0.0325 | 0.6831 +/- 0.0345 | 0.9642 +/- 0.0068 | 0.8383 +/- 0.0058 |
| Adaptive-Aux | 3 | 0.6917 +/- 0.0281 | 0.6697 +/- 0.0343 | 0.9544 +/- 0.0079 | 0.8221 +/- 0.0048 |
| SAMO | 3 | 0.6845 +/- 0.0268 | 0.6607 +/- 0.0402 | 0.9546 +/- 0.0082 | 0.8179 +/- 0.0072 |
| RA-SAMO | 3 | 0.6950 +/- 0.0108 | 0.6671 +/- 0.0227 | 0.9564 +/- 0.0048 | 0.8221 +/- 0.0046 |

## Statistical Tests (RA-SAMO vs others, paired t-test on Macro-F1)

| Comparison | Mean Diff | t-stat | p-value | N |
|------------|-----------|--------|---------|---|
| RA-SAMO vs Image-Only | -0.0031 | -0.279 | 0.7801 | 3 |
| RA-SAMO vs Meta-Only | +0.4994 | 114.183 | 0.0000 * | 3 |
| RA-SAMO vs AdamW | -0.0142 | -1.140 | 0.2541 | 3 |
| RA-SAMO vs SAM | -0.0094 | -0.600 | 0.5483 | 3 |
| RA-SAMO vs Adaptive-Aux | +0.0032 | 0.257 | 0.7970 | 3 |
| RA-SAMO vs SAMO | +0.0104 | 0.892 | 0.3721 | 3 |

## Per-Method Details

### Image-Only

- Seed 42: F1=0.7230, Acc=0.6965, AUROC=0.9633, Best epoch=30, Time=1373s
- Seed 43: F1=0.6895, Acc=0.6642, AUROC=0.9559, Best epoch=13, Time=1410s
- Seed 44: F1=0.6818, Acc=0.6487, AUROC=0.9526, Best epoch=17, Time=1406s

### Meta-Only

- Seed 42: F1=0.1924, Acc=0.2029, AUROC=0.7369, Best epoch=5, Time=1344s
- Seed 43: F1=0.2068, Acc=0.2393, AUROC=0.6451, Best epoch=1, Time=1287s
- Seed 44: F1=0.1876, Acc=0.2067, AUROC=0.7756, Best epoch=30, Time=1288s

### AdamW

- Seed 42: F1=0.7311, Acc=0.7271, AUROC=0.9624, Best epoch=30, Time=1391s
- Seed 43: F1=0.7263, Acc=0.7210, AUROC=0.9613, Best epoch=16, Time=1348s
- Seed 44: F1=0.6700, Acc=0.6646, AUROC=0.9556, Best epoch=9, Time=1369s

### SAM

- Seed 42: F1=0.7279, Acc=0.7025, AUROC=0.9714, Best epoch=26, Time=1535s
- Seed 43: F1=0.7267, Acc=0.7120, AUROC=0.9661, Best epoch=22, Time=1528s
- Seed 44: F1=0.6584, Acc=0.6346, AUROC=0.9550, Best epoch=26, Time=1458s

### Adaptive-Aux

- Seed 42: F1=0.7125, Acc=0.6985, AUROC=0.9609, Best epoch=14, Time=1356s
- Seed 43: F1=0.7108, Acc=0.6892, AUROC=0.9589, Best epoch=19, Time=1373s
- Seed 44: F1=0.6520, Acc=0.6215, AUROC=0.9432, Best epoch=20, Time=1353s

### SAMO

- Seed 42: F1=0.6882, Acc=0.6858, AUROC=0.9625, Best epoch=20, Time=1366s
- Seed 43: F1=0.7154, Acc=0.6923, AUROC=0.9580, Best epoch=15, Time=1383s
- Seed 44: F1=0.6500, Acc=0.6039, AUROC=0.9432, Best epoch=18, Time=1401s

### RA-SAMO

- Seed 42: F1=0.6999, Acc=0.6762, AUROC=0.9605, Best epoch=24, Time=2047s
- Seed 43: F1=0.7050, Acc=0.6892, AUROC=0.9589, Best epoch=19, Time=2021s
- Seed 44: F1=0.6800, Acc=0.6358, AUROC=0.9497, Best epoch=23, Time=2022s
