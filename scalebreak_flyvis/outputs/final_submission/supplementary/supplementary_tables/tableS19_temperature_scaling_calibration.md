| model               | calibration        |   temperature |   accuracy |   ece |   brier |   cross_entropy |   direction_bits_proxy |
|:--------------------|:-------------------|--------------:|-----------:|------:|--------:|----------------:|-----------------------:|
| FlyVis linear probe | uncalibrated       |          1    |      0.921 | 0.092 |   0.176 |           2.73  |                 -1.353 |
| FlyVis linear probe | temperature_scaled |          2.9  |      0.921 | 0.313 |   0.291 |           0.812 |                  1.414 |
| STN-CNN             | uncalibrated       |          1    |      0.576 | 0.133 |   0.541 |           0.981 |                  1.17  |
| STN-CNN             | temperature_scaled |          1.05 |      0.576 | 0.12  |   0.535 |           0.976 |                  1.177 |