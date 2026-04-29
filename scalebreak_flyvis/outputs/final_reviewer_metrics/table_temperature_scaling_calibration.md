| model               | calibration        |   temperature |   accuracy |       ece |    brier |   cross_entropy |   direction_bits_proxy |
|:--------------------|:-------------------|--------------:|-----------:|----------:|---------:|----------------:|-----------------------:|
| FlyVis linear probe | uncalibrated       |          1    |   0.920833 | 0.0917623 | 0.175591 |        2.72962  |               -1.35305 |
| FlyVis linear probe | temperature_scaled |          2.9  |   0.920833 | 0.313038  | 0.291151 |        0.811793 |                1.41379 |
| STN-CNN             | uncalibrated       |          1    |   0.576389 | 0.133427  | 0.540925 |        0.980832 |                1.16992 |
| STN-CNN             | temperature_scaled |          1.05 |   0.576389 | 0.119848  | 0.535097 |        0.975698 |                1.17733 |