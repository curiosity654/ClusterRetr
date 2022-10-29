# Supplementary List

1. src, model training, post processing and evaluation code
2. data, gallery and query features using SAKE model with distribution alignment loss

# Code Usage

1. Calculate and save scores.
    ```
    python src/pre_process.py
    ```

2. Run cluster and retrieval experiments。
* KMeans w/ fuse
    ```
    python src/clusterretr.py --feature_file ./data/features_processed.pickle > results/kld_sketchy_fuse0.2.log --fuse 0.2
    ```

* KMeans w/o fuse
    ```
    python src/clusterretr.py --feature_file ./data/features_processed.pickle > results/kld_sketchy.log
    ```