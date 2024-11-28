# GenCodeSearchNet: A Benchmark Test Suite for Evaluating Generalization in Programming Language Understanding

This repository contains data and code to reproduce the results in our paper 'GenCodeSearchNet: A Benchmark Test Suite for Evaluating Generalization in Programming Language Understanding' published in Proceedings of the 1st GenBench Workshop on (Benchmarking) Generalisation in NLP 2023.

Preprint link: https://arxiv.org/abs/2311.09707

The paper introduces a new benchmark test suite to evaluate programming language understanding generalization in language models and is included in the GenBench Collaborative Benchmarking Task (CBT) 2023. As part of the test suite we also include the newly created StatCodeSearch dataset, which includes 1070 code-comment pairs from social science research code written in R.

Link to CBT repo: https://github.com/GenBench/genbench_cbt_2023

Link to Zenodo repository of the dataset: https://zenodo.org/records/8310891

### Folder structure:
    ├── data                                 # Code to create GenCodeSearchNet test sets
    │   ├── clf                              # Code to create the text-code matching subsets
    │   ├── mrr                              # Code to create the search/rank subsets
    │   ├── statcodesearch_full              # Includes the full StatCodeSearch dataset with additional metadata and prefiltered versions
    ├── experiments                          # Code for evaluation
### Contributors:
Andor Diera, Abdelhalim Dahou, Lukas Galke, Fabian Karl, Florian Sihler, Ansgar Scherp
