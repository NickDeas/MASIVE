# MASIVE (Columbia NLP)
Code accompanying the paper ["MASIVE: Open-Ended Affective State Identification in English and Spanish"](https://arxiv.org/pdf/2407.12196), Deas et al., 2024 presented at the 2024 Conference on Empirical Methods in Natural Language Processing.

<div style="margin-left:auto;margin-right:auto">
  <img src="/data_example.png?" height="400"/> 
  <img src="/bootstrap_diagram.png?" height="400"/> 
</div>

# Setup

1. Clone the repository
2. Create a virtual environment and install all dependencies
   ```
    pip install -r requirements.txt
   ```
3. Run `setup.sh` to download the data and make all experiment scripts executable. The data download includes the English and Spanish subsets of masive, train/validation/test splits, regional Spanish data, translated data, and subsetted data used throughout experimental results.

# Data
The data is automatically downloaded through the `setup.sh` script. If you want to download the data separately or the download does not work, you can access the data at [https://drive.google.com/file/d/1MxvcL7M3iGa4-j_hznMTBLXe9ULEAHDf/view?usp=sharing](https://drive.google.com/file/d/1MxvcL7M3iGa4-j_hznMTBLXe9ULEAHDf/view?usp=sharing).

# Experiments

Scripts are included in the `code/scripts` directory to train and evaluate models used in each experiment. The `code/configs` directory contains the configs used for each experiment and can be modified to alter training.
mT5 and T5 experiments were run using 2 A100 GPU's.

## MASIVE Benchmark

To reproduce the benchmark experiments (Table 4,6 & Figure 3), follow the steps below:
1. Navigate to the base directory, `MASIVE`
2. Run the following two scripts to train the English and Spanish models.
   ```
   CUDA_VISIBLE_DEVICES=<GPU_IDS> code/scripts/train_internal_en.sh
   CUDA_VISIBLE_DEVICES=<GPU_IDS> code/scripts/train_internal_es.sh
   ```
3. Run the following script to evaluate the finetuned English and Spanish models.
   ```
   CUDA_VISIBLE_DEVICES=<GPU_IDS> code/scripts/masive_eval.sh
   ```

## External Evaluation
To reproduce the external evaluation experiments (Table 5), follow the steps below:
1. Run steps 1 and 2 in [__MASIVE Benchmark__](#MASIVE-Benchmark) to finetune the necessary mT5 models
2. Run the following scripts to finetune the mT5 models on the necessary emotion datasets.
   ```
   CUDA_VISIBLE_DEVICES=<GPU_IDS> code/scripts/train_external.sh
   ```
3. Run the following script to evaluate the finetuned mT5 models.
   ```
   CUDA_VISIBLE_DEVICES=<GPU_IDS> code/scripts/eval_external.sh
   ```
4. _(Optional)_ To repeat experiments with T5, run the following scripts.
   ```
   CUDA_VISIBLE_DEVICES=<GPU_IDS> code/scripts/train_external_t5.sh
   CUDA_VISIBLE_DEVICES=<GPU_IDS> code/scripts/eval_external_t5.sh
   ```

## Translation Experiments
To reproduce the translation experiments (Table 8), follow the steps below:
1. Run steps 1 and 2 in [__MASIVE Benchmark__](#MASIVE-Benchmark)
) to finetune the necessary native mT5 models
2. Run the following scripts to finetune the subset English models, subset translated Spanish models, and translated English model respectively.
   ```
   CUDA_VISIBLE_DEVICES=<GPU_IDS> code/scripts/train_internal_en_sub.sh
   CUDA_VISIBLE_DEVICES=<GPU_IDS> code/scripts/train_internal_es_trans_sub.sh
   CUDA_VISIBLE_DEVICES=<GPU_IDS> code/scripts/train_internal_en_trans.sh
   ```
3. Run the following script to evaluate the finetuned mT5 models.
   ```
   CUDA_VISIBLE_DEVICES=<GPU_IDS> code/scripts/eval_internal_en_sub.sh
   CUDA_VISIBLE_DEVICES=<GPU_IDS> code/scripts/eval_internal_es_trans_sub.sh
   CUDA_VISIBLE_DEVICES=<GPU_IDS> code/scripts/eval_internal_en_trans.sh
   ```

## Tables and Figures
All Tables and Figures can be generated with the ["Tables and Figures.ipynb"](./code/Tables-and-Figures.ipynb) notebook.

# Citation
If you use MASIVE or the resulting models in your work, please cite our paper.
```
@inproceedings{deas-etal-2024-masive,
    title = "{MASIVE}: Open-Ended Affective State Identification in {E}nglish and {S}panish",
    author = "Deas, Nicholas  and
      Turcan, Elsbeth  and
      Mejia, Ivan Ernesto Perez  and
      McKeown, Kathleen",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.1139",
    pages = "20467--20485",
    abstract = "In the field of emotion analysis, much NLP research focuses on identifying a limited number of discrete emotion categories, often applied across languages. These basic sets, however, are rarely designed with textual data in mind, and culture, language, and dialect can influence how particular emotions are interpreted. In this work, we broaden our scope to a practically unbounded set of affective states, which includes any terms that humans use to describe their experiences of feeling. We collect and publish MASIVE, a dataset of Reddit posts in English and Spanish containing over 1,000 unique affective states each. We then define the new problem of affective state identification for language generation models framed as a masked span prediction task. On this task, we find that smaller finetuned multilingual models outperform much larger LLMs, even on region-specific Spanish affective states. Additionally, we show that pretraining on MASIVE improves model performance on existing emotion benchmarks. Finally, through machine translation experiments, we find that native speaker-written data is vital to good performance on this task.",
}

```

# Contact
For any questions, please contact [ndeas@cs.columbia.edu](mailto:ndeas@cs.columbia.edu)
