# MASIVE (Code upload in progress)
Code accompanying the paper ["MASIVE: Open-Ended Affective State Identification in English and Spanish"](https://arxiv.org/pdf/2407.12196), Deas et al., 2024 presented at the 2024 Conference on Empirical Methods in Natural Language Processing.

<div style="margin-left:auto;margin-right:auto">
  <img src="/data_example.png" height="400"/>
  <img src="/bootstrap_diagram.png" height="400"/> 
</div>

# Setup

1. Clone the repository
2. Create a virtual environment and install all dependencies
   ```
    pip install -r requirements.txt
   ```
3. Run `setup.sh` to download the data and make all experiment scripts executable. The data download includes the English and Spanish subsets of masive, train/validation/test splits, regional Spanish data, translated data, and subsetted data used throughout experimental results.

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
@misc{deas2024masiveopenendedaffectivestate,
      title={MASIVE: Open-Ended Affective State Identification in English and Spanish}, 
      author={Nicholas Deas and Elsbeth Turcan and Iván Pérez Mejía and Kathleen McKeown},
      year={2024},
      eprint={2407.12196},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.12196}, 
}
```

# Contact
For any questions, please contact [ndeas@cs.columbia.edu](mailto:ndeas@cs.columbia.edu)
