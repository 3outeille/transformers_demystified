# transformers_demystified

- Config:
    - Ubuntu 20.04.5 LTS
    - torch 1.11.0
    - Cuda 11.4 (cuda for torch wheel 11.3 >= works too)
- When creating custom kernel/env for jupyter notebook, do not forget to activate env before running `python -m ipykernel install --user --name=env`

- CNN attention:
    - install torch==1.8 and torchtext==0.9 to understand structure of dataset
    - https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb