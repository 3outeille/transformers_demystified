# transformers_demystified

- Config:
    - Ubuntu 20.04.5 LTS
    - torch 1.11.0
    - Cuda 11.4 (cuda for torch wheel 11.3 >= works too)
- When creating custom kernel/env for jupyter notebook, do not forget to activate env before running `python -m ipykernel install --user --name=env`
---
- CNN without attention: [Neural Machine Translation in Linear Time](https://arxiv.org/abs/1610.10099)
    - https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb
- CNN  + attention: [paper "Convolutional Sequence to Sequence Learning"](https://arxiv.org/pdf/1705.03122.pdf)
    - install torch==1.8 and torchtext==0.9 to understand structure of dataset
    - https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb