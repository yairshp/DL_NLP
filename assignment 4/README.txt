# How to run the code

## Prerequisite
 - install torch, numpy and torchtext==0.9.0
 - the relevant files:
   - utils.py
   - bi_lstm_with_inner_attention.py
   - train.py

## Running the code

run the following command:

```python3 train.py```

## Important Notes

the code is using the torchtext package, which downloads the dataset and the embeddings while running (it will only download the first time it runs).