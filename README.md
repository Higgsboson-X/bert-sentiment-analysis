# Bert-Sentiment-Analysis

Sentiment analysis using pretrained Bert model


## Dependencies

```
python3.6
torch==1.3.1
torchtext==0.5.0
transformers==2.8.0
```

## Prepare Data

Prepare `test.json` and `train.json` in the data directory. Each file contain lines of json strings with following fields.
```
{
  "text": [string, the sentence],
  "label": [integer, true sentiment label],
  "id": [integer, sentence id]
}
```


## Training

Run the following command.
```
$ cd code
$ python train.py \
    --data_path [data path] \
    --epochs 40 \
    --epochs_per_val 5 \
    --save_id [id] \
    --device_id [device id] \
    --bidirectional
```
Here are some more options.
```
optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        data path, contains `train.json` and `test.json`
  --epochs EPOCHS       training epochs
  --epochs_per_val EPOCHS_PER_VAL
                        epochs per evaluation
  --batch_size BATCH_SIZE
                        batch size of data
  --device_id DEVICE_ID
                        gpu index
  --save_id SAVE_ID     model save id
  --h_dim H_DIM         hidden dimension of GRU
  --n_cls N_CLS         number of classes
  --n_layers N_LAYERS   number of layers
  --bidirectional       whether to use bidirectional GRU
  --dropout DROPOUT     dropout rate
```

## Testing

Run the following command.
```
$ cd code
$ python eval.py --data_path [data path] --save_id [id] --device_id [device id]
```


