# Pytorch BERT Pretrain / Finetuning  
HuggingFace transformer BERT Trainer    

## Requirements
- python 3.6   
- pytorch 1.12  
- tensorflow 1.14 (for tensorboard)  
- pytorch_transformers  
- gluonnlp >= 0.6.0  
- apex (for mixed precision training)  


Pretrained Korean Bert Model ([ETRI](http://aiopen.etri.re.kr/service_dataset.php) or [SKT](http://aiopen.etri.re.kr/service_dataset.php))  
1. ETRI kobert  
Make directory `pretrained_model/etri/` and put `berg_config.json`, `pytorch_model.bin`, `tokenization.py` and `vocab.korean.rawtext.list`  

2. SKT kobert  
Make directory `pretrained_model/skt/` and put `bert_config.json`, `pytorch_model.bin`, `tokenizer.model` and `vocab.json`  

## DATASETS  
- [한국어 단발성 대화 데이터셋](http://aicompanion.or.kr/kor/tech/data.php)  
- Any Dataset containing binary label(긍정, 부정)  
Datasets should have two columns Sentence and Emotion.  
or you can modify a few codes below which are in `datasets.py` to fit your own datasets  
```python
def get_data(self, file_path):
    data = pd.read_csv(file_path)
    corpus = data['Sentence']
    label = None
    try:
        label = [self.label2idx[l] for l in data['Emotion']]
    except:
        pass
    return corpus, label
```

## Usage  
For maksed language model pretrain  
```
$ python train_mlm.py
```  
  
For text classification  
```
$ python train_classification.py
```  
  
Use fp16 argument for [mixed precision training](https://github.com/NVIDIA/apex)  
```
$ python train_classification.py --fp16
```
