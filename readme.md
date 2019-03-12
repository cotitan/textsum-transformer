### Transformer for text summarization, implemented in pytorch
- author: Kirk
- mail: cotitan@outlook.com

### Requirments, * means not necessary
- pytorch==0.4.0
- numpy==1.12.1+
- python==3.5+
- tensorboardx==1.2*
- word2vec==0.10.2*
- allennlp==0.8.2*
- pyrouge==0.1.3*

### Data
Training and evaluation data for Gigaword is available https://drive.google.com/open?id=0B6N7tANPyVeBNmlSX19Ld2xDU1E

Training and evaluation data for CNN/DM is available https://s3.amazonaws.com/opennmt-models/Summary/cnndm.tar.gz

### Noticement
1. we use another thread to preprocess a batch of data, which would not terminate after the main process terminate. So you need to press ctrl+c again to terminate the thread.

### Directories:
```
.                  
├── log           
├── models         
├── sumdata       
├── tmp           
├── transformer 
├── Beam.py       
├── config.py    
├── train.py          
├── mytest.py        
├── Transformer.py
├── translate.py  
└── utils.py      
```
Make sure your project contains the folders above.

### How-to
1. Run _python train.py_ to train
2. Run _python mytest.py_ to generate summaries

