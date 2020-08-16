# CoLAKE

Source code for paper "CoLAKE: Contextualized Language and Knowledge Embedding". If you have any problem about reproducing the experiments, please feel free to contact us or propose an issue.

## Prepare your environment

We recommend to create a new environment.

```bash
conda create --name colake python=3.7
source activate colake
```

CoLAKE is implemented based on [fastNLP](https://github.com/fastnlp/fastNLP) and [huggingface's transformers](https://github.com/huggingface/transformers), and uses [fitlog](https://github.com/fastnlp/fitlog) to record the experiments.

```bash
git clone https://github.com/fastnlp/fastNLP.git
cd fastNLP/ & python setup.py install
git clone https://github.com/fastnlp/fitlog.git
cd fitlog/ & python setup.py install
pip install transformers
pip install sklearn
```

To re-train CoLAKE, you may need mixed CPU-GPU training to handle the large number of entities. Our implementation is based on KVStore provided by [DGL](https://github.com/dmlc/dgl). In addition, to reproduce the experiments on link prediction, you may also need [DGL-KE](https://github.com/awslabs/dgl-ke).

```bash
pip install dgl
pip install dglke
```

## Reproduce the experiments

### 1. Download the model and entity embeddings

Download the pre-trained CoLAKE [model](https://drive.google.com/file/d/1MEGcmJUBXOyxKaK6K88fZFyj_IbH9U5b) and [embeddings](https://drive.google.com/file/d/1_FG9mpTrOnxV2NolXlu1n2ihgSZFXHnI) for more than 3M entities. To reproduce the experiments on LAMA and LAMA-UHN, you only need to download the model. You can use the `download_gdrive.py` in this repo to directly download files from Google Drive to your server:

```bash
mkdir model
python download_gdrive.py 1MEGcmJUBXOyxKaK6K88fZFyj_IbH9U5b ./model/model.bin
python download_gdrive.py 1_FG9mpTrOnxV2NolXlu1n2ihgSZFXHnI ./model/entities.npy
```

Alternatively, you can use `gdown`:

```bash
pip install gdown
gdown https://drive.google.com/uc?id=1MEGcmJUBXOyxKaK6K88fZFyj_IbH9U5b
gdown https://drive.google.com/uc?id=1_FG9mpTrOnxV2NolXlu1n2ihgSZFXHnI
```

### 2. Run the experiments

Download the datasets for the experiments in the paper: [Google Drive](https://drive.google.com/file/d/1UNXICdkB5JbRyS5WTq6QNX4ndpMlNob6/view?usp=sharing).

```bash
python download_gdrive.py 1UNXICdkB5JbRyS5WTq6QNX4ndpMlNob6 ./data.tar.gz
tar -xzvf data.tar.gz
cd finetune/
```

#### FewRel

```bash
python run_re.py --debug --gpu 0
```

#### Open Entity

```bash
python run_typing.py --debug --gpu 0
```

#### LAMA and LAMA-UHN

```bash
cd ../lama/
python eval_lama.py
```

## Re-train CoLAKE

### 1. Download the data

Download the latest wiki dump (XML format):

```bash
wget -c https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```

Download the knowledge graph (Wikidata5M):

```bash
wget -c https://www.dropbox.com/s/86gukevtdbhpcbk/wikidata5m_triplet.txt.gz
gzip wikidata5m_triplet.txt.gz -d
```

Download the Wikidata5M entity & relation aliases:

```bash
wget -c https://www.dropbox.com/s/s1q38yzqzvuodl3/wikidata5m_alias.tar.gz
tar -xzvf wikidata5m_alias.tar.gz
```

### 2. Preprocess the data

Preprocess wiki dump:

```bash
mkdir pretrain_data
# process xml-format wiki dump
python preprocess/WikiExtractor.py enwiki-latest-pages-articles.xml.bz2 -o pretrain_data/output -l --min_text_length 100 --filter_disambig_pages -it abbr,b,big --processes 4
# Modify anchors
python preprocess/extract.py 4
python preprocess/gen_data.py 4
# Count entity & relation frequency and generate vocabs
python statistic.py
```

### 3. Train CoLAKE

Initialize entity and relation embeddings with the average of RoBERTa BPE embedding of entity and relation aliases:

```bash
cd pretrain/
python init_ent_rel.py
```

Train CoLAKE with mixed CPU-GPU:

```bash
./run_pretrain.sh
```

## Cite

If you use the code and model, please cite this paper:

```
@inproceedings{sun2020colake,
  author = {Tianxiang Sun and Yunfan Shao and Xipeng Qiu and Qipeng Guo and Yaru Hu and Xuanjing Huang and Zheng Zhang},
  title = {CoLAKE: Contextualized Language and Knowledge Embedding},
  booktitle = {Proceedings of COLING 2020},
  year = {2020}
}
```

## Acknowledgments

- [fastNLP](https://github.com/fastnlp/fastNLP)

- [LAMA](https://github.com/facebookresearch/LAMA)

- [ERNIE](https://github.com/thunlp/ERNIE)

