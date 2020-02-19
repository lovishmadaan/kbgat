# Knowledge Base Graph Attention Networks (KBGAT)

Various tests on [ACL 2019](http://www.acl2019.org/EN/index.xhtml) paper: [Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs](https://arxiv.org/abs/1906.01195)

Baseline code taken from [relationPrediction](https://github.com/deepakn97/relationPrediction).

### Baselines
Direct reproduction of baselines is present in the `baseline` folder on Freebase and Wordnet. Scripts to reproduce the results are `fb15.sh` and `wordnet.sh`, and the corresponding outputs are `fb15_out.txt` and `wordnet_out.txt`, respectively.


### Tests

* **Using only GAT trained model to evaluate the results (without ConvKB)**

    Results present in folder  `baseline/` - file is `test_gat.py` and script is `gatonly.sh`

* **DISTMULT scoring function instead of TransE**

    Results present in folder `tests/distmult/`

* **Random Initialization of Entity and Relational Embeddings instead of TransE**

    Results present in folder `tests/transe/`