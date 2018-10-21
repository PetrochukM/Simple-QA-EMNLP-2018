# Simple-QA-EMNLP-2018

This is the code for the EMNLP 2018 paper "SimpleQuestions Nearly Solved: A New Upperbound and Baseline Approach".

The SimpleQuestions dataset is one of the most commonly used benchmarks for studying single-relation factoid questions.
We present new evidence that this benchmark can be nearly solved by standard methods. First we show that ambiguity in the data bounds performance on this benchmark at 83.4%; there are often multiple answers that cannot be disambiguated from the linguistic signal alone. Second we introduce a baseline that sets a new state-of-the-art performance level at 78.1% accuracy,  using only standard methods.

## Example

![Preview of the software](https://github.com/PetrochukM/Simple-QA-EMNLP-2018/blob/master/preview.png)

## Structure

```bash
.
├── /notebooks/                          
│   ├── /Simple QA End-To-End/           # Experiments on components of the end-to-end QA pipeline
│   ├── /Simple QA Models                # Experiments on various neural models
│   ├── /Simple QA KG to PostgreSQL DB   # Scripts to populate postgreSQL
│   ├── /Simple QA Numbers               # Scripts for computing and verifying various numbers
├── /pretrained_models/                   
├── /lib/                                # Various utility functionality
├── /tests/                               
├── .flake8                               
└── requirements.txt                     # Required python packages
```


## Prerequisites

This repository requires Python 3.5 or greater and PostgreSQL.

## Installation

* Clone the repository and cd into it
```
git clone https://github.com/PetrochukM/Simple-QA-EMNLP-2018.git
cd Simple-QA-EMNLP-2018
```

* Install the required packages
```
python -m pip install -r requirements.txt
```

* Create and populate a PostgreSQL table named ``fb_two_subject_name`` with ``notebooks/Simple QA KG to PostgreSQL DB/fb_two_subject_name.csv.gz``

* Download the SimpleQuestions v2 dataset from [Facebook Research](https://research.fb.com/downloads/babi/). Use the notebook at ``Simple-QA-EMNLP-2018/notebooks/Simple QA KG to PostgreSQL DB/FB5M & FB2M KG to DB.ipynb`` to create and populate a PostgreSQL table. 

* You're done! Feel free to run ``Simple-QA-EMNLP-2018/notebooks/Simple QA End-To-End``.

## Citation

```
@article{Petrochuk2018SimpleQuestionsNS,
  title={SimpleQuestions Nearly Solved: A New Upperbound and Baseline Approach},
  author={Michael Petrochuk and Luke S. Zettlemoyer},
  journal={CoRR},
  year={2018},
  volume={abs/1804.08798}
}
```

## Important Notes

* The FB2M and FB5M subsets of Freebase KG can complete 7,188,636 and 7,688,234 graph queries respectively; therefore, the FB5M subset is 6.9% larger than the FB2M subset. Also, the FB5M dataset only contains 3.98M entities. This contradicts the statement that "FB5M, is much larger with about 5M entities" (Bordes et al., 2015). 
* FB5M and FB2M contain 4,322,266 and 3,654,470 duplicate grouped facts respectively. 
* FB2M is not a subset of FB5M, 1 atomic fact is in FB2M that is not in FB5M: ``(01g4wmh, music/album/acquire_webpage, 02q5zps)``.
* FB5M and FB2M do not contain the answer for 24 and 36 examples in SimpleQuestions dataset respectively; therefore, those examples are unanswerable.

## Other Important Papers
- [Conditional Focused Neural Question Answering with Large-scale Knowledge Bases](https://www.aclweb.org/anthology/P/P16/P16-1076.pdf)
- [Simple Question Answering by Attentive Convolutional Neural Network - SOTA results](https://arxiv.org/abs/1606.03391)
- [Character-Level Question Answering with Attention](https://arxiv.org/abs/1604.00727)
- [Neural Network-based Question Answering over Knowledge Graphs on Word and Character Level](http://jens-lehmann.org/files/2017/www_nn_factoid_qa.pdf)
- [Simple and Effective Question Answering with Recurrent Neural Networks](https://arxiv.org/abs/1606.05029)
- [Improved Neural Relation Detection for Knowledge Base Question Answering - SOTA results](https://arxiv.org/pdf/1704.06194.pdf)
- [Question Answering on Freebase via Relation Extraction and Textual Evidence](https://arxiv.org/abs/1603.00957)
- [Comparative Study of CNN and RNN for Natural Language Processing](https://arxiv.org/abs/1702.01923)
- [Knowledge-based Question Answering by Jointly Generating, Copying and Paraphrasing](http://dl.acm.org/citation.cfm?id=3133064)
- [Open-domain Factoid Question Answering via Knowledge Graph Search](https://pdfs.semanticscholar.org/6414/5a8edf9add1156d97c489028539fa4c5c66c.pdf)
- [Large-scale Simple Question Answering with Memory Networks](https://research.fb.com/wp-content/uploads/2016/11/large-scale_simple_question_answering_with_memory_networks.pdf?)
- [Core Techniques of Question Answering Systems over Knowledge Bases: a Survey](https://hal.archives-ouvertes.fr/hal-01637143/document)

## Other Important GitHub Repositories
- https://github.com/zihangdai/cfo
- https://github.com/Gorov/SimpleQuestions-EntityLinking
- https://github.com/yinwenpeng/KBQA_IBM
- https://github.com/yinwenpeng/KBQA_IBM_New
- https://github.com/WDAqua/teafacto
- https://github.com/syxu828/QuestionAnsweringOverFB
- https://github.com/facebook/MemNN
- https://github.com/castorini/BuboQA
