# Question Answering Research

## Synopsis

Project to build an end to end question answering system in multiple languages with industry
precision of 95% and a 90% recall. 

## Design Goal

Overall this repository is built on top of PyTorch; therefore, it will follow PyTorch's design
footsteps. As O’Reilly Data Show Podcast explained, this means, "PyTorch is designed to be
intuitive, linear in thought and easy to use. When you execute a line of code, it gets executed.
There isn’t an asynchronous view of the world".

In NLP deep learning, areas of research typically include datasets, models and training procedure.
The goal of this repository is to add minimal overhead to these components of training.

## TODO Prioritized

- [x] Add classification model for Simple Questions predicate
- [ ] Add a CLI and different hyperparameter configurations to train script
- [ ] Recreate the Simple Questions using the new Simple Questions names
- [ ] Import notebooks for Simple Questions end to end
- [ ] Add CRF model for Simple Questions objects

## Important GitHub repositories
- https://github.com/zihangdai/cfo
- https://github.com/Gorov/SimpleQuestions-EntityLinking
- https://github.com/yinwenpeng/KBQA_IBM
- https://github.com/yinwenpeng/KBQA_IBM_New
- https://github.com/WDAqua/teafacto
- https://github.com/syxu828/QuestionAnsweringOverFB
- https://github.com/facebook/MemNN
- https://github.com/castorini/BuboQA


## Important papers
- [Conditional Focused Neural Question Answering with Large-scale Knowledge Bases](https://www.aclweb.org/anthology/P/P16/P16-1076.pdf)
- [Simple Question Answering by Attentive Convolutional Neural Network - SOTA results](https://arxiv.org/abs/1606.03391)
- [Character-Level Question Answering with Attention](https://arxiv.org/abs/1604.00727)
- [Neural Network-based Question Answering over Knowledge Graphs on Word and Character Level](http://jens-lehmann.org/files/2017/www_nn_factoid_qa.pdf)
- [Simple and Effective Question Answering with Recurrent Neural Networks](https://arxiv.org/abs/1606.05029)
- [Improved Neural Relation Detection for Knowledge Base Question Answering - SOTA results](https://arxiv.org/pdf/1704.06194.pdf)
- [Question Answering on Freebase via Relation Extraction and Textual Evidence](https://arxiv.org/abs/1603.00957)
- [Comparative Study of CNN and RNN for Natural Language Processing](https://arxiv.org/abs/1702.01923)
