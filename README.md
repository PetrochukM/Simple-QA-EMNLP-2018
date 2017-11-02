# Independent Study

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

- [ ] Add classification model for Simple Questions predicate
- [ ] Add a CLI and different hyperparameter configurations to train script
- [ ] Recreate the Simple Questions using the new Simple Questions names
- [ ] Import notebooks for Simple Questions end to end
- [ ] Add CRF model for Simple Questions objects
