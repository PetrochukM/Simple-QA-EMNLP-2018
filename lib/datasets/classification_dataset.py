# Missing: Using PyTorch dataloaders
class SeqToClassDataset(object):

    def __init__():
        pass

    # DEFINES: The iterator or table? Defines the columns of the batch?
    # MISSING: If a model requires a classification dataset, the dataset has to define a label
    # and text in it's batches during iteration. 

    # Batch can be defined with multiple outputs as long as there is a text and label for the model
    # to refer too? What about more complicated achitectures; then there more fields per row.
    # FOR example, the french and english can have a source and destination


class SeqToSeqDataset(object):
