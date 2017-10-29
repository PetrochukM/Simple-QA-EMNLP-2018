# TODO: Consider can this Object wrapper be removed in favor of a more SkickitOptimize like API
# for tunning, CLI, server, etc?
# This could be two different projects. One is a torchtext like extensions while the other is
# serving, cli, tunning, etc, productionizing, 
# NOTE: A model is defined by it's text encoding. It's not defined by it's dataset; therefore,
# that is an optional parameter

class Model(object):

    def __init__(self):
        pass

    def evaluate():
        pass

    def predict():
        pass

    def checkpoint():
        pass

    # TODO: Calling a model will be an alias to predict

class SimpleModel(object):
      def __init__(self, config=None, checkpoint=None):
        # TODO: init_logging
        # NOTE: Remove any thing but the API. Other functions should not be part of the object model
        # and are imported much like metrics.
        if checkpoint:
            self.model = checkpoint[0]
            self.text_encoder = checkpoint[1]
        else:
            self.train_dataset = ""
            self.dev_dataset = ""
            self.model = Seq2Seq()
            self.text_encoder = text_encoder()
            self.train_loss = loss()
            self.evaluate_loss = loss()
            # MISSING: Training process and logging until finished and model is ready to use.
        pass

    def evaluate():
        pass

    def predict():
        pass

    def checkpoint():
        pass
    