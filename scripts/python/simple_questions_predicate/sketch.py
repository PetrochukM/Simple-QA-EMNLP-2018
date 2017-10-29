# MISSING: Loading the configuration
# MISSING: Ability for hyperparameter tunning requires wrapping the training.
# MISSING: Ability to run a CLI with model
# MISSING: Metrics knowing where to save their files too. That should be with the experiment. Or that
# could be part of the configuration. Pass the save_directory to metrics.
# NOTE: Configuration at the top of the file, why not? Allows for a smaller footprint. Makes sense
# to define the entirety of the model together. Does not allow for multiple configurations but it does
# MISSING: Support for similar models and training procedures but different datasets

configuration = {
    'seq2seq':
        'models':
            'decoder_rnn.DecoderRNN.__init__':
                << : &base_rnn
                    'embedding_size': 300
                    'rnn_size': 256
                    'n_layers': 2
                    'rnn_cell': lstm
                    'embedding_dropout': 0.4
                    'rnn_variational_dropout': 0.0
                'rnn_dropout': 0.0
                use_attention: true
            encoder_rnn.EncoderRNN.__init__:
                << : *base_rnn
                rnn_dropout: 0.25
                bidirectional: true
                freeze_embeddings: true
            attention.Attention.__init__.attention_type: general

        fields.utils.utils:
            get_input_field:
                lower: true
                tokenizer: Moses
            build_input_vocab.vectors: *embeddings
            get_output_field.tokenizer: Identity

        metrics:
            utils.get_loss.loss: &loss NLLLoss
            utils.get_evaluate_loss.loss: *loss
            accuracy_top_n.AccuracyTopN.__init__.top_n: 2
            random_sample.RandomSample.__init__.n_samples: 25
            bucket_metric.BucketMetric.__init__.bucket_key: target_first_token
              
        controllers:
            trainer.Trainer.__init__:
                experiments_directory: &experiments_directory ./experiments
                random_seed: *random_seed
            trainer.Trainer.train:
                n_epochs: 10
                batch_size_inference: &inference_batch_size 128
                batch_size_train: 16
            evaluator.Evaluator.evaluate.batch_size: *inference_batch_size

        util.checkpoint.Checkpoint:
            get_checkpoint.experiment_path: *experiments_directory
            get_latest_checkpoint_name.experiment_path: *experiments_directory

        optim.optim.Optimizer.__init__:
            optim_class_name: Adam
            lr: 0.001584
            weight_decay: 0
            max_grad_norm: 0.65
}

# OTHER IDEA: Split this up into a model and a dataset that are not dependant. Models are depedent
# on the dataset format on the data its self; therefore, there should be restriction on the 
# types of datasets something can accept?
# Dataset needs specific hyperparameter configuration
# Load up a model, dataset, hyperparameters, etc
# Similar to the last thing we were doing
# The last thing we were doing had a trainer, model, dataset, etc. The problem was the inability to 
# easily mess with or experiement with the training procedure. The checkpoints were also a problem. 
# It did not follow an imperative programming style. 
# In order to follow an imperative proramming style, we want helpers or utility methods for trainning
# and ideally control it via a script like torchtext. 
# Overall we want to define the scripts via the problems they are solving and provide utilites to 
# solve those problems. We do not want to limit to configuration files; therefore, if you have 
# to choose a model and configure it, it wont be helpful. We will provide model utils and a 

class Model():
    pass


class ClassificationModel():
    pass

class Seq2SeqModel():
    # NOTE: The configuration of the rest of the system is not set by the model. The model only
    # puts the components together. 
    # NOTE: Logs is where the model stores any logging or files it creates for the sake of
    # reproducibility.
    def __init__(load_checkpoint=None, save_directory=''):
        if checkpoint:
            model, text_encoder = checkpoint
        else:
            model = init_model()
            text_encoder = init_text_encoder()

        self.model = model
        self.text_encoder = text_encoder

    def train(dataset=default_dataset):
        # NOTE: Defining the metrics. The metrics are core with the task; therefore, they can be
        # defined globally
        self.metrics = []

        train_dataset, dev_dataset, _ = load_datasets()
        for dataset in (train_dataset, dev_dataset):
            for i, item in dataset:
                dataset[i] = text_encoder.encode(dataset[i])
                # Everything in the dataset is in row tensor form, now it needs to be batched

            # torch_iterator
        for i in range(num_epochs):
            model.train(mode=True)
            # Train
            for batch in iterator(train_dataset):
                # Train on BATCH

                # Forward propagation
                outputs = model(batch)[0]

                # TODO: Consider using predictor?
                # Compute batch-specific metrics
                for metric in self.metrics:
                    metric.eval_batch(outputs, batch)

                # Compute loss
                self.loss.reset()
                self.loss.eval_batch(outputs, batch)

                # Backward propagation
                model.zero_grad()
                self.loss.backward()
                self.optimizer.step()

            save(model, text_encoder)

            model.train(mode=False)
            # Train then evaluate
            for batch in iterator(dev_dataset):
                # Compute batch-specific metrics
                for metric in self.metrics:
                    # MISSING: In an imperative style, how would the metrics decode the numbers for using.
                    # Is it appropriate to pass them the text_encoder?
                    # Can I look at other examples for this.
                    metric.eval_batch(outputs, batch)

            # MISSING: Printing the metrics and reseting them

    def predicate():
        pass

    def evaluate():
        pass

    def checkpoint():
        pass


if __name__ == '__main__':
    main()