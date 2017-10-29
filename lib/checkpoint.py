# MISSING: Load text encoding and pickled model to predict and evaluate
# TODO: When predicting, to turn the model train=False
class Checkpoint():

    def __init__(location=None):
        """ Load a checkpoint """
        self.model = model
        self.input_field = input_field
        self.output_field = output_field
        self.device = device_default(device)
        pass

    @classmethod
    def save():
        pass

    def predict():
        self.model.train(mode=False)

        preprocessed = self.input_field.preprocess(sequence)
        preprocessed = self.input_field.preprocess_with_vocab(preprocessed)
        padded = self.input_field.pad([preprocessed])
        batch_data = self.input_field.numericalize(padded, self.device, train=False)
        batch = Batch.fromvars(None, 1, train=False, input=batch_data)
        logger.info('Preprocessed %s', padded)

        logger.info('Predicting...')
        decoder_outputs = self.model(batch)[0]
        decoder_outputs = decoder_outputs.data.squeeze(1)
        output_sequences = decoder_outputs.topk(top, dim=1)[1]

        ret = []
        for i in range(min(top, output_sequences.size()[1])):
            output_sequence = output_sequences[:, i]
            log_confidence = [
                decoder_outputs[j][token_idx] for j, token_idx in enumerate(output_sequence)
            ]
            confidence = [math.exp(x) for x in log_confidence]

            # Make sure not to have side affects
            self.model.train(mode=True)
            decoded = [self.output_field.vocab.itos[idx] for idx in output_sequence.tolist()]
            ret.append([decoded, confidence])

        if top == 1:
            return tuple(ret[0])
        else:
            return ret

        pass

    # MISSING: Pass observers and folder location to evaluate on the test set or an arbitrary set. 
    def evaluate():
        data_iter = BucketIterator(data, batch_size=batch_size, device=self.device, train=False)

        # Required to turn off dropout
        model.train(mode=False)

        if self.loss is not None:
            self.loss.reset()
        if self.metrics is not None:
            [metric.reset() for metric in self.metrics]

        last_percentage = 0.0
        for i, batch in enumerate(data_iter):
            # First return of the model should be targets.size by vocab_size
            outputs = model(batch)[0]
            if print_progress and abs((i / len(data_iter)) - last_percentage) > .05:
                # Print every 5%
                logger.info('Evaluator progress: %f%%', i / len(data_iter))
                last_percentage = i / len(data_iter)

            if self.metrics is not None:
                [metric.eval_batch(outputs, batch) for metric in self.metrics]
            if self.loss is not None:
                self.loss.eval_batch(outputs, batch)

        if self.metrics is not None:
            for metric in self.metrics:
                logger.info('Eval %s', str(metric))
        if self.loss is not None:
            dev_loss = self.loss.get_measurement()
            if not self.minimize_loss:
                dev_loss = -dev_loss
            logger.info("Eval %s: %.4f", self.loss.name, dev_loss)

        model.train(mode=True)

        # Dev loss is used by the trainer to use to decide the learning rate dynamically
        if self.loss is not None:
            return dev_loss
        else:
            return None

        pass
