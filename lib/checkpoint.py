import logging
import os
import time

import dill
import torch

logger = logging.getLogger(__name__)


# MISSING: Load text encoding and pickled model to predict and evaluate
# TODO: When predicting, to turn the model train=False
class Checkpoint(object):
    # TODO: If location is not a string, load the most recent checkpoint; else, load the checkpoint
    # at location.
    # Not the experiments folder, but the checkpoint parent
    # List of checkpoints
    # Checkpoint container
    def __init__(self, device, checkpoint_path):
        """
        Load a checkpoint.

        Args:
            device (int)
            checkpoint_path (str or None): Given a non-none checkpoint path, the checkpoint is
                loaded
        """
        self.device = device
        self.checkpoint_path = checkpoint_path

        logger.info("Loading checkpoints from %s onto device %d", self.checkpoint_path, self.device)

        # http://pytorch.org/docs/master/torch.html?highlight=torch%20load#torch.load
        def remap(storage, loc):
            if 'cuda' in loc and self.device >= 0:
                return storage.cuda(device=self.device)
            return storage

        data = torch.load(self.checkpoint_path, map_location=remap)
        # https://stackoverflow.com/questions/2535917/copy-kwargs-to-self
        for (k, v) in data.items():
            setattr(self, k, v)

        self.model.flatten_parameters()  # make RNN parameters contiguous

    @classmethod
    def load(cls, device, checkpoint_path=None, save_directory=None):
        """
        Load a checkpoint.

        Pre: There exists a save_directory with some experiments; where each experiment has some
             number of checkpoints, logs, visualizations, etc.
        Args:
            device (int)
            checkpoint_path (str or None): Given a non-none checkpoint path, the checkpoint is
                loaded
            save_directory (str or None): By default the lastest checkpoint is loaded from
                save_directory
        """
        if checkpoint_path is not None:
            checkpoint_path = checkpoint_path
        elif save_directory is not None:
            all_filenames = sorted(os.listdir(save_directory), reverse=True)
            all_checkpoints = [filename for filename in all_filenames if '.pt' in filename]
            if len(all_checkpoints) == 0:
                return None
            checkpoint_path = os.path.join(save_directory, all_checkpoints[0])
        else:
            raise ValueError('checkpoint_path or save_directory must be non-none')

        return cls(device, checkpoint_path)

    @classmethod
    def save(cls,
             save_directory,
             model,
             optimizer,
             input_text_encoder,
             output_text_encoder,
             device=None):
        """
        Saves the current model and related training parameters into a subdirectory of the
        checkpoint directory. The name of the subdirectory is the current local time in
        M_D_H_M_S format.

        Args:
            save_directory (str): path to the save directory
            model
            optimizer
            input_text_encoder
            output_text_encoder
            device (int): give a device number to be appended to the end of the path
        """
        date_time = time.strftime('%mm_%dd_%Hh_%Mm_%Ss', time.localtime())
        name = '%s_[%d]' % (date_time, device) if device else date_time
        name += '.pt'
        path = os.path.join(save_directory, name)

        if os.path.exists(path):
            logger.error('Cannot save checkpoint; directory (%s) already exists.', path)
            return

        logger.info('Saving checkpoint %s', name)

        torch.save(
            {
                'optimizer': optimizer,
                'model': model,
                'input_text_encoder': input_text_encoder,
                'output_text_encoder': output_text_encoder,
            },
            path,
            pickle_module=dill)

    def predict(self):
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
    def evaluate(self):
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
