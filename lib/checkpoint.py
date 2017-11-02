import logging
import math
import os
import time

from torch.autograd import Variable

import dill
import torch

from lib.utils import device_default

logger = logging.getLogger(__name__)


# MISSING: Load text encoding and pickled model to predict and evaluate
# TODO: When predicting, to turn the model train=False
class Checkpoint(object):
    # TODO: If location is not a string, load the most recent checkpoint; else, load the checkpoint
    # at location.
    # Not the experiments folder, but the checkpoint parent
    # List of checkpoints
    # Checkpoint container
    def __init__(self, checkpoint_path, device=None):
        """
        Load a checkpoint.

        Args:
            device (int)
            checkpoint_path (str or None): Given a non-none checkpoint path, the checkpoint is
                loaded
        """
        self.device = device_default(device)
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
    def recent(cls, save_directory, device=None):
        """
        Load a checkpoint or returns `None` if save_directory has no checkpoint.

        Args:
            save_directory (str or None): Lastest checkpoint is loaded from save_directory
            device (int)
        """
        all_filenames = sorted(os.listdir(save_directory), reverse=True)
        all_checkpoints = [filename for filename in all_filenames if '.pt' in filename]
        if len(all_checkpoints) == 0:
            return None
        checkpoint_path = os.path.join(save_directory, all_checkpoints[0])
        return cls(checkpoint_path, device)

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

    def predict(self, input_, batch_first=False, top_k=1):
        """ Make prediction given `input_`."""
        self.model.train(mode=False)

        tensor = self.input_text_encoder.encode(input_)

        # Create the batch
        batch_source = tensor.unsqueeze(0)  # Batch size of 1
        if not batch_first:
            batch_source = batch_source.t_()
        batch_source = Variable(batch_source.contiguous())
        batch_source_length = torch.LongTensor([tensor.size()[0]])

        # Predict
        output_batch = self.model(batch_source, batch_source_length)[0]
        output_tensor = output_batch.data.squeeze(1)  # Squeeze
        output_sequences = output_tensor.topk(top_k, dim=1)[1]

        # Make human readable
        ret = []
        for i in range(min(top_k, output_sequences.size()[1])):
            output_sequence = output_sequences[:, i]
            log_confidence = [
                output_tensor[j][token_index] for j, token_index in enumerate(output_sequence)
            ]
            confidence = [math.exp(x) for x in log_confidence]

            # Make sure not to have side affects
            decoded = self.output_text_encoder.decode(output_sequence)
            ret.append([decoded, confidence])

        self.model.train(mode=True)
        if top_k == 1:
            return tuple(ret[0])
        else:
            return ret

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
