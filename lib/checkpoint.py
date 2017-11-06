import logging
import math
import os
import time

from torch.autograd import Variable

import dill
import torch

import lib.utils

logger = logging.getLogger(__name__)


class Checkpoint(object):

    def __init__(self, checkpoint_path, device=None):
        """
        Load a checkpoint.

        Args:
            device (int)
            checkpoint_path (str or None): Given a non-none checkpoint path, the checkpoint is
                loaded
        """
        self.device = lib.utils.device_default(device)
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

        if hasattr(self.model, 'flatten_parameters'):
            self.model.flatten_parameters()  # make RNN parameters contiguous

    @classmethod
    def recent(cls, log_directory, device=None):
        """
        Load a checkpoint or returns `None` if log_directory has no checkpoint.

        Args:
            log_directory (str or None): Lastest checkpoint is loaded from log_directory
            device (int)
        """
        all_filenames = sorted(os.listdir(log_directory), reverse=True)
        all_checkpoints = [filename for filename in all_filenames if '.pt' in filename]
        if len(all_checkpoints) == 0:
            return None
        checkpoint_path = os.path.join(log_directory, all_checkpoints[0])
        return cls(checkpoint_path, device)

    @classmethod
    def save(cls,
             log_directory,
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
            log_directory (str): path to the save directory
            model
            optimizer
            input_text_encoder
            output_text_encoder
            device (int): give a device number to be appended to the end of the path
        """
        date_time = time.strftime('%mm_%dd_%Hh_%Mm_%Ss', time.localtime())
        name = '%s_[%d]' % (date_time, device) if device else date_time
        name += '.pt'
        path = os.path.join(log_directory, name)

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
        if torch.cuda.is_available():
            batch_source = batch_source.cuda()
            batch_source_length = batch_source_length.cuda()

        # Predict
        output_batch = self.model(batch_source, batch_source_length)[0]
        if len(output_batch.size()) > 1:
            output_batch = output_batch.data.squeeze(1)  # Squeeze 1 batch

        output_sequences = output_batch.topk(top_k, dim=len(output_batch.size()) - 1)[1]
        print('output_sequences', output_sequences.size())

        # Make human readable
        ret = []
        for i in range(min(top_k, output_sequences.size()[-1])):
            print('output_sequences', output_sequences.size())
            output_sequence = output_sequences[..., i]
            print('output_sequence', output_sequence.size())
            log_confidence = [
                output_batch[j][token_index] for j, token_index in enumerate(output_sequence)
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
