from seq2seq.config import configurable
from seq2seq.util.util import get_variable_from_string
from seq2seq.fields.seq_field import SeqField

import seq2seq


@configurable
def get_input_field(field='SeqField', checkpoint=None, **kwargs):
    """ Instantiate a input_field. @configurable in order to allow configuration of input_field """
    if checkpoint:
        return checkpoint.input_field
    return _get_field_class(field)(**kwargs)


@configurable
def get_output_field(field='SeqField', checkpoint=None, **kwargs):
    """ Instantiate a output_field. @configurable in order to allow configuration of output_field """
    if checkpoint:
        return checkpoint.output_field
    return _get_field_class(field)(eos_token=SeqField.EOS_TOKEN, **kwargs)


def _get_field_class(field_class_name):
    """ Given a field class name, find the field class in seq2seq.fields """
    return get_variable_from_string(field_class_name, [seq2seq.fields])


@configurable
def build_input_vocab(input_field, examples, **kwargs):
    """ Build input_field vocab. @configurable in order to allow configuration of vocab build """
    input_field.build_vocab(examples, **kwargs)


@configurable
def build_output_vocab(output_field, examples, **kwargs):
    """ Build output_field vocab. @configurable in order to allow configuration of vocab build """
    output_field.build_vocab(examples, **kwargs)
