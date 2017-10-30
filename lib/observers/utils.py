import torch
import pandas as pd


def iterate_batch(*args, batch_first=False):
    """
    Get a generator through a batch of outputs/targets.

    Args:
        *args:
            outputs (torch.Tensor [seq_len, batch_size, dictionary_size]): outputs of a batch.
            targets (torch.Tensor [seq_len, batch_size]): expected output of a batch.
            source  (torch.Tensor [seq_len, batch_size]): source tensor of a batch.
        batch_first (bool): batch is the second dimension if False, else it is the first
    Returns:
        generator for tuples with two objects ->
            output (torch.Tensor [seq_len, dictionary_size]): outputs of a batch.
            target (torch.Tensor [seq_len]): expected output of a batch.
    """
    args = list(args)
    if not batch_first:
        for i, arg in enumerate(args):
            args[i] = args[i].transpose(0, 1)

    # Batch is first
    batch_size = args[0].size(0)

    for i in range(batch_size):
        ret = []
        for arg in args:
            ret.append(arg[i])
        yield tuple(ret)


def flatten_batch(outputs, targets):
    """
    Take outputs and their targets and return both with their batch dimension flattened.

    Example:
      `torch.nn._Loss` accepts only targets of 1D and outputs of 2D. For an efficient loss
      computation, it can be useful to flatten a 2D targets and 3D output to 1D and 2D respectively.
    Args:
        outputs (torch.Tensor [seq_len, batch_size, dictionary_size]): outputs of a batch.
        targets (torch.Tensor [seq len, batch size]): expected output of a batch.
    Returns:
        outputs (torch.Tensor [seq_len * batch_size, dictionary_size]): outputs of a batch.
        targets (torch.Tensor [seq len * batch size]): expected output of a batch.
        batch_size (int): size of the batch
    """
    batch_size = outputs.size(1)
    # (seq len, batch size, dictionary size) -> (batch size * seq len, dictionary size)
    outputs = outputs.view(-1, outputs.size(2))
    # (seq len, batch size) -> (batch size * seq len)
    targets = targets.view(-1)
    return outputs, targets, batch_size


def output_to_prediction(output):
    """
    Given output from a decoder, return predictions from the softmax layer.

    Args:
        output (torch.Tensor [seq_len, dictionary_size]): output from decoder
    Returns:
        prediction (torch.Tensor [seq_len]): predictions
    """
    return output.max(1)[1].view(-1)


def df_to_string_option_context(df, option_context):
    """
    Compute pandas.Dataframe.__str__ but use option context.
    
    Args:
        df (pandas.Dataframe)
        option_context (list or None): Pandas options context for display
        https://pandas.pydata.org/pandas-docs/stable/generated/pandas.option_context.html#pandas.option_context
    Returns:
        (str) dataframe expressed as a string
    """
    if option_context is not None:
        with pd.option_context(*option_context):
            return str(df)
    else:
        return str(df)


def torch_equals_ignore_index(target, prediction, ignore_index=None):
    """
    Compute torch.equals with the optional mask parameter.
   
    Args:
        ignore_index (int, optional): specifies a target index that is ignored
    Returns:
        (bool) iff target and prediction are equal
    """
    if ignore_index is not None:
        mask_arr = target.ne(ignore_index)
        target = target.masked_select(mask_arr)
        prediction = prediction.masked_select(mask_arr)

    return torch.equal(target, prediction)
