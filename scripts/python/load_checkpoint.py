from lib.checkpoint import Checkpoint
from lib.datasets import reverse

checkpoint_path = 'save/reverse_11m_01d_23h_07m_58s/11m_01d_23h_09m_48s_[-1].pt'
checkpoint = Checkpoint(checkpoint_path)
output_sequence, confidence = checkpoint.predict('1 2 3')
print(output_sequence)
print(confidence)

test_dataset = reverse(test=True)
# NOTE: `source` and `target` are typically associated with sequences
assert 'source' in train_dataset and 'source' in dev_dataset
assert 'target' in train_dataset and 'target' in dev_dataset

logger.info('Num Training Data: %d', len(train_dataset))
logger.info('Num Development Data: %d', len(dev_dataset))