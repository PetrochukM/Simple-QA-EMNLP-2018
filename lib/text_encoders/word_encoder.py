from lib.text_encoders.static_tokenizer_encoder import StaticTokenizerEncoder


class WordEncoder(StaticTokenizerEncoder):

    def decode(self, tensor):
        tokens = [self.itos[index] for index in tensor]
        return ' '.join(tokens)
