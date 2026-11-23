import tempfile
import unittest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config

from swift.model import get_model_processor


class TestMaxModelLen(unittest.TestCase):

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.model_dir = self._tmp_dir.name
        Qwen2Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32768).save_pretrained(self.model_dir)
        vocab = {'<unk>': 0, '<|endoftext|>': 1, '<|im_start|>': 2, '<|im_end|>': 3}
        tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, unk_token='<unk>', eos_token='<|im_end|>',
            pad_token='<|endoftext|>').save_pretrained(self.model_dir)

    def tearDown(self):
        self._tmp_dir.cleanup()

    def _get_max_model_len(self, max_model_len):
        _, processor = get_model_processor(
            self.model_dir, load_model=False, model_type='qwen2', max_model_len=max_model_len)
        return processor.model_info.max_model_len

    def test_default_comes_from_config(self):
        self.assertEqual(self._get_max_model_len(None), 32768)

    def test_max_model_len_updates_model_info(self):
        # The inference engines read `model_info.max_model_len` for the default max_tokens and length check.
        self.assertEqual(self._get_max_model_len(200), 200)
        self.assertEqual(self._get_max_model_len(65536), 65536)


if __name__ == '__main__':
    unittest.main()
