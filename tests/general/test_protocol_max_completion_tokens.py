import unittest
from pydantic import TypeAdapter

from swift.infer_engine.protocol import ChatCompletionRequest


def _request(**kwargs):
    return ChatCompletionRequest(model='test', messages=[{'role': 'user', 'content': 'hi'}], **kwargs)


class TestChatCompletionRequestMaxCompletionTokens(unittest.TestCase):

    def test_max_completion_tokens_sets_max_tokens(self):
        _, request_config = _request(max_completion_tokens=16).parse()
        self.assertEqual(request_config.max_tokens, 16)

    def test_max_completion_tokens_in_request_body(self):
        # FastAPI parses the request body this way; unknown fields are dropped silently.
        request = TypeAdapter(ChatCompletionRequest).validate_python({
            'model': 'test',
            'messages': [{
                'role': 'user',
                'content': 'hi'
            }],
            'max_completion_tokens': 16
        })
        _, request_config = request.parse()
        self.assertEqual(request_config.max_tokens, 16)

    def test_max_tokens_still_works(self):
        _, request_config = _request(max_tokens=16).parse()
        self.assertEqual(request_config.max_tokens, 16)

    def test_max_tokens_wins_when_both_are_set(self):
        _, request_config = _request(max_tokens=8, max_completion_tokens=16).parse()
        self.assertEqual(request_config.max_tokens, 8)

    def test_neither_set_keeps_default(self):
        _, request_config = _request().parse()
        self.assertIsNone(request_config.max_tokens)


if __name__ == '__main__':
    unittest.main()
