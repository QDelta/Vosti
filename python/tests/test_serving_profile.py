import unittest
from unittest.mock import AsyncMock, patch

from scripts.serving_benchmark import profile_decode


class ProfileWaveTests(unittest.IsolatedAsyncioTestCase):
    async def test_preserves_prompt_usage_geometry(self):
        prompts = [dict(prompt='first', prompt_tokens=8192), dict(prompt='second', prompt_tokens=8193)]
        spec = dict(base_url='http://127.0.0.1:18300', served_name='model')
        send = AsyncMock(return_value=dict(success=True, output_tokens=256))
        with patch.object(profile_decode, 'send_request', send), \
             patch.object(profile_decode, 'read_server_metrics', AsyncMock(return_value={'engine': {}})):
            result = await profile_decode.wave(spec, prompts, 256)
        self.assertEqual(len(result['requests']), 2)
        for call, prompt in zip(send.call_args_list, prompts):
            self.assertEqual(call.kwargs['request'], {**prompt, 'max_tokens': 256})
            self.assertTrue(call.kwargs['require_usage'])

    async def test_incomplete_wave_is_rejected(self):
        spec = dict(base_url='http://127.0.0.1:18300', served_name='model')
        with patch.object(profile_decode, 'send_request', AsyncMock(return_value=dict(success=True, output_tokens=1))), \
             patch.object(profile_decode, 'read_server_metrics', AsyncMock(return_value={})), \
             self.assertRaisesRegex(RuntimeError, 'output lengths'):
            await profile_decode.wave(spec, [dict(prompt='test', prompt_tokens=1)], 256)


if __name__ == '__main__':
    unittest.main()
