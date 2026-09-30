from collections import UserDict
import json
from pathlib import Path
import tempfile
import unittest

from vosti_kernels.serving_workload import (
    decode_serving_tokens,
    encode_serving_chat,
    encode_serving_prompt,
    load_generation_eos_token_ids,
    make_graph_warmup_prompt_ids,
    normalize_eos_token_ids,
    resolve_prompt_ids,
)


class FakeTokenizer:
    def encode(self, prompt):
        return [ord(character) for character in prompt]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.messages = messages
        self.template_options = (tokenize, add_generation_prompt)
        return UserDict({"input_ids": [17, 18, 19], "attention_mask": [1, 1, 1]})

    def decode(self, token_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        self.decode_options = (skip_special_tokens, clean_up_tokenization_spaces)
        return "/".join(str(token) for token in token_ids)


class ServingTokenizerTests(unittest.TestCase):
    def test_raw_prompt_encoding_is_nonempty_and_integer_normalized(self) -> None:
        tokenizer = FakeTokenizer()
        self.assertEqual(encode_serving_prompt(tokenizer, "ab"), [97, 98])
        for prompt in ("", None, ["not", "text"]):
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                encode_serving_prompt(tokenizer, prompt)

    def test_chat_encoding_uses_checkpoint_template(self) -> None:
        tokenizer = FakeTokenizer()
        encoded = encode_serving_chat(
            tokenizer,
            json.dumps([{"role": "user", "content": "hello"}]),
        )
        self.assertEqual(encoded, [17, 18, 19])
        self.assertEqual(tokenizer.messages[0]["role"], "user")
        self.assertEqual(tokenizer.template_options, (True, True))

    def test_generated_token_decoding_preserves_special_tokens(self) -> None:
        tokenizer = FakeTokenizer()
        self.assertEqual(decode_serving_tokens(tokenizer, [1, 2]), "1/2")
        self.assertEqual(tokenizer.decode_options, (False, False))
        for token_ids in ((1, 2), [1, True], [1, "2"]):
            with self.subTest(token_ids=token_ids), self.assertRaises(ValueError):
                decode_serving_tokens(tokenizer, token_ids)


class EosTokenMetadataTests(unittest.TestCase):
    def test_normalizes_scalar_and_alternative_token_lists(self) -> None:
        self.assertEqual(normalize_eos_token_ids(1), [1])
        self.assertEqual(normalize_eos_token_ids([1, 106]), [1, 106])

    def test_malformed_empty_or_duplicate_metadata_fails_closed(self) -> None:
        for value in (None, [], [1, 1], [1, -1], [1, True], [1, 2, 3, 4], "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_eos_token_ids(value)

    def test_loads_generation_config_without_tokenizer_collapsing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generation_config.json"
            path.write_text(json.dumps({"eos_token_id": [1, 106]}))
            self.assertEqual(
                load_generation_eos_token_ids(directory),
                [1, 106],
            )


class GraphWarmupPromptTests(unittest.TestCase):
    def test_rounds_preserve_shapes_and_within_round_prefixes(self) -> None:
        shared = list(range(80))
        prompts = [shared + [90, 91], shared + [92], list(range(20, 100))]

        rounds = make_graph_warmup_prompt_ids(prompts, vocab_size=256, rounds=2)

        self.assertEqual(len(rounds), 2)
        for transformed in rounds:
            self.assertEqual(
                [len(prompt) for prompt in transformed],
                [len(prompt) for prompt in prompts],
            )
            self.assertEqual(transformed[0][:80], transformed[1][:80])

        measured_pages = {tuple(prompt[:64]) for prompt in prompts if len(prompt) >= 64}
        first_pages = {
            tuple(prompt[:64]) for prompt in rounds[0] if len(prompt) >= 64
        }
        second_pages = {
            tuple(prompt[:64]) for prompt in rounds[1] if len(prompt) >= 64
        }
        self.assertTrue(measured_pages.isdisjoint(first_pages))
        self.assertTrue(measured_pages.isdisjoint(second_pages))
        self.assertTrue(first_pages.isdisjoint(second_pages))

    def test_invalid_parameters_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            make_graph_warmup_prompt_ids([[0]], vocab_size=1, rounds=2)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            make_graph_warmup_prompt_ids([[0]], vocab_size=2, rounds=-1)
        with self.assertRaisesRegex(ValueError, "block size"):
            make_graph_warmup_prompt_ids(
                [[0]], vocab_size=2, rounds=1, block_size=0
            )


class PromptInputTests(unittest.TestCase):
    def test_workload_token_ids_are_validated(self) -> None:
        prompt_ids, source = resolve_prompt_ids(
            [{"prompt_token_ids": [97, 98]}], ["ab"], FakeTokenizer()
        )
        self.assertEqual(prompt_ids, [[97, 98]])
        self.assertEqual(source, "workload")

    def test_missing_ids_are_resolved_before_engine_timing(self) -> None:
        prompt_ids, source = resolve_prompt_ids([{}], ["ab"], FakeTokenizer())
        self.assertEqual(prompt_ids, [[97, 98]])
        self.assertEqual(source, "tokenizer")

    def test_partial_or_incorrect_ids_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "every request or none"):
            resolve_prompt_ids(
                [{"prompt_token_ids": [97]}, {}], ["a", "b"], FakeTokenizer()
            )
        with self.assertRaisesRegex(ValueError, "disagree"):
            resolve_prompt_ids(
                [{"prompt_token_ids": [98]}], ["a"], FakeTokenizer()
            )


if __name__ == "__main__":
    unittest.main()
