import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


from scripts.common.tokenizers import tokenizer_artifact_sha256
from scripts.serving_benchmark.workloads import sharegpt as WORKLOADS


class FakeTokenizer:
    name_or_path = "/tmp/test-tokenizer"

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def __call__(self, texts, *, truncation, max_length):
        return {"input_ids": [self.encode(text)[:max_length] for text in texts]}


class ShareGptWorkloadTests(unittest.TestCase):
    def test_tokenizer_artifact_hashes_only_present_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tokenizer.json").write_bytes(b"tokenizer")
            (root / "unrelated.bin").write_bytes(b"ignored")
            hashes = tokenizer_artifact_sha256(root)

        self.assertEqual(
            hashes,
            {
                "tokenizer.json": (
                    "5f97e3774c51edd1d63706c2ec3826c564a067794770cdab0f8c4797971cacf9"
                )
            },
        )

    def test_missing_tokenizer_artifacts_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "no tokenizer artifacts"):
                tokenizer_artifact_sha256(Path(directory))

    def test_prefix_pairs_are_phased_and_preserve_full_page_sharing(self) -> None:
        tokenizer = FakeTokenizer()
        rows = [
            {
                "source_index": 7,
                "prompt": "a" * 100,
                "prompt_tokens": 100,
                "max_tokens": 8,
            }
        ]
        pairs, shared_blocks = WORKLOADS.prefix_pair_sample(
            rows, pair_count=1, seed=42, excluded=set(), tokenizer=tokenizer
        )

        self.assertEqual([row["arrival_phase"] for row in pairs], [0, 1])
        self.assertEqual([row["branch"] for row in pairs], ["donor", "consumer"])
        self.assertGreaterEqual(shared_blocks[0], 1)
        self.assertEqual(pairs[0]["shared_prefix_blocks"], shared_blocks[0])
        self.assertEqual(pairs[1]["shared_prefix_blocks"], shared_blocks[0])


def test_sharegpt_cli_preserves_seeded_disjoint_products(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps([
        {"conversations": [{"value": f"{i}: " + "a" * 100}, {"value": "answer"}]}
        for i in range(32)
    ]))
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("fixture")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda path: FakeTokenizer())))
    snapshots = []
    for name in ("first", "second"):
        output = tmp_path / name
        monkeypatch.setattr(sys, "argv", ["sharegpt.py", "--dataset", str(dataset),
            "--tokenizer-path", str(tokenizer), "--output-dir", str(output), "--requests", "4"])
        WORKLOADS.main()
        snapshots.append({path.name: path.read_bytes() for path in output.iterdir()})
    assert snapshots[0] == snapshots[1]
    manifest = json.loads(snapshots[0]["manifest.json"])
    assert len(manifest["workloads"]) == 4
    seen = set()
    for name, meta in manifest["workloads"].items():
        assert meta["n_requests"] == 4
        indices = set(meta["source_indices"])
        assert not seen & indices
        seen.update(indices)
        rows = json.loads(snapshots[0][name])["requests"]
        if "prefix_pairs" in name:
            assert [row["arrival_phase"] for row in rows] == [0, 0, 1, 1]
    capsys.readouterr()


if __name__ == "__main__":
    unittest.main()
