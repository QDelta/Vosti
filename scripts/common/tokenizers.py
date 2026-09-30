"""Local tokenizer fingerprints shared by benchmark workload generators."""

import hashlib
from pathlib import Path


TOKENIZER_ARTIFACT_NAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
)


def tokenizer_artifact_sha256(tokenizer_path: Path) -> dict[str, str]:
    """Hash the local tokenizer artifacts that determine prompt token IDs."""
    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name in TOKENIZER_ARTIFACT_NAMES
        if (path := tokenizer_path / name).is_file()
    }
    if not hashes:
        raise ValueError(f"no tokenizer artifacts found under {tokenizer_path}")
    return hashes
