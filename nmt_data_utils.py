"""Lightweight data utilities used by the NMT examples.

The goal is to keep the call sites in nmt_zh2en.py and nmt_de2en.py almost
unchanged while avoiding an extra text-data runtime dependency.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from torch.utils.data import IterableDataset, get_worker_info

TokenList = List[str]
LanguagePair = Tuple[str, str]
Split = Union[str, Sequence[str]]


class Vocab:
    """Minimal vocabulary compatible with the APIs used in this project."""

    def __init__(self, tokens: Iterable[str], default_index: Optional[int] = None) -> None:
        self.itos: List[str] = []
        self.stoi: Dict[str, int] = {}
        self.default_index = default_index

        for token in tokens:
            if token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    def __getitem__(self, token: str) -> int:
        if token in self.stoi:
            return self.stoi[token]
        if self.default_index is not None:
            return self.default_index
        raise RuntimeError(f"Token {token!r} not found in vocabulary")

    def __call__(self, tokens: Iterable[str]) -> List[int]:
        return [self[token] for token in tokens]

    def set_default_index(self, index: Optional[int]) -> None:
        self.default_index = index

    def get_default_index(self) -> Optional[int]:
        return self.default_index

    def lookup_tokens(self, indices: Iterable[int]) -> List[str]:
        tokens: List[str] = []
        for index in indices:
            index = int(index)
            if index < 0 or index >= len(self.itos):
                raise IndexError(f"Vocabulary index out of range: {index}")
            tokens.append(self.itos[index])
        return tokens

    def lookup_token(self, index: int) -> str:
        return self.lookup_tokens([index])[0]

    def get_itos(self) -> List[str]:
        return list(self.itos)

    def get_stoi(self) -> Dict[str, int]:
        return dict(self.stoi)


def build_vocab_from_iterator(
    iterator: Iterable[Iterable[str]],
    min_freq: int = 1,
    specials: Optional[Sequence[str]] = None,
    special_first: bool = True,
) -> Vocab:
    """Build a :class:`Vocab` from an iterator of token sequences.

    Tokens are ordered by descending frequency and then alphabetically.
    """

    counter: Counter[str] = Counter()
    for tokens in iterator:
        counter.update(tokens)

    specials = list(specials or [])
    special_set = set(specials)
    vocab_tokens = [
        token
        for token, freq in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        if freq >= min_freq and token not in special_set
    ]

    if special_first:
        ordered_tokens = specials + vocab_tokens
    else:
        ordered_tokens = vocab_tokens + specials

    return Vocab(ordered_tokens)


def get_tokenizer(tokenizer: str, language: Optional[str] = None) -> Callable[[str], TokenList]:
    """Return a tokenizer function.

    Only the spaCy mode used by this project is implemented. This keeps spaCy as
    the tokenizer backend while avoiding an extra wrapper dependency.
    """

    if tokenizer != "spacy":
        raise ValueError("Only tokenizer='spacy' is supported by this project")
    if language is None:
        raise ValueError("A spaCy model name must be provided via language=...")

    try:
        import spacy
    except ImportError as exc:
        raise ImportError("spaCy is required for tokenizer='spacy'") from exc

    nlp = spacy.load(language)

    def _tokenize(text: str) -> TokenList:
        return [token.text for token in nlp.tokenizer(text)]

    return _tokenize


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


class ParallelTextIterableDataset(IterableDataset):
    """Iterable dataset for aligned parallel text files.

    The dataset supports PyTorch DataLoader workers by sharding examples across
    workers. Newline characters are preserved; existing collate functions already
    strip them before tokenization.
    """

    def __init__(
        self,
        src_path: Union[str, Path],
        tgt_path: Union[str, Path],
        num_lines: Optional[int] = None,
        offset: int = 0,
    ) -> None:
        self.src_path = Path(src_path)
        self.tgt_path = Path(tgt_path)
        self.num_lines = num_lines
        self.offset = offset

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker.id
            num_workers = worker.num_workers

        start = max(0, self.offset)
        stop = None if self.num_lines is None else start + self.num_lines

        with self.src_path.open("r", encoding="utf-8") as src_file, self.tgt_path.open(
            "r", encoding="utf-8"
        ) as tgt_file:
            for line_idx, (src_line, tgt_line) in enumerate(zip(src_file, tgt_file)):
                if line_idx < start:
                    continue
                if stop is not None and line_idx >= stop:
                    break

                sample_idx = line_idx - start
                if sample_idx % num_workers == worker_id:
                    yield src_line, tgt_line

    def __len__(self) -> int:
        if self.num_lines is None:
            self.num_lines = min(_count_lines(self.src_path), _count_lines(self.tgt_path))
        return self.num_lines


MULTI30K_URL: Dict[str, str] = {
    "train": "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz",
    "valid": "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz",
    "test": "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz",
}

MULTI30K_FILES: Dict[str, Dict[str, str]] = {
    "train": {"de": "train.de", "en": "train.en"},
    "valid": {"de": "val.de", "en": "val.en"},
    "test": {"de": "test_2016_flickr.de", "en": "test_2016_flickr.en"},
}


def _safe_extract_tar(archive_path: Path, target_dir: Path) -> None:
    target_dir_resolved = target_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = (target_dir / member.name).resolve()
            if not (
                member_path == target_dir_resolved
                or str(member_path).startswith(str(target_dir_resolved) + os.sep)
            ):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
        archive.extractall(target_dir)


def _download_and_extract(url: str, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / Path(url).name

    try:
        with urllib.request.urlopen(url) as response, archive_path.open("wb") as archive_file:
            shutil.copyfileobj(response, archive_file)
        _safe_extract_tar(archive_path, target_dir)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download or extract Multi30k from {url!r}. "
            f"Please download the archive manually and place the extracted files under {target_dir}."
        ) from exc


def _find_file(root: Path, filename: str) -> Optional[Path]:
    direct_path = root / filename
    if direct_path.is_file():
        return direct_path

    matches = sorted(root.rglob(filename)) if root.exists() else []
    return matches[0] if matches else None


def _multi30k_root(root: Union[str, Path]) -> Path:
    root_path = Path(root).expanduser()
    return root_path if root_path.name == "Multi30k" else root_path / "Multi30k"


def _resolve_multi30k_paths(
    root: Union[str, Path], split: str, language_pair: LanguagePair
) -> Tuple[Path, Path]:
    if split not in MULTI30K_FILES:
        raise ValueError(f"Unknown split {split!r}; expected one of {tuple(MULTI30K_FILES)}")

    src_lang, tgt_lang = language_pair
    try:
        src_filename = MULTI30K_FILES[split][src_lang]
        tgt_filename = MULTI30K_FILES[split][tgt_lang]
    except KeyError as exc:
        raise ValueError("Multi30k only supports language_pair containing 'de' and 'en'") from exc

    dataset_root = _multi30k_root(root)
    src_path = _find_file(dataset_root, src_filename)
    tgt_path = _find_file(dataset_root, tgt_filename)

    if src_path is None or tgt_path is None:
        _download_and_extract(MULTI30K_URL[split], dataset_root)
        src_path = _find_file(dataset_root, src_filename)
        tgt_path = _find_file(dataset_root, tgt_filename)

    if src_path is None or tgt_path is None:
        raise FileNotFoundError(
            f"Cannot find Multi30k files {src_filename!r} and {tgt_filename!r} under {dataset_root}"
        )

    return src_path, tgt_path


def Multi30k(
    root: Union[str, Path] = ".data",
    split: Split = "train",
    language_pair: LanguagePair = ("de", "en"),
):
    """Return an iterable German-English Multi30k dataset.

    The signature mirrors the subset of Multi30k used by the training script.
    When the expected files are absent, the function downloads and extracts the
    standard archives with Python's standard library.
    """

    if isinstance(split, (tuple, list)):
        return tuple(Multi30k(root=root, split=item, language_pair=language_pair) for item in split)

    src_path, tgt_path = _resolve_multi30k_paths(root, split, language_pair)
    return ParallelTextIterableDataset(src_path, tgt_path)
