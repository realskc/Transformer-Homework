# From https://data.statmt.org/news-commentary/v15/training/

from pathlib import Path
from typing import Sequence, Tuple, Union

from nmt_data_utils import ParallelTextIterableDataset

# 各个数据集的行数，调整这里来实验不同的数据集大小对模型翻译效果的影响
NUM_LINES = {
    'train': 290000,
    'valid': 20000,
    'test': 1000,
}

DATASET_NAME = "NEWS-Commentary"

_LANGUAGE_FILES = {
    'zh': 'news_zh.txt',
    'en': 'news_en.txt',
}


def _candidate_roots(root: Union[str, Path]):
    root_path = Path(root).expanduser()
    yield root_path
    yield root_path / DATASET_NAME
    yield root_path / "News-Commentary"


def _resolve_dataset_root(root: Union[str, Path], language_pair: Tuple[str, str]) -> Path:
    try:
        required_files = [_LANGUAGE_FILES[language] for language in language_pair]
    except KeyError as exc:
        raise ValueError("NEWSCOM only supports language_pair containing 'zh' and 'en'") from exc

    for candidate in _candidate_roots(root):
        if all((candidate / filename).is_file() for filename in required_files):
            return candidate

    root_path = Path(root).expanduser()
    dataset_root = root_path if root_path.name in {DATASET_NAME, "News-Commentary"} else root_path / DATASET_NAME
    dataset_root.mkdir(parents=True, exist_ok=True)
    expected = ", ".join(str(dataset_root / filename) for filename in required_files)
    raise FileNotFoundError(f"Cannot find News Commentary files. Expected: {expected}")


def _newcom_single(root: Union[str, Path], split: str, language_pair: Tuple[str, str]):
    if split not in NUM_LINES:
        raise ValueError(f"Unknown split {split!r}; expected one of {tuple(NUM_LINES)}")

    dataset_root = _resolve_dataset_root(root, language_pair)
    src_lang, tgt_lang = language_pair
    src_path = dataset_root / _LANGUAGE_FILES[src_lang]
    tgt_path = dataset_root / _LANGUAGE_FILES[tgt_lang]

    # Keep the original behavior: each split reads from the beginning of the
    # same aligned files and is limited only by NUM_LINES.
    return ParallelTextIterableDataset(src_path, tgt_path, num_lines=NUM_LINES[split])


def NEWSCOM(
    root: Union[str, Path] = ".data",
    split: Union[str, Sequence[str]] = 'train',
    language_pair: Tuple[str, str] = ('zh', 'en'),
):
    if isinstance(split, (tuple, list)):
        return tuple(_newcom_single(root, item, language_pair) for item in split)
    return _newcom_single(root, split, language_pair)
