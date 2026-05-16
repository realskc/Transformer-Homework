import argparse
import csv
from datetime import datetime
from pathlib import Path

import torch


def parse_runtime_args(description, nearest_languages):
    parser = argparse.ArgumentParser(description=description)
    for language, label in nearest_languages:
        parser.add_argument(
            f"--nearest-{language}",
            type=str,
            default=None,
            help=f"After training, find nearest {label} embeddings for this token.",
        )
    parser.add_argument(
        "--nearest-k",
        type=int,
        default=5,
        help="Number of nearest tokens to show. Default: 5.",
    )
    return parser.parse_args()


def has_nearest_query(args, languages):
    return any(getattr(args, f"nearest_{language}", None) is not None for language in languages)


def synchronize_device(device):
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "npu":
        reset_fn = getattr(torch.npu, "reset_peak_memory_stats", None)
        if reset_fn is not None:
            try:
                reset_fn(device)
            except TypeError:
                reset_fn()


def get_max_memory_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    if device.type == "npu":
        memory_fn = getattr(torch.npu, "max_memory_allocated", None)
        if memory_fn is not None:
            try:
                return memory_fn(device) / (1024 ** 2)
            except TypeError:
                return memory_fn() / (1024 ** 2)
    return None


def save_training_log(rows, task_name, base_dir=None):
    if base_dir is None:
        base_dir = Path.cwd()
    log_dir = Path(base_dir) / "log" / task_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = ["epoch", "train_loss", "val_loss", "max_memory_mb", "epoch_time"]
    with log_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Training log saved to {log_path}")


def checkpoint_path(task_name, base_dir=None):
    if base_dir is None:
        base_dir = Path.cwd()
    return Path(base_dir) / "checkpoints" / f"{task_name}.pt"


def save_checkpoint(model, task_name, base_dir=None):
    path = checkpoint_path(task_name, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved to {path}")
    return path


def load_checkpoint_if_exists(model, task_name, device, base_dir=None):
    path = checkpoint_path(task_name, base_dir)
    if not path.is_file():
        return False
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Checkpoint loaded from {path}")
    return True


def resolve_vocab_token(language, word, vocab_transform, token_transform):
    vocab = vocab_transform[language]
    stoi = vocab.get_stoi()
    if word in stoi:
        return word

    tokens = token_transform[language](word)
    if len(tokens) == 1 and tokens[0] in stoi:
        return tokens[0]

    raise ValueError(f"{word!r} is not a known {language} token.")


def get_embedding_weight(model, language, src_language, tgt_language):
    if language == src_language:
        return model.src_tok_emb.embedding.weight
    if language == tgt_language:
        return model.tgt_tok_emb.embedding.weight
    raise ValueError(f"Unknown language: {language}")


def nearest_tokens(
    model,
    language,
    word,
    vocab_transform,
    token_transform,
    src_language,
    tgt_language,
    top_k=5,
):
    token = resolve_vocab_token(language, word, vocab_transform, token_transform)
    vocab = vocab_transform[language]
    token_id = vocab[token]
    embedding = get_embedding_weight(model, language, src_language, tgt_language).detach()

    normalized_embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    scores = normalized_embedding @ normalized_embedding[token_id]

    excluded_tokens = ["<unk>", "<pad>", "<bos>", "<eos>", token]
    for excluded_token in excluded_tokens:
        if excluded_token in vocab:
            scores[vocab[excluded_token]] = float("-inf")

    k = min(top_k, len(vocab) - 1)
    top_scores, top_indices = torch.topk(scores, k)
    nearest = vocab.lookup_tokens(top_indices.cpu().tolist())
    return token, list(zip(nearest, top_scores.cpu().tolist()))


def print_nearest_tokens(
    model,
    language,
    word,
    top_k,
    vocab_transform,
    token_transform,
    src_language,
    tgt_language,
):
    token, results = nearest_tokens(
        model,
        language,
        word,
        vocab_transform,
        token_transform,
        src_language,
        tgt_language,
        top_k,
    )
    print(f"Nearest {language} tokens to {token!r}:")
    token_width = max(len(nearest_token) for nearest_token, _ in results)
    for nearest_token, score in results:
        print(f"  {nearest_token:<{token_width}}  {score:>7.4f}")


def run_nearest_queries(
    args,
    model,
    languages,
    vocab_transform,
    token_transform,
    src_language,
    tgt_language,
):
    for language in languages:
        word = getattr(args, f"nearest_{language}", None)
        if word is not None:
            print_nearest_tokens(
                model,
                language,
                word,
                args.nearest_k,
                vocab_transform,
                token_transform,
                src_language,
                tgt_language,
            )
