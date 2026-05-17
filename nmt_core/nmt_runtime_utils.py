import argparse
import csv
from datetime import datetime
from pathlib import Path

import torch


def parse_runtime_args(description, nearest_languages):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--nearest-k",
        type=int,
        default=5,
        help="Number of nearest tokens to show. Default: 5.",
    )
    parser.add_argument(
        "--nearest",
        action="store_true",
        help="Enter interactive nearest-neighbor query mode after loading a checkpoint or training.",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Enter interactive translation mode after loading a checkpoint or training.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for translation. 0 uses greedy decoding. Default: 0.",
    )
    return parser.parse_args()


def has_nearest_query(args, languages):
    return args.nearest


def should_load_checkpoint(args, languages):
    return args.translate or has_nearest_query(args, languages)


def run_translate_repl(translate_fn, temperature):
    mode = "greedy" if temperature == 0 else f"temperature={temperature}"
    print(f"Interactive translation mode ({mode}). Enter an empty line to quit.")
    while True:
        try:
            sentence = input("> ").strip()
        except EOFError:
            break
        if not sentence:
            break
        print(translate_fn(sentence, temperature=temperature))


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


def save_checkpoint(model, task_name, model_config, base_dir=None):
    path = checkpoint_path(task_name, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": dict(model_config),
    }, path)
    print(f"Checkpoint saved to {path}")
    return path


def load_checkpoint_if_exists(model, task_name, device, model_config, base_dir=None):
    path = checkpoint_path(task_name, base_dir)
    if not path.is_file():
        return False
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_config" not in checkpoint:
        raise RuntimeError(
            f"Checkpoint {path} does not contain model_config. "
            "Please retrain the model to create a new checkpoint."
        )

    saved_config = checkpoint["model_config"]
    mismatches = {
        key: (saved_config.get(key), value)
        for key, value in model_config.items()
        if saved_config.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={saved}, current={current}"
            for key, (saved, current) in mismatches.items()
        )
        raise RuntimeError(f"Checkpoint hyperparameters do not match current model: {details}")

    model.load_state_dict(checkpoint["model_state_dict"])
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


def run_nearest_repl(
    args,
    model,
    languages,
    vocab_transform,
    token_transform,
    src_language,
    tgt_language,
):
    language_list = ", ".join(languages)
    print(f"Interactive nearest-neighbor mode. Languages: {language_list}.")
    print("Enter '<language> <token>' to query, for example: 'en China'.")
    print("Enter an empty line to quit.")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break

        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            print(f"Please enter '<language> <token>'. Available languages: {language_list}.")
            continue

        language, word = parts
        if language not in languages:
            print(f"Unknown language {language!r}. Available languages: {language_list}.")
            continue

        try:
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
        except ValueError as exc:
            print(exc)
