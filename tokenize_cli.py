import argparse
import sys

from nmt_data_utils import get_tokenizer


LANGUAGE_MODELS = {
    "zh": "zh_core_web_sm",
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize text with the project's spaCy tokenizers.")
    parser.add_argument(
        "--lang",
        choices=sorted(LANGUAGE_MODELS),
        required=True,
        help="Language to tokenize: zh, en, or de.",
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Text to tokenize. If omitted, text is read from stdin.",
    )
    parser.add_argument(
        "--sep",
        default=" ",
        help="Separator used when printing tokens. Default: a single space.",
    )
    return parser.parse_args()


def read_text(args):
    if args.text:
        return " ".join(args.text)
    return sys.stdin.read().strip()


def main():
    args = parse_args()
    text = read_text(args)
    if not text:
        raise SystemExit("No input text provided.")

    tokenizer = get_tokenizer("spacy", language=LANGUAGE_MODELS[args.lang])
    tokens = tokenizer(text)
    print(args.sep.join(tokens))


if __name__ == "__main__":
    main()
