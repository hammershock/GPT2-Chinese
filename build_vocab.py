"""
build vocabulary from corpus lines

input:
corpus data: a txt, each line is a sentence
user dict(Optional): a jieba user dict txt file, each line is a user word. (This helps to cut the sentence)

output:
vocabulary file: a json file, contains vocabulary dictionary, including special tokens. (only word freq > 3 will be added to file)
word frequency stats: a json file, contains frequency of each word in vocabulary.
"""
import json
import os.path
from collections import Counter, OrderedDict
from typing import List, Dict, Iterator, Tuple
from typing import Counter as CounterType

from tqdm import tqdm


def data_loader(filepath) -> Iterator[Tuple[str, str]]:
    """
    My customized load_data function, returns preprocessed corpus lines
    :return:
    """
    with open(filepath, "r", encoding="utf-8") as f:
        gen = (line.strip().replace("[verified]", "") for line in f)
        for line in gen:
            parts = line.split("\t")
            yield parts[0], parts[1]


def _count_words(lines: List[str], words: List[str]) -> CounterType:
    counter = Counter()
    words = set(words)
    for line in lines:
        for word in words:
            counter[word] += line.count(word)
    return counter


def re_count_user_dict(filepath: str, lines) -> None:
    """
    Notice this function will cover the original user-dict file,
    remember to make a back-up to ensure your data safety!

    :param filepath:
    :param lines:
    :return:
    """
    words = list(load_jieba_user_dict(filepath).keys())
    counter = _count_words(lines, words)
    sorted_counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for word, count in sorted_counter:
            if count > 0:
                f.write(f"{word}\t{count}\n")


def load_jieba_user_dict(filepath, default_freq=10) -> Dict[str, int]:
    """
    jieba user dict is a txt file, each line is formatted like this:
    <word> [word-frequency] [part-of-speech]
    :param filepath:
    :param default_freq: default frequency of words(if not found in file)
    :return:
    """
    word_freq = {}
    with open(filepath, "r", encoding="utf-8") as f:
        gen = (line for line in f if line)
        for line in gen:
            parts = line.strip().split()
            if len(parts):
                word = parts[0]
                freq = int(parts[1]) if len(parts) >= 2 else default_freq
                word_freq[word] = freq
    return word_freq


if __name__ == "__main__":
    data_path = "./data/data.txt"
    user_dict_path = "./data/user_dict.txt"

    freq_output_path = "./data/vocab_freq.json"
    vocab_output_path = "./data/vocabulary.json"
    # the word frequency should be higher than this threshold to be added to the vocabulary
    min_word_freq = 3

    import jieba

    if os.path.exists(user_dict_path):
        # count user dict words frequency and load user dict
        lines = [content for bvid, content in data_loader(data_path)]
        re_count_user_dict(user_dict_path, lines)
        jieba.load_userdict(user_dict_path)

    # count token frequencies
    token_count = Counter()
    for bvid, content in tqdm(data_loader(data_path)):
        tokens = jieba.cut(content, cut_all=False)
        token_count.update(tokens)

    # build word frequency dict
    token_count = {token: count for token, count in token_count.items() if count > min_word_freq}
    sorted_token_count = OrderedDict(sorted(token_count.items(), key=lambda t: t[1], reverse=True))
    with open(freq_output_path, "w", encoding="utf-8") as f:
        json.dump(sorted_token_count, f, ensure_ascii=False, indent=4)

    # build vocabulary
    special_words = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    vocabulary = {word: idx for idx, word in enumerate(sorted_token_count.keys())}
    for special_word in special_words:
        vocabulary[special_word] = len(vocabulary)
    with open(vocab_output_path, "w", encoding="utf-8") as f:
        json.dump(vocabulary, f, ensure_ascii=False, indent=4)
