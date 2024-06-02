import os.path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from joblib import Memory

cache_path = os.path.join(os.path.dirname(__file__), ".cache")
memory = Memory(location=cache_path, verbose=0)


def load_vocab(vocab_path: str) -> Dict[str, int]:
    """Load vocabulary from txt file, and add them to jieba"""
    import jieba
    vocab_dict = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            word, freq = line.strip().split("\t")
            if word not in vocab_dict:
                vocab_dict[word] = len(vocab_dict)
                jieba.add_word(word, freq=int(freq))
    special_words = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    for special_word in special_words:
        vocab_dict[special_word] = len(vocab_dict)  # word: word_idx
    return vocab_dict


def tokenize(text: str, vocab: Dict[str, int]) -> List[int]:
    import jieba
    return [vocab.get(token, vocab['[UNK]']) for token in jieba.cut(text)]


def _create_batches(token_ids: List[int], max_seq_len: int, overlap: int, pad_id: int) -> Tuple[np.ndarray, np.ndarray]:
    stride = max_seq_len - overlap
    num_batches = (len(token_ids) + stride - 1) // stride  # ceil(len(token_ids) / stride)
    batch_input_ids = np.full((num_batches, max_seq_len), pad_id, dtype=np.int32)
    attention_mask = np.zeros((num_batches, max_seq_len), dtype=np.int32)

    for i in range(num_batches):
        start_idx = i * stride
        end_idx = min(start_idx + max_seq_len, len(token_ids))
        batch_input_ids[i, :end_idx - start_idx] = token_ids[start_idx:end_idx]
        attention_mask[i, :end_idx - start_idx] = 1

    return batch_input_ids, attention_mask


@memory.cache
def make_tensor_dataset(txt_data_path, vocabulary_path, max_len=256, overlap=128):
    vocab = load_vocab(vocabulary_path)

    with open(txt_data_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    with open(txt_data_path, 'r', encoding='utf-8') as f:
        input_ids = [tokenize(line.strip(), vocab) for line in tqdm(f, "processing data", total=total_lines)]
        data = [_create_batches(token_ids, max_len, overlap, vocab["[PAD]"]) for token_ids in input_ids]
        batch_input_ids, attention_mask = zip(*data)
        batch_input_ids = torch.from_numpy(np.concatenate(batch_input_ids))
        attention_mask = torch.from_numpy(np.concatenate(attention_mask))
        return TensorDataset(batch_input_ids, attention_mask)


if __name__ == '__main__':
    dataset = make_tensor_dataset(txt_data_path='data/data.txt', vocabulary_path='data/vocabulary.txt')
    print(dataset[0])
