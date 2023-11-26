import os
from functools import lru_cache
from typing import Dict, List
import json
import re


@lru_cache()
def default_dict(path = "./data/vocab.json", charlen = 94):
    charlist_raw = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    charlist = charlist_raw[:charlen]
    word2idx = {"PAD_IDX":0,"SOS_IDX":1,"EOS_IDX":2}
    idx = len(word2idx)
    for t in charlist:
        word2idx[t] = idx
        idx += 1
    with open(path,"w") as f:
        json.dump(word2idx,f,indent=4,ensure_ascii=False)
    print("Length of vocab: ", idx)
    return path


class CROHMEVocab:

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2

    def __init__(self, charlen=94) -> None:
        dict_path = default_dict(charlen=charlen)
        with open(dict_path, "r") as f:
            self.word2idx = json.load(f)
        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}

        target_charset = "".join([key for key in self.word2idx.keys() if self.word2idx[key] > 3])
        self.lowercase_only = target_charset == target_charset.lower()
        self.uppercase_only = target_charset == target_charset.upper()
        self.unsupported = re.compile(f'[^{re.escape(target_charset)}]')
        print("target_charset ", target_charset)
        print("lowercase_only: ", self.lowercase_only)
        print(f"Init vocab with size: {len(self.word2idx)}")

    def words2indices(self, words: List[str]) -> List[int]:
        print("In word: ", words)
        if self.lowercase_only:
            words = words.lower()
        elif self.uppercase_only:
            words = words.upper()
        # Remove unsupported characters
        words = self.unsupported.sub('', words)
        # return [self.word2idx[w] for w in words.replace(" ","") if w in self.word2idx]
        return [self.word2idx[w] for w in words]

    def indices2words(self, id_list: List[int]) -> List[str]:
        return [self.idx2word[i] for i in id_list]

    def indices2label(self, id_list: List[int]) -> str:
        words = self.indices2words(id_list)
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)
