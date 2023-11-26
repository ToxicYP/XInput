import os
from functools import lru_cache
from typing import Dict, List
import json

@lru_cache()
def default_dict(path = "./data/vocab.json"):
    charlist = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
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

    def __init__(self, dict_path: str = default_dict()) -> None:

        with open(dict_path, "r") as f:
            self.word2idx = json.load(f)
        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}

        print(f"Init vocab with size: {len(self.word2idx)}")

    def words2indices(self, words: List[str]) -> List[int]:
        return [self.word2idx[w] for w in words.replace(" ","") if w in self.word2idx]
        # return [self.word2idx[w] for w in words.replace(" ","")]

    def indices2words(self, id_list: List[int]) -> List[str]:
        return [self.idx2word[i] for i in id_list]

    def indices2label(self, id_list: List[int]) -> str:
        words = self.indices2words(id_list)
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)

if __name__ == "__main__":
    import json
    with open("/root/share/Project/ICDAR23_equation/BTTR/data/test.json","r") as f:
        testjson = json.load(f)
    vocab = CROHMEVocab()
    for k, v in testjson.items():
        print(k)
        label = v["caption"].split(" ")
        indlist = vocab.words2indices(label)

