from typing import Dict, List
import re
import unicodedata


all_charlist = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

class CROHMEVocab:
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2

    def __init__(self, charlen=94) -> None:
        idx = 3
        target_charset = all_charlist[:charlen]
        word2idx = {"PAD_IDX": self.PAD_IDX,"SOS_IDX":self.SOS_IDX,"EOS_IDX":self.EOS_IDX}
        for i in range(len(word2idx),idx):
            word = "MEANLESS" + str(i)
            word2idx[word] = i
        for t in target_charset:
            word2idx[t] = idx
            idx += 1
        
        self.target_charset = target_charset
        self.word2idx = word2idx
        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}
        self.unsupported = re.compile(f'[^{re.escape(target_charset)}]')

        print(f"Init vocab with size: {len(self.word2idx)}")

    def label2indices(self, label: str) -> List[int]:
        # Normally, whitespace is removed from the labels.
        label = ''.join(label.split())
        # Normalize unicode composites (if any) and convert to compatible ASCII characters
        label = unicodedata.normalize('NFKD', label).encode('ascii', 'ignore').decode()
        label = self.unsupported.sub('', label)             
        return [self.word2idx[w] for w in label]

    def indices2label(self, id_list: List[int]) -> str:
        return "".join([self.idx2word[i] for i in id_list])

    def __len__(self):
        return len(self.word2idx)

