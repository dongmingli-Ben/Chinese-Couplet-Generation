from typing import List, Tuple
import torch
import torch.utils.data as data
import logging

logging.basicConfig(level=logging.DEBUG)

UNK_token = 0
PAD_token = 1
SOS_token = 2
EOS_token = 3
SEP_token = 4

class Lang:
    """As a dictionary"""

    def __init__(self):
        self.word2index = {'UNK': UNK_token,
                           'PAD': PAD_token,
                           'SOS': SOS_token,
                           'EOS': EOS_token,
                           '，':  SEP_token,
                           }
        self.index2word = {UNK_token: 'UNK',
                           PAD_token: 'PAD',
                           SOS_token: 'SOS',
                           EOS_token: 'EOS',
                           SEP_token: '，',
                           }
        self.vocab_size = 5
        print('Loading pretrained vectors')
        self.build_dict('sgns.sikuquanshu.word')
        print('Loaded')

    def build_dict(self, path):
        vectors = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            f.readline()
            for line in f:
                if ' ' not in line: continue
                word, s = line.split(maxsplit=1)
                if word in self.word2index:
                    continue
                self.word2index[word] = self.vocab_size
                self.index2word[self.vocab_size] = word
                self.vocab_size += 1
                vector = [eval(num) for num in s.strip('\n').split()]
                vectors.append(vector)
        vectors = torch.Tensor(vectors).cuda()
        self.vectors = vectors


    def add_words(self, words: str):
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = self.vocab_size
                self.index2word[self.vocab_size] = word
                self.vocab_size += 1

    def words2indices(self, words: str, tar=False):
        indices = [self.word2index.get(word, UNK_token) for word in words]
        if tar:
            return [SOS_token] + indices + [EOS_token]
        return indices


class Dataset:
    """For DataLoader"""

    def __init__(self, pairs, lang: Lang):
        self.pairs = pairs
        self.lang = lang

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        p = {}
        p['src'] = self.lang.words2indices(pair[0], False)
        p['tar'] = self.lang.words2indices(pair[1], True)
        p['tar_oneway'] = self.lang.words2indices(pair[1], False)
        p['tar_length'] = len(pair[1]) + 1   # EOS_token
        p['tar_text'] = pair[1]
        return p

    def get_batch(self, pairs):

        def merge(seqs: List[List[int]]) -> torch.Tensor:
            max_len = max(list(map(len, seqs)))
            tensor = torch.ones((len(seqs), max_len)).long()
            for i in range(len(seqs)):
                tensor[i, :len(seqs[i])] = torch.tensor(seqs[i])
            return tensor

        ds = {}
        for key in pairs[0].keys():
            ds[key] = []
        for p in pairs:
            for key in p.keys():
                ds[key].append(p[key])
        src = merge(ds['src'])
        tar = merge(ds['tar'])
        tar_oneway = merge(ds['tar_oneway'])
        src = src.cuda()
        tar = tar.cuda()
        tar_oneway = tar_oneway.cuda()
        batch = {'src': src, 'tar': tar,
                 'tar_oneway': tar_oneway,
                 'tar_length': ds['tar_length'],
                 'tar_text': ds['tar_text']}
        return batch

def read_file(path: str) -> List[Tuple[str, str]]:
    max_len = 0
    with open(path, 'r', encoding='utf-8') as f:
        cont = []
        pair = []
        for line in f:
            if line == '\n':
                if pair:
                    cont.append(tuple(pair))
                    pair = []
            else:
                pair.append(line.strip('\n'))
                max_len = max(max_len, len(line.strip('\n')))
    return list(set(cont)), max_len

def get_dataset(path, lang: Lang, b_sz=32, shuffle=False):
    pairs, max_len = read_file(path)
    # if add_to_dict:
    #     for p in pairs:
    #         lang.add_words(p[0])
    #         lang.add_words(p[1])
    ds = Dataset(pairs, lang)
    return data.DataLoader(ds, b_sz, shuffle, collate_fn=ds.get_batch), \
        len(pairs), max_len


def preprocess(batch_size=16):
    lang = Lang()
    trn, trn_num, max_len_trn = get_dataset('./utils/train.txt', lang, 
                                            b_sz=batch_size, shuffle=True)
    logging.info(f'number of samples {trn_num}')
    dev, dev_num, max_len_dev = get_dataset('./utils/dev.txt', lang, 
                                            b_sz=batch_size)
    logging.info(f'number of samples {dev_num}')
    tst, tst_num, max_len_tst = get_dataset('./utils/test.txt', lang, 
                                            b_sz=batch_size)
    max_len = max(max_len_trn, max_len_dev, max_len_tst) + 1 # EOS
    logging.info(f'number of samples {tst_num}')
    logging.info(f'number of words {lang.vocab_size}')
    logging.info(f'max length {max_len}')
    return trn, dev, tst, lang, max_len


if __name__ == '__main__':
    trn, dev, tst, lang, max_len = preprocess(4)
    for i, batch in enumerate(trn):
        if i == 0:
            print(batch)
            # break