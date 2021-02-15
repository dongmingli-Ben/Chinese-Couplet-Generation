from random import shuffle
from typing import List, Tuple

def read_raw(path: str) -> List[Tuple[str, str]]:
    max_len = 0
    replace_list = list('；。？！ 、：(·‘’')
    with open(path, 'r', encoding='utf-8') as f:
        cont = []
        pair = []
        for line in f:
            if line == '\n':
                if pair:
                    cont.append(tuple(pair))
                    pair = []
            else:
                line = line.strip('\n ')
                for ch in replace_list:
                    line = line.replace(ch, '，')
                while '，，' in line:
                    line = line.replace('，，', '，')
                line = line.strip('，')
                pair.append(line)
                max_len = max(max_len, len(line))
    return list(set(cont)), max_len

def write_to_file(pairs, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for p in pairs:
            f.write(p[0])
            f.write('\n')
            f.write(p[1])
            f.write('\n')
            f.write('\n')

def split(pairs, Shuffle=True, train=0.6, devel=0.2, test=0.2):
    if Shuffle:
        shuffle(pairs)
    num = len(pairs)
    trn = pairs[:int(train*num)]
    dev = pairs[int(train*num):int((train+devel)*num)]
    tst = pairs[int((train+devel)*num):]
    write_to_file(trn, './utils/train.txt')
    write_to_file(dev, './utils/dev.txt')
    write_to_file(tst, './utils/test.txt')

if __name__ == '__main__':
    ds, max_len = read_raw('./utils/raw.txt')
    split(ds)