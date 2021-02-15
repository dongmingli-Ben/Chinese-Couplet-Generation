from lxml import html
import requests

def get_urls(path: str, tmp_file='tmp.txt'):
    tot = []
    queue = []
    queue.append('https://www.lz13.cn/lizhikouhao/58546.html')
    rule = '//div/div/div/div/div/div/a/@href'
    while len(queue) > 0:
        url = queue[0]
        tot.append(url)
        queue = queue[1:]
        with open(tmp_file, 'wb') as f:
            r = requests.get(url)
            f.write(r.content)
        with open(tmp_file, 'r', encoding='utf-8') as f:
            page = html.fromstring(f.read())
        title = page.xpath('//title/text()')
        if '联' not in title[0]:
            continue
        info = page.xpath(rule)
        for u in info:
            if u in tot or u in queue:
                continue
            queue.append(u)
        if len(tot) % 100 == 0:
            print('{} found, {} processed, {} unfinished'.format(len(tot)+len(queue), 
                                                                 len(tot), len(queue)))
            with open(path, 'w') as f:
                for u in tot:
                    f.write(u)
                    f.write('\n')
    with open(path, 'w') as f:
        for u in tot:
            f.write(u)
            f.write('\n')

def crawl(url: str, tmp_file: str, file: str):
    response = requests.get(url)
    with open(tmp_file, 'wb') as f:
        f.write(response.content)
    with open(tmp_file, 'r', encoding='utf-8') as f:
        html_format = html.fromstring(f.read())
    rule = '//div/div/div/div/div/p/text()'
    info = html_format.xpath(rule)
    f = open(file, 'a', encoding='utf-8')
    for ele in info:
        # import pdb; pdb.set_trace()
        s = ele.replace('\u3000', '')
        if s:
            if s[:3] == '上联：':
                f.write(s[3:])
                f.write('\n')
            elif s[:3] == '下联：':
                f.write(s[3:])
                f.write('\n')
                f.write('\n')
    f.close()
    
get_urls('urls.txt')

with open('urls.txt', 'r') as f:
    for url in f.readlines():
        crawl(url.strip('\n'), 'tmp.txt', 'raw.txt')

# filter valid couplets
contents = []
fir, sec = None, None
with open('raw.txt', 'r', encoding='utf-8') as f:
    for line in f:
        if fir == None:
            if line == '\n':
                continue
            fir = line.strip('\n')
        else:
            if sec == None:
                if line == '\n':
                    fir = None
                    continue
                sec = line.strip('\n')
            else:
                if line == '\n':
                    if fir is not None and len(fir) == len(sec):
                        contents.append(fir)
                        contents.append(sec)
                    fir, sec = None, None
with open('raw.txt', 'w', encoding='utf-8') as f:
    for i, line in enumerate(contents):
        f.write(line)
        f.write('\n')
        if i % 2:
            f.write('\n')
