import pickle
from model import VanillaTransformer, TwowayTransformer, MemoryTransformer
from utils.preprocess import Lang

path = './save/TwowayTransformer-best.pt'
lang = Lang()

model = TwowayTransformer(path, lang)

while True:
    src = input('请输入上联（按q退出）:')
    if src == 'q': break
    try:
        out, status = model.generate_response(src)
        print(f'上联：{src}')
        print(f'下联：{out}')
        if not status:
            print('生成失败，请换一上联')
    except:
        print('发生未知错误，请再试一次')
    print()