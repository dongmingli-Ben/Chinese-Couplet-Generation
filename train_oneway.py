from utils.preprocess import *
from model import OnewayTransformer

batch_size = 16
path = ''
n_layers = 4
d_ff = 1024
head = 6
model_dim = 300
dropout = 0.2
lr = 0.0001
max_epochs = 400
early_stopping = True

trn, dev, tst, lang, max_len = preprocess(batch_size)

# with open('./save/lang.txt', 'wb') as f:
#     pickle.dump(lang, f)

mdl = OnewayTransformer(path, lang, n_layers, model_dim, 
                        head, d_ff, dropout, lr=lr,
                        max_len=max_len)

best = 0
cnt = 0
for epoch in range(max_epochs):
    print('epoch', epoch)
    mdl.train()
    mdl.fit(trn)
    mdl.eval()
    with torch.no_grad():
        print("validating")
        bleu_score, format_acc = mdl.evaluate(dev)
        mdl.save(f'./save/{mdl.name}-{bleu_score}-{format_acc}.pt')
        print('model saved')
    if early_stopping:
        if bleu_score*.5 + format_acc*.5 > best:
            best = bleu_score*.5 + format_acc*.5
            cnt = 0
            continue
        cnt += 1
        if cnt >= 20:
            print('Early Stopping...')
            break