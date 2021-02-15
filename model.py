from typing import List
from utils.preprocess import PAD_token, UNK_token, EOS_token, SOS_token
from transformer import make_MemoryModel, make_OnewayModel, make_TwowayModel, make_model, subsequent_mask
import torch.nn as nn
import torch
import torch.functional as F
from tqdm import tqdm
from measure.bleu import corpus_bleu

def sequence_mask(sequence_length, max_len=None):   
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long() 
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len) 
    seq_range_expand = torch.autograd.Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(log_probs, target, length):  
    """
    Args:
        log_probs: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """
    length = torch.autograd.Variable(torch.LongTensor(length)).cuda()

    log_probs_flat = log_probs.view(-1, log_probs.size(-1)) ## -1 means infered from other dimentions 
    target_flat = target.contiguous().view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))  
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss


class VanillaTransformer(nn.Module):

    def __init__(self, path, lang, n_layers=3, 
                 d_model=128, head=4, 
                 d_ff=512, dropout=0.2, lr=0.001,
                 max_len=30):
        super(VanillaTransformer, self).__init__()
        if path:
            self.model = torch.load(path)
        else:
            self.model = make_model(lang.vectors,
                                    N=n_layers,
                                    d_model=d_model,
                                    d_ff=d_ff,
                                    h=head,
                                    dropout=dropout)
        self.lang = lang
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.max_len = min(30, max_len)
        self.model.cuda()
        self.name = 'VanillaTransformer'
        # import pdb; pdb.set_trace()

    def save(self, path: str):
        torch.save(self.model, path)

    def fit_batch(self, batch) -> float:
        src = batch['src']
        tar = batch['tar']
        decoder_input = tar[:, :-1]
        tar = tar[:, 1:]
        src_mask = (src != PAD_token).unsqueeze(-2)
        tar_mask = (decoder_input != PAD_token).unsqueeze(-2)
        tar_mask = tar_mask & \
            subsequent_mask(decoder_input.size(-1)).type_as(tar_mask.data)
        decoder_output = self.model(src, decoder_input,
                                    src_mask, tar_mask)
        log_probs = self.model.generator(decoder_output)
        length = batch['tar_length']
        loss = masked_cross_entropy(log_probs.contiguous(), tar, length)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        return loss

    def fit(self, dataset):
        pbar = tqdm(dataset, total=len(dataset))
        tot_loss = 0
        num = 0
        for batch in pbar:
            loss = self.fit_batch(batch)
            tot_loss += loss
            num += 1
            pbar.set_description(f'L: {tot_loss/num:<.2f}')

    def decode_batch(self, src: torch.Tensor) -> List[str]:
        batch_size = src.size(0)
        decoder_input = torch.LongTensor([SOS_token]*batch_size).cuda()
        decoder_input = decoder_input.unsqueeze(1)
        src_mask = (src != PAD_token).unsqueeze(-2)
        for i in range(self.max_len):
            tar_mask = (decoder_input != PAD_token)
            tar_mask = tar_mask.unsqueeze(-2)
            tar_mask = tar_mask & \
                subsequent_mask(decoder_input.size(-1)).type_as(tar_mask.data)
            decoder_output = self.model(src, 
                                        decoder_input,
                                        src_mask,
                                        tar_mask)
            log_probs = self.model.generator(decoder_output[:, -1, :])
            topv, topi = torch.topk(log_probs, 1)
            # topi = topi.squeeze(1)
            decoder_input = torch.cat([decoder_input, topi], dim=1)
        # decode to string
        decoded_chs = []
        decoded_str = []
        for i in range(batch_size):
            code = decoder_input[i, :]
            for j in range(1, self.max_len):
                if code[j] == EOS_token:
                    decoded_str.append(' '.join(decoded_chs))
                    decoded_chs.clear()
                    break
                decoded_chs.append(self.lang.index2word[code[j].item()])
            else:
                decoded_str.append(' '.join(decoded_chs))
                decoded_chs.clear()
        return decoded_str


    def evaluate(self, dataset):
        decoded_output, tar_text = [], []
        pbar = tqdm(dataset, total=len(dataset))
        # import pdb; pdb.set_trace()
        for batch in pbar:
            decoded_output += self.decode_batch(batch['src'])
            text = batch['tar_text']
            text = list(map(lambda s: ' '.join(list(s)), text))
            tar_text += text
        with open('hypothesis.txt', 'w', encoding='utf-8') as f:
            for out in decoded_output:
                f.write(''.join(out.split()))
                f.write('\n')
        with open('reference.txt', 'w', encoding='utf-8') as f:
            for out in tar_text:
                f.write(''.join(out.split()))
                f.write('\n')
        # import pdb; pdb.set_trace()
        total = 0
        format_correct = 0
        for out, tar in zip(decoded_output, tar_text):
            total += 1
            out_list, tar_list = out.split(), tar.split()
            if len(out_list) == len(tar_list):
                for out_ch, tar_ch in zip(out_list, tar_list):
                    if (tar_ch == '，' or out_ch == '，') and out_ch != tar_ch:
                        break
                else:
                    format_correct += 1
        format_acc = format_correct / total
        # import pdb; pdb.set_trace()
        tar_text = list(map(lambda s: [s], tar_text))
        bleu_score = corpus_bleu(decoded_output, tar_text)[0][0]*100
        print(f'BLEU score: {bleu_score:.2f}')
        print(f'Format acc: {format_acc:.2f}')
        return bleu_score, format_acc

    def generate_response(self, src: str, wait=10) -> str:
        import time
        t0 = time.time()
        length = len(src)
        src_text = src
        src = self.lang.words2indices(src)
        src = torch.tensor(src).unsqueeze(0).cuda()
        valid = False
        success = True
        decoded_str = []
        while not valid:
            if time.time() - t0 > wait:
                success = False
                break
            decoder_input = torch.tensor([[SOS_token]]).cuda()
            for i in range(length+1):
                tar_mask = subsequent_mask(decoder_input.size(-1)).type(torch.bool).cuda()
                decoder_output = self.model(src,
                                            decoder_input,
                                            None,
                                            tar_mask)
                log_probs = self.model.generator(decoder_output[:, -1, :])
                probs = torch.exp(log_probs)
                sampler = torch.distributions.categorical.Categorical(
                    probs=probs.squeeze()
                )
                index = sampler.sample().item()
                decoder_input = torch.cat([decoder_input, torch.tensor([[index]]).cuda()], dim=1)
            decoded_str = []
            # import pdb; pdb.set_trace()
            j = decoder_input.size(1) - 1
            if decoder_input[0, j] != EOS_token:
                continue
            for j in range(1, decoder_input.size(1)-1):
                decoded_str.append(self.lang.index2word[decoder_input[0, j].item()])
                if (src_text[j-1] == '，' or decoded_str[-1] == '，') \
                        and src_text[j-1] != decoded_str[-1]:
                    break
                if decoder_input[0, j] == EOS_token \
                        and j != length + 1:
                    break
            else:
                valid = True
        if not success: decoded_str = []
        return ''.join(decoded_str), success



class OnewayTransformer(nn.Module):

    def __init__(self, path, lang, n_layers=3, 
                 d_model=128, head=4, 
                 d_ff=512, dropout=0.2, lr=0.001,
                 max_len=30):
        super(OnewayTransformer, self).__init__()
        if path:
            self.model = torch.load(path)
        else:
            self.model = make_OnewayModel(lang.vectors,
                                          N=n_layers,
                                          d_model=d_model,
                                          d_ff=d_ff,
                                          h=head,
                                          dropout=dropout)
        self.lang = lang
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.max_len = min(30, max_len)
        self.model.cuda()
        self.name = 'OnewayTransformer'
        # import pdb; pdb.set_trace()

    def save(self, path: str):
        torch.save(self.model, path)

    def fit_batch(self, batch) -> float:
        src = batch['src']
        tar = batch['tar_oneway']
        # decoder_input = tar[:, :-1]
        # tar = tar[:, 1:]
        src_mask = (src != PAD_token).unsqueeze(-2)
        # tar_mask = tar_mask & \
        #     subsequent_mask(decoder_input.size(-1)).type_as(tar_mask.data)
        decoder_output = self.model(src, src_mask)
        log_probs = self.model.generator(decoder_output)
        length = list(map(lambda x: x-1, batch['tar_length']))   # EOS
        loss = masked_cross_entropy(log_probs.contiguous(), tar, length)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        return loss

    def fit(self, dataset):
        pbar = tqdm(dataset, total=len(dataset))
        tot_loss = 0
        num = 0
        for batch in pbar:
            loss = self.fit_batch(batch)
            tot_loss += loss
            num += 1
            pbar.set_description(f'L: {tot_loss/num:<.2f}')

    def decode_batch(self, src: torch.Tensor) -> List[str]:
        batch_size = src.size(0)
        src_mask = (src != PAD_token).unsqueeze(-2)
        src_length = torch.sum(src_mask, dim=2).squeeze()
        decoder_output = self.model(src, src_mask)
        log_probs = self.model.generator(decoder_output)
        # decode to string
        decoded_chs = []
        decoded_str = []
        topv, topi = log_probs.topk(1)
        for i in range(batch_size):
            code = topi[i, :]
            length = src_length[i].item()
            for j in range(length):
                decoded_chs.append(self.lang.index2word[code[j].item()])
            else:
                decoded_str.append(' '.join(decoded_chs))
                decoded_chs.clear()
        return decoded_str


    def evaluate(self, dataset):
        decoded_output, tar_text = [], []
        pbar = tqdm(dataset, total=len(dataset))
        # import pdb; pdb.set_trace()
        for batch in pbar:
            decoded_output += self.decode_batch(batch['src'])
            text = batch['tar_text']
            text = list(map(lambda s: ' '.join(list(s)), text))
            tar_text += text
        with open('hypothesis.txt', 'w', encoding='utf-8') as f:
            for out in decoded_output:
                f.write(''.join(out.split()))
                f.write('\n')
        with open('reference.txt', 'w', encoding='utf-8') as f:
            for out in tar_text:
                f.write(''.join(out.split()))
                f.write('\n')
        # import pdb; pdb.set_trace()
        total = 0
        format_correct = 0
        for out, tar in zip(decoded_output, tar_text):
            total += 1
            out_list, tar_list = out.split(), tar.split()
            if len(out_list) == len(tar_list):
                for out_ch, tar_ch in zip(out_list, tar_list):
                    if (tar_ch == '，' or out_ch == '，') and out_ch != tar_ch:
                        break
                else:
                    format_correct += 1
        format_acc = format_correct / total
        # import pdb; pdb.set_trace()
        tar_text = list(map(lambda s: [s], tar_text))
        bleu_score = corpus_bleu(decoded_output, tar_text)[0][0]*100
        print(f'BLEU score: {bleu_score:.2f}')
        print(f'Format acc: {format_acc:.2f}')
        return bleu_score, format_acc

    def generate_response(self, src: str, wait=10) -> str:
        import time
        t0 = time.time()
        length = len(src)
        src_text = src
        src = self.lang.words2indices(src)
        src = torch.tensor(src).unsqueeze(0).cuda()
        valid = False
        success = True
        decoded_str = []
        while not valid:
            if time.time() - t0 > wait:
                success = False
                break
            decoder_output = self.model(src, None)
            log_probs = self.model.generator(decoder_output)
            probs = torch.exp(log_probs).squeeze()
            decoded_str = []
            for i in range(length):
                sampler = torch.distributions.categorical.Categorical(
                    probs=probs[i, :]
                )
                index = sampler.sample().item()
                decoded_str.append(self.lang.index2word[index])
                if (src_text[i] == '，' or decoded_str[-1] == '，') \
                        and src_text[i] != decoded_str[-1]:
                    break
            else:
                valid = True
        if not success: decoded_str = []
        return ''.join(decoded_str), success



class TwowayTransformer(nn.Module):

    def __init__(self, path, lang, n_layers=3, 
                 d_model=128, head=4, 
                 d_ff=512, dropout=0.2, lr=0.001,
                 max_len=30):
        super(TwowayTransformer, self).__init__()
        if path:
            self.model = torch.load(path)
        else:
            self.model = make_TwowayModel(lang.vectors,
                                          N=n_layers,
                                          d_model=d_model,
                                          d_ff=d_ff,
                                          h=head,
                                          dropout=dropout)
        self.lang = lang
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.max_len = min(30, max_len)
        self.model.cuda()
        self.name = 'TwowayTransformer'
        # import pdb; pdb.set_trace()

    def save(self, path: str):
        torch.save(self.model, path)

    def fit_batch(self, batch) -> float:
        decoder_input = batch['src']
        tar = batch['tar_oneway']
        sos = torch.LongTensor([SOS_token]*tar.size(0)).unsqueeze(1).cuda()
        context = torch.cat([sos ,tar[:, :-1]], dim=1)
        context_mask = (context != PAD_token).unsqueeze(-2)
        context_mask = context_mask & \
            subsequent_mask(decoder_input.size(-1)).type_as(context_mask)
        decoder_mask = (decoder_input != PAD_token).unsqueeze(-2)
        decoder_mask = decoder_mask & \
            subsequent_mask(decoder_input.size(-1)).type_as(decoder_mask)
        # import pdb; pdb.set_trace()
        decoder_output = self.model(context, decoder_input,
                                    context_mask, decoder_mask)
        log_probs = self.model.generator(decoder_output)
        length = list(map(lambda x: x-1, batch['tar_length']))   # EOS
        loss = masked_cross_entropy(log_probs.contiguous(), tar, length)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        return loss

    def fit(self, dataset):
        pbar = tqdm(dataset, total=len(dataset))
        tot_loss = 0
        num = 0
        for batch in pbar:
            loss = self.fit_batch(batch)
            tot_loss += loss
            num += 1
            pbar.set_description(f'L: {tot_loss/num:<.2f}')

    def decode_batch(self, src: torch.Tensor) -> List[str]:
        batch_size = src.size(0)
        max_len = src.size(1)
        context = torch.LongTensor([SOS_token]*batch_size).cuda()
        context = context.unsqueeze(1)
        for i in range(max_len):
            context_mask = (context != PAD_token)
            context_mask = context_mask.unsqueeze(-2)
            context_mask = context_mask & \
                subsequent_mask(src.size(-1))[:, :, :i+1].type_as(context_mask)
            decoder_mask = (src != PAD_token).unsqueeze(-2)
            decoder_mask = decoder_mask & \
                subsequent_mask(src.size(-1)).type_as(decoder_mask)
            # import pdb; pdb.set_trace()
            decoder_output = self.model(context, 
                                        src,
                                        context_mask,
                                        decoder_mask)
            log_probs = self.model.generator(decoder_output[:, i, :])
            topv, topi = torch.topk(log_probs, 1)
            # topi = topi.squeeze(1)
            context = torch.cat([context, topi], dim=1)
        # decode to string
        decoded_chs = []
        decoded_str = []
        src_length = torch.sum(src != PAD_token, dim=1)
        for i in range(batch_size):
            code = context[i, :]
            for j in range(1, src_length[i].item()+1):
                decoded_chs.append(self.lang.index2word[code[j].item()])
            else:
                decoded_str.append(' '.join(decoded_chs))
                decoded_chs.clear()
        return decoded_str


    def evaluate(self, dataset):
        decoded_output, tar_text = [], []
        pbar = tqdm(dataset, total=len(dataset))
        # import pdb; pdb.set_trace()
        for batch in pbar:
            decoded_output += self.decode_batch(batch['src'])
            text = batch['tar_text']
            text = list(map(lambda s: ' '.join(list(s)), text))
            tar_text += text
        with open('hypothesis.txt', 'w', encoding='utf-8') as f:
            for out in decoded_output:
                f.write(''.join(out.split()))
                f.write('\n')
        with open('reference.txt', 'w', encoding='utf-8') as f:
            for out in tar_text:
                f.write(''.join(out.split()))
                f.write('\n')
        # import pdb; pdb.set_trace()
        total = 0
        format_correct = 0
        for out, tar in zip(decoded_output, tar_text):
            total += 1
            out_list, tar_list = out.split(), tar.split()
            if len(out_list) == len(tar_list):
                for out_ch, tar_ch in zip(out_list, tar_list):
                    if (tar_ch == '，' or out_ch == '，') and out_ch != tar_ch:
                        break
                else:
                    format_correct += 1
        format_acc = format_correct / total
        # import pdb; pdb.set_trace()
        tar_text = list(map(lambda s: [s], tar_text))
        bleu_score = corpus_bleu(decoded_output, tar_text)[0][0]*100
        print(f'BLEU score: {bleu_score:.2f}')
        print(f'Format acc: {format_acc:.2f}')
        return bleu_score, format_acc

    def generate_response(self, src: str, wait=10) -> str:
        import time
        t0 = time.time()
        length = len(src)
        src_text = src
        src = self.lang.words2indices(src)
        src = torch.tensor(src).unsqueeze(0).cuda()
        valid = False
        success = True
        decoded_str = []
        while not valid:
            if time.time() - t0 > wait:
                success = False
                break
            context = torch.tensor([[SOS_token]]).cuda()
            for i in range(length):
                context_mask = subsequent_mask(src.size(-1))[:, :, :i+1].type(torch.bool).cuda()
                decoder_mask = subsequent_mask(length).type(torch.bool).cuda()
                decoder_output = self.model(context,
                                            src,
                                            context_mask,
                                            decoder_mask)
                log_probs = self.model.generator(decoder_output[:, i, :])
                probs = torch.exp(log_probs)
                sampler = torch.distributions.categorical.Categorical(
                    probs=probs.squeeze()
                )
                index = sampler.sample().item()
                context = torch.cat([context, torch.tensor([[index]]).cuda()], dim=1)
            decoded_str = []
            # import pdb; pdb.set_trace()
            for j in range(1, context.size(1)):
                decoded_str.append(self.lang.index2word[context[0, j].item()])
                if (src_text[j-1] == '，' or decoded_str[-1] == '，') \
                        and src_text[j-1] != decoded_str[-1]:
                    break
            else:
                valid = True
        if not success: decoded_str = []
        return ''.join(decoded_str), success


class MemoryTransformer(nn.Module):

    def __init__(self, path, lang, n_layers=3, 
                 d_model=128, head=4, 
                 d_ff=512, dropout=0.2, lr=0.001,
                 max_len=30):
        super(MemoryTransformer, self).__init__()
        if path:
            self.model = torch.load(path)
        else:
            self.model = make_MemoryModel(lang.vectors,
                                          N=n_layers,
                                          d_model=d_model,
                                          d_ff=d_ff,
                                          h=head,
                                          dropout=dropout)
        self.lang = lang
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.max_len = min(30, max_len)
        self.model.cuda()
        self.name = 'MemoryTransformer'
        # import pdb; pdb.set_trace()

    def save(self, path: str):
        torch.save(self.model, path)

    def fit_batch(self, batch) -> float:
        decoder_input = batch['src']
        tar = batch['tar_oneway']
        sos = torch.LongTensor([SOS_token]*tar.size(0)).unsqueeze(1).cuda()
        context = torch.cat([sos ,tar[:, :-1]], dim=1)
        context_mask = (context != PAD_token).unsqueeze(-2)
        context_mask = context_mask & \
            subsequent_mask(decoder_input.size(-1)).type_as(context_mask)
        decoder_mask = (decoder_input != PAD_token).unsqueeze(-2)
        # decoder_mask = decoder_mask & \
        #     subsequent_mask(decoder_input.size(-1)).type_as(decoder_mask)
        # import pdb; pdb.set_trace()
        decoder_output = self.model(context, decoder_input,
                                    context_mask, decoder_mask)
        log_probs = self.model.generator(decoder_output)
        length = list(map(lambda x: x-1, batch['tar_length']))   # EOS
        loss = masked_cross_entropy(log_probs.contiguous(), tar, length)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        return loss

    def fit(self, dataset):
        pbar = tqdm(dataset, total=len(dataset))
        tot_loss = 0
        num = 0
        for batch in pbar:
            loss = self.fit_batch(batch)
            tot_loss += loss
            num += 1
            pbar.set_description(f'L: {tot_loss/num:<.2f}')

    def decode_batch(self, src: torch.Tensor) -> List[str]:
        batch_size = src.size(0)
        max_len = src.size(1)
        context = torch.LongTensor([SOS_token]*batch_size).cuda()
        context = context.unsqueeze(1)
        for i in range(max_len):
            context_mask = (context != PAD_token)
            context_mask = context_mask.unsqueeze(-2)
            context_mask = context_mask & \
                subsequent_mask(src.size(-1))[:, :, :i+1].type_as(context_mask)
            decoder_mask = (src != PAD_token).unsqueeze(-2)
            # decoder_mask = decoder_mask & \
            #     subsequent_mask(src.size(-1)).type_as(decoder_mask)
            # import pdb; pdb.set_trace()
            decoder_output = self.model(context, 
                                        src,
                                        context_mask,
                                        decoder_mask)
            log_probs = self.model.generator(decoder_output[:, i, :])
            topv, topi = torch.topk(log_probs, 1)
            # topi = topi.squeeze(1)
            context = torch.cat([context, topi], dim=1)
        # decode to string
        decoded_chs = []
        decoded_str = []
        src_length = torch.sum(src != PAD_token, dim=1)
        for i in range(batch_size):
            code = context[i, :]
            for j in range(1, src_length[i].item()+1):
                decoded_chs.append(self.lang.index2word[code[j].item()])
            else:
                decoded_str.append(' '.join(decoded_chs))
                decoded_chs.clear()
        return decoded_str


    def evaluate(self, dataset):
        decoded_output, tar_text = [], []
        pbar = tqdm(dataset, total=len(dataset))
        # import pdb; pdb.set_trace()
        for batch in pbar:
            decoded_output += self.decode_batch(batch['src'])
            text = batch['tar_text']
            text = list(map(lambda s: ' '.join(list(s)), text))
            tar_text += text
        with open('hypothesis.txt', 'w', encoding='utf-8') as f:
            for out in decoded_output:
                f.write(''.join(out.split()))
                f.write('\n')
        with open('reference.txt', 'w', encoding='utf-8') as f:
            for out in tar_text:
                f.write(''.join(out.split()))
                f.write('\n')
        # import pdb; pdb.set_trace()
        total = 0
        format_correct = 0
        for out, tar in zip(decoded_output, tar_text):
            total += 1
            out_list, tar_list = out.split(), tar.split()
            if len(out_list) == len(tar_list):
                for out_ch, tar_ch in zip(out_list, tar_list):
                    if (tar_ch == '，' or out_ch == '，') and out_ch != tar_ch:
                        break
                else:
                    format_correct += 1
        format_acc = format_correct / total
        # import pdb; pdb.set_trace()
        tar_text = list(map(lambda s: [s], tar_text))
        bleu_score = corpus_bleu(decoded_output, tar_text)[0][0]*100
        print(f'BLEU score: {bleu_score:.2f}')
        print(f'Format acc: {format_acc:.2f}')
        return bleu_score, format_acc

    def generate_response(self, src: str, wait=10) -> str:
        import time
        t0 = time.time()
        length = len(src)
        src_text = src
        src = self.lang.words2indices(src)
        src = torch.tensor(src).unsqueeze(0).cuda()
        valid = False
        success = True
        decoded_str = []
        while not valid:
            if time.time() - t0 > wait:
                success = False
                break
            context = torch.tensor([[SOS_token]]).cuda()
            for i in range(length):
                context_mask = subsequent_mask(src.size(-1))[:, :, :i+1].type(torch.bool).cuda()
                # decoder_mask = subsequent_mask(length).type(torch.bool).cuda()
                decoder_output = self.model(context,
                                            src,
                                            context_mask,
                                            None)
                log_probs = self.model.generator(decoder_output[:, i, :])
                probs = torch.exp(log_probs)
                sampler = torch.distributions.categorical.Categorical(
                    probs=probs.squeeze()
                )
                index = sampler.sample().item()
                context = torch.cat([context, torch.tensor([[index]]).cuda()], dim=1)
            decoded_str = []
            # import pdb; pdb.set_trace()
            for j in range(1, context.size(1)):
                decoded_str.append(self.lang.index2word[context[0, j].item()])
                if (src_text[j-1] == '，' or decoded_str[-1] == '，') \
                        and src_text[j-1] != decoded_str[-1]:
                    break
            else:
                valid = True
        if not success: decoded_str = []
        return ''.join(decoded_str), success