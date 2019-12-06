import torch
from torch.utils.data import Dataset
import pandas as pd
import random
import copy
import re
import html
import unicodedata


class Datasets(Dataset):
    def __init__(self, file_path, label_list=None, pretrained_type='skt', objective='classification', max_len=64):
        self.objective = objective
        self.max_len = max_len
        if label_list is not None:
            # multi_label = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']
            # binary_label = ['긍정', '부정']
            self.label2idx = dict(zip(label_list, range(len(label_list))))
        self.corpus, self.label = self.get_data(file_path)
        self.pretrained_type = pretrained_type
        self.tokenizer, self.vocab = get_pretrained_model(pretrained_type)

        self.pad_idx = self.vocab['[PAD]']
        self.cls_idx = self.vocab['[CLS]']
        self.sep_idx = self.vocab['[SEP]']
        self.mask_idx = self.vocab['[MASK]']

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        # Normalize
        tokens = self.normalize_string(self.corpus[idx])

        # Tokenize by Wordpiece tokenizer
        tokens = self.tokenize(tokens)

        # Change wordpiece to indices
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        label = self.label[idx] if self.label is not None else None

        return tokens, label

    def batch_sequence(self, batch):
        tokens, label = list(zip(*batch))

        # Get max length from batch
        max_len = min(self.max_len, max([len(i) for i in tokens]))
        vocab_label = None

        # Use unmasked tokens as vocab labels for MLM
        if self.objective != 'classification':
            vocab_label = torch.tensor(
                [self.pad([-1] + t + [-1], max_len, for_lm=True) for t in tokens])
            tokens = [self.make_masked_input(t) for t in tokens]

        tokens = torch.tensor([self.pad(
            [self.cls_idx] + t + [self.sep_idx], max_len, for_lm=False) for t in tokens])
        masks = torch.ones_like(tokens).masked_fill(
            tokens == self.pad_idx, self.pad_idx)

        return tokens, masks, label, vocab_label

    def normalize_string(self, s):
        s = html.unescape(s)
        s = re.sub(r"[\s]", r" ", s)
        s = re.sub(r"[^a-zA-Z가-힣ㄱ-ㅎ0-9.!?]+", r" ", s)
        return s

    def get_data(self, file_path):
        data = pd.read_csv(file_path)
        corpus = data['Sentence']
        label = None
        try:
            label = [self.label2idx[l] for l in data['Emotion']]
        except:
            pass
        return corpus, label

    def tokenize(self, tokens):
        if self.pretrained_type == 'etri':
            return self.tokenizer.tokenize(tokens)
        elif self.pretrained_type == 'skt':
            return self.tokenizer(tokens)

    def pad(self, sample, max_len, for_lm):
        p = -1 if for_lm else self.pad_idx  # use -1 as pad idx in language model
        diff = max_len - len(sample)
        if diff > 0:
            sample += [p] * diff
        else:
            sample = sample[-max_len:]
        return sample

    def make_masked_input(self, tokens):
        for i, token in enumerate(tokens):
            prob = random.random()
            # mask token with 15%
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.mask_idx

                # 10% randomly change token to random token
                elif prob < 0.9:
                    special_tokens = [self.pad_idx,
                                      self.mask_idx, self.sep_idx, self.cls_idx]
                    r = random.choice(range(len(self.vocab)))
                    while r in special_tokens:
                        r = random.choice(range(len(self.vocab)))
                    tokens[i] = r

                # -> rest 10% randomly keep current token
        return tokens


def get_pretrained_model(pretrained_type):
    if pretrained_type == 'etri':
        # use etri tokenizer
        from pretrained_model.etri.tokenization import BertTokenizer
        tokenizer_path = './pretrained_model/etri/vocab.korean.rawtext.list'
        tokenizer = BertTokenizer.from_pretrained(
            tokenizer_path, do_lower_case=False)
        vocab = tokenizer.vocab
    elif pretrained_type == 'skt':
        # use gluonnlp tokenizer
        import gluonnlp as nlp
        vocab_path = './pretrained_model/skt/vocab.json'
        tokenizer_path = './pretrained_model/skt/tokenizer.model'
        vocab = nlp.vocab.BERTVocab.from_json(open(vocab_path, 'rt').read())
        tokenizer = nlp.data.BERTSPTokenizer(
            path=tokenizer_path, vocab=vocab, lower=False)
        vocab = tokenizer.vocab.token_to_idx
    else:
        TypeError('Invalid pretrained model type')
    return tokenizer, vocab
