import torch
from torch.utils.data import Dataset

class BounceTextDataset(Dataset):
    def __init__(self, X, Y, vocab, text_len=50, tokenizer=lambda x: x.split(' '), empty_token_index=1):
        self.x = X
        self.y = Y
        self.vocab = vocab
        self.text_len = text_len
        self.tokenizer = tokenizer
        self.empty_token_index = empty_token_index
        
    def encode_text(self, text):
        encoded = [self.vocab[x] for x in self.tokenizer(text)]
        return encoded 
    
    def padd_txt(self, txt):
        l = len(txt)
        if l > self.text_len:
            return txt[:self.text_len]
        elif l < self.text_len:
            return txt + [self.empty_token_index for i in range(self.text_len - l)]
        else:
            return txt
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        txt = self.x[idx]
        txt = self.encode_text(self.x[idx])
        txt = self.padd_txt(txt)
        txt = torch.tensor(txt, dtype=torch.int64)
        label = self.y[idx]
        label = torch.tensor(label, dtype=torch.int64)
        
        return txt, label