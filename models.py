import torch.nn as nn

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, input_len=40):
        super(EmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim*input_len, 128)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        
    def forward(self, txt):
        embedded = self.embedding(txt)
        embedded = embedded.view((txt.shape[0],-1))
        fc = self.fc(embedded)
        relu1 = self.relu1(fc)
        dropout = self.dropout(relu1)
        fc2 = self.fc2(dropout)
        return fc2