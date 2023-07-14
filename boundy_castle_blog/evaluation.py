import sklearn
import torch
from tqdm.notebook import tqdm

class EpochEvaluator:
    def __init__(self):
        self.real = []
        self.predicted = []

    def add_data(self, r, p):
        self.real += list(r.detach().numpy())
        self.predicted += list(p.detach().numpy())
    
    def eval_data(self):
        assert len(self.real) == len(self.predicted)
        
        precision = sklearn.metrics.precision_score(self.real, self.predicted, average='macro', zero_division=0.0)
        recall = sklearn.metrics.recall_score(self.real, self.predicted, average='macro', zero_division=0.0)
        f1 = sklearn.metrics.f1_score(self.real, self.predicted, average='macro')
        accuracy = sklearn.metrics.accuracy_score(self.real, self.predicted)
        
        n = len(self.real)
        
        self.real = []
        self.predicted = []
        r = {'precision': precision, 'recall': recall, 'f1': f1, 'acc': accuracy, 'n': n}
        return r
    

def eval_model(model, dataloader, vocab, decode_function):
    real = []
    pred = []
    model.eval()
    for _, (txt, label) in tqdm(enumerate(dataloader)):
        outputs = model(txt)
        predicted = torch.max(outputs, 1)[1]
        
        # check prediction only when batch size is 1
        if label.numpy()[0] != predicted.numpy()[0]:
            txt = txt.numpy()[0]
            txt = [vocab.get_itos()[x] for x in txt]
            print('Wrong prediction')
            print(f'text: {" ".join(txt)}')
            real_class = decode_function(label.numpy()[0])
            predicted_class = decode_function(predicted.numpy()[0])
            print(f'real class: {real_class}')
            print(f'predicted class: {predicted_class}')
        pred += list(predicted.numpy())
        real += list(label.numpy())
        
    return real, pred