import datetime
import time

import torch

def fit(model, dataloaders, epochs, optimizer, criterion, evaluator, log_path='../train_out'):
    t = datetime.datetime.now()
    t = t.strftime("run_%m_%d_%H_%M_%S")
    
    # t1 = os.path.join(log_path, t + '_train')
    # writer_train = SummaryWriter(t1)
    # t2 = os.path.join(log_path, t + '_val')
    # writer_val = SummaryWriter(t2)
    for e in range(1, epochs+1):
        epoch_start = time.time()
        epoch_loss = 0
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            batch_counter = 0
            for idx, (txt, label) in enumerate(dataloaders[phase]):
                optimizer.zero_grad()
                outputs = model(txt)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                predicted = torch.max(outputs, 1)[1]
                evaluator.add_data(label, predicted)
                epoch_loss += loss
                batch_counter += 1

            epoch_duration = time.time() - epoch_start
            epoch_loss = epoch_loss / batch_counter
            metrics = evaluator.eval_data()
            print('{:5} Epoch {} loss: {:3f} time: {:4f}ms \
                   pre: {:.2f} rec: {:.2f} f1: {:.2f} acc: {:.2f} items: {}'
                  .format(phase, e, epoch_loss, epoch_duration, \
                          metrics['precision'], metrics['recall'], metrics['f1'], metrics['acc'], metrics['n']))
            
            # if phase == 'train':
            #     writer_train.add_scalar('Loss', epoch_loss, e)
            #     writer_train.add_scalar('Rec', metrics['recall'], e)
            #     writer_train.add_scalar('Pre', metrics['precision'], e)
            #     writer_train.add_scalar('F1', metrics['f1'], e)
            #     writer_train.add_scalar('Acc', metrics['acc'], e)
            # else:
            #     writer_val.add_scalar('Loss', epoch_loss, e)
            #     writer_val.add_scalar('Rec', metrics['recall'], e)
            #     writer_val.add_scalar('Pre', metrics['precision'], e)
            #     writer_val.add_scalar('F1', metrics['f1'], e)
            #     writer_val.add_scalar('Acc', metrics['acc'], e)
            #writer.add_scalar('AvgLoss/'+ phase, epoch_loss, e)
    return model