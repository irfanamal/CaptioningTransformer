import dataset.ZaloraDataset as zd
import nltk
import numpy
import os
import pandas as pd
import pickle
import shutil
import time
import torch
from cider.cider import Cider
from dataset.Vocabulary import Vocabulary
from models.CaT import CaT
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == '__main__':
    train_batch_size = 16
    val_batch_size = 16
    val_batch_size2 = 1
    seq_length = 10

    best_nhead = 100
    best_forward_size = 300
    best_num_layer = 0.001
    best_learning_rate = 0.1
    best_weight_decay = 1e-08
    best_momentum = 0

    nheads = [12,10,8]
    forward_sizes = [3000,2000,1000]
    num_layers = [12,10,8]
    learning_rates = [0.1,0.01,0.001]
    weight_decays = [0,0.0001,0.00000001]
    momentums = [0,0.001,0.01]
    one_factor_at_a_time = [nheads, forward_sizes, num_layers, learning_rates, weight_decays, momentums]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('dataset/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    transforms = transforms.Compose([transforms.Resize(224),transforms.RandomCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = zd.ZaloraDataset('dataset/train.csv', 'dataset/images', vocab, transforms)
    val_dataset = zd.ZaloraDataset('dataset/val.csv', 'dataset/images', vocab, transforms)
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2, collate_fn=zd.collate_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=2, collate_fn=zd.collate_fn)
    val_loader2 = DataLoader(dataset=val_dataset, batch_size=val_batch_size2, shuffle=False, num_workers=4, collate_fn=zd.collate_fn)

    def train(experiment_number, nhead, forward_size, num_layer, learning_rate, weight_decay, momentum):
        if not os.path.exists('logs/CaT/{}/'.format(experiment_number)):
            os.mkdir('logs/CaT/{}/'.format(experiment_number))

        cat = CaT(len(vocab), nhead, forward_size, num_layer, 20).to(device)

        criterion = torch.nn.NLLLoss()
        nesterov = False
        if momentum > 0:
            nesterov = True
        optimizer = torch.optim.SGD(cat.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)

        min_val_loss = float('inf')
        epoch = 1
        while epoch <= 50:
            epoch_start = time.time()

            train_loss_epoch = 0
            train_loss_count = 0
            cat.train()

            for (_, images, titles, lengths) in train_loader:
                images = images.to(device)

                for i in range(len(lengths)):
                    lengths[i] -= 1

                tgt = titles.clone()
                tgt = tgt[:,:-1]
                tgt[tgt==2] = 0
                tgt = tgt.to(device)

                targets = titles.clone()
                targets = targets[:,1:]
                targets = targets.to(device)
                targets = pack_padded_sequence(targets, lengths, batch_first=True)[0]

                tgt_masks = (torch.triu(torch.ones(max(lengths), max(lengths))) == 1).transpose(0,1)
                tgt_masks = tgt_masks.float().masked_fill(tgt_masks == 0, float('-inf')).masked_fill(tgt_masks == 1, float(0.0)).to(device)
                tgt_key_padding_masks = [[True for _ in range(max(lengths))] for _ in range(len(lengths))]
                for i, length in enumerate(lengths):
                    tgt_key_padding_masks[i][:length] = [False for _ in range(length)]
                tgt_key_padding_masks = torch.tensor(tgt_key_padding_masks).to(device)
                outputs = cat(images, tgt, tgt_masks, tgt_key_padding_masks)
                outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]

                loss = criterion(outputs, targets)
                cat.zero_grad()
                
                loss.backward()
                cat.step()

                train_loss_epoch += loss.item() * targets.size(0)
                train_loss_count += sum(lengths)

            epoch_end = time.time()
            
            val_loss_epoch = 0
            val_loss_count = 0
            with torch.no_grad():
                cat.eval()

                for (_, images, titles, lengths) in val_loader:
                    images = images.to(device)

                    for i in range(len(lengths)):
                        lengths[i] -= 1

                    tgt = titles.clone()
                    tgt = tgt[:,:-1]
                    tgt[tgt==2] = 0
                    tgt = tgt.to(device)

                    targets = titles.clone()
                    targets = targets[:,1:]
                    targets = targets.to(device)
                    targets = pack_padded_sequence(targets, lengths, batch_first=True)[0]

                    tgt_masks = (torch.triu(torch.ones(max(lengths), max(lengths))) == 1).transpose(0,1)
                    tgt_masks = tgt_masks.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)
                    tgt_key_padding_masks = [[True for _ in range(max(lengths))] for _ in range(len(lengths))]
                    for i, length in enumerate(lengths):
                        tgt_key_padding_masks[i][:length] = [False for _ in range(length)]
                    tgt_key_padding_masks = torch.tensor(tgt_key_padding_masks).to(device)
                    outputs = cat(images, tgt, tgt_masks, tgt_key_padding_masks)
                    outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]

                    loss = criterion(outputs, targets)

                    val_loss_epoch += loss.item() * targets.size(0)
                    val_loss_count += sum(lengths)

            train_loss = train_loss_epoch/train_loss_count
            val_loss = val_loss_epoch/val_loss_count
            training_time = epoch_end-epoch_start
            with open('logs/CaT/{}/train_log.txt'.format(experiment_number), 'a+') as f:
                f.write('Epoch: {}, Train Loss: {}, Validation Loss: {}, Training Time: {}\n'.format(epoch, train_loss, val_loss, training_time))
            print("Epoch: {}, Train Loss: {}, Validation Loss: {}, Training Time: {}".format(epoch, train_loss, val_loss, training_time))
            epoch += 1

            if val_loss <= min_val_loss:
                torch.save(cat.state_dict(), 'checkpoint/CaT/temp/cat.ckpt')
                min_val_loss = val_loss
        
        with open('logs/CaT/{}/hyperparams.txt'.format(experiment_number), 'a+') as f:
            f.write('nhead: {}\n'.format(nhead))
            f.write('forward_size: {}\n'.format(forward_size))
            f.write('num_layer: {}\n'.format(num_layer))
            f.write('learning_rate: {}\n'.format(learning_rate))
            f.write('weight_decay: {}\n'.format(weight_decay))
            f.write('momentum: {}\n'.format(momentum))

    def validate(experiment_number, best_experiment_number, best_bleu, best_cider, nhead, forward_size, num_layer):
        cat = CaT(len(vocab), nhead, forward_size, num_layer, 20).to(device)

        cat.load_state_dict(torch.load('checkpoint/CaT/temp/cat.ckpt'))

        df = pd.read_csv('dataset/val.csv')

        results = []
        with torch.no_grad():
            for ids, images, _ in val_loader2:
                start = time.time()

                images = images.to(device)
                word_ids = [1]
                memory = None

                for i in range(10):
                    tgt = torch.tensor(word_ids).unsqueeze(0).to(device)
                    tgt_masks = (torch.triu(torch.ones(i+1, i+1)) == 1).transpose(0,1)
                    tgt_masks = tgt_masks.float().masked_fill(tgt_masks == 0, float('-inf')).masked_fill(tgt_masks == 1, float(0.0)).to(device)
                    memory, output = cat.decode(images, tgt, tgt_masks, memory)
                    word_ids.append(torch.argmax(output[-1]).item())
                    if word_ids[-1] == 2:
                        break

                titles = df.loc[df['id'].isin(ids)]['produk'].values
                ground_truth = []
                ground_truth.append(['<S>'] + titles[0].lower().split() + ['</S>'])
                
                generated = []
                for id in word_ids:
                    generated.append(vocab.getWord(id))
                    if id == 2:
                        break

                end = time.time()
                test_time = end-start
                result = {'id':ids[0], 'generated':generated, 'ground_truth':ground_truth, 'test_time':test_time}
                results.append(result)
        
        bleu = []
        gts = {}
        res = {}
        infer_time = []
        for result in results:
            bleu.append(nltk.translate.bleu_score.sentence_bleu(result['ground_truth'], result['generated'], smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method7))
            gts[result['id']] = [' '.join(result['ground_truth'][0])]
            res[result['id']] = [' '.join(result['generated'])]
            infer_time.append(result['test_time'])
        
        current_bleu = numpy.mean(bleu)
        current_cider = Cider().compute_score(gts,res)[0]
        current_infer_time = numpy.mean(infer_time)

        print('BLEU Cumulative: {}'.format(current_bleu))
        print('CIDER: {}'.format(current_cider))
        print('Inference Time: {}'.format(current_infer_time))

        with open('logs/CaT/{}/val_log.txt'.format(experiment_number), 'a+') as f:
            f.write('BLEU Cumulative: {}\n'.format(current_bleu))
            f.write('CIDER: {}\n'.format(current_cider))
            f.write('Inference Time: {}\n'.format(current_infer_time))

        if current_cider > best_cider:
            torch.save(cat.state_dict(), 'checkpoint/CaT/best/cat.ckpt')
            return experiment_number, current_bleu, current_cider, current_infer_time
        else:
            return best_experiment_number, best_bleu, best_cider, best_infer_time

    best_experiment_number = 1
    best_bleu = 0
    best_cider = 0
    best_infer_time = float('inf')

    experiment_number = 1
    print("Experiments Start")
    for i,factors in enumerate(one_factor_at_a_time):
        for j,factor in enumerate(factors):
            if i==0:
                print("Experiment Number {} Starts".format(experiment_number))
                print('Training Experiment Number {} Starts'.format(experiment_number))
                train(experiment_number, factor, best_forward_size, best_num_layer, best_learning_rate, best_weight_decay, best_momentum)
                print("Validating Experiment Number {} Starts".format(experiment_number))
                best_experiment_number, best_bleu, best_cider, best_infer_time = validate(experiment_number, best_experiment_number, best_bleu, best_cider, best_infer_time, factor, best_forward_size, best_num_layer)
                if best_experiment_number == experiment_number:
                    best_nhead = factor
                print('Best Experiment Number: {}'.format(best_experiment_number))
                experiment_number += 1
            elif i==1:
                if j>0:
                    print("Experiment Number {} Starts".format(experiment_number))
                    print('Training Experiment Number {} Starts'.format(experiment_number))
                    train(experiment_number, best_nhead, factor, best_num_layer, best_learning_rate, best_weight_decay, best_momentum)
                    print("Validating Experiment Number {} Starts".format(experiment_number))
                    best_experiment_number, best_bleu, best_cider, best_infer_time = validate(experiment_number, best_experiment_number, best_bleu, best_cider, best_infer_time, best_nhead, factor, best_num_layer)
                    if best_experiment_number == experiment_number:
                        best_forward_size = factor
                    print('Best Experiment Number: {}'.format(best_experiment_number))
                    experiment_number += 1
            elif i==2:
                if j>0:
                    print("Experiment Number {} Starts".format(experiment_number))
                    print('Training Experiment Number {} Starts'.format(experiment_number))
                    train(experiment_number, best_nhead, best_forward_size, factor, best_learning_rate, best_weight_decay, best_momentum)
                    print("Validating Experiment Number {} Starts".format(experiment_number))
                    best_experiment_number, best_bleu, best_cider, best_infer_time = validate(experiment_number, best_experiment_number, best_bleu, best_cider, best_infer_time, best_nhead, best_forward_size, factor)
                    if best_experiment_number == experiment_number:
                        best_num_layer = factor
                    print('Best Experiment Number: {}'.format(best_experiment_number))
                    experiment_number += 1
            elif i==3:
                if j>0:
                    print("Experiment Number {} Starts".format(experiment_number))
                    print('Training Experiment Number {} Starts'.format(experiment_number))
                    train(experiment_number, best_nhead, best_forward_size, best_num_layer, factor, best_weight_decay, best_momentum)
                    print("Validating Experiment Number {} Starts".format(experiment_number))
                    best_experiment_number, best_bleu, best_cider, best_infer_time = validate(experiment_number, best_experiment_number, best_bleu, best_cider, best_infer_time, best_nhead, best_forward_size, best_num_layer)
                    if best_experiment_number == experiment_number:
                        best_learning_rate = factor
                    print('Best Experiment Number: {}'.format(best_experiment_number))
                    experiment_number += 1
            elif i==4:
                if j>0:
                    print("Experiment Number {} Starts".format(experiment_number))
                    print('Training Experiment Number {} Starts'.format(experiment_number))
                    train(experiment_number, best_nhead, best_forward_size, best_num_layer, best_learning_rate, factor, best_momentum)
                    print("Validating Experiment Number {} Starts".format(experiment_number))
                    best_experiment_number, best_bleu, best_cider, best_infer_time = validate(experiment_number, best_experiment_number, best_bleu, best_cider, best_infer_time, best_nhead, best_forward_size, best_num_layer)
                    if best_experiment_number == experiment_number:
                        best_weight_decay = factor
                    print('Best Experiment Number: {}'.format(best_experiment_number))
                    experiment_number += 1
            elif i==5:
                if j>0:
                    print("Experiment Number {} Starts".format(experiment_number))
                    print('Training Experiment Number {} Starts'.format(experiment_number))
                    train(experiment_number, best_nhead, best_forward_size, best_num_layer, best_learning_rate, best_weight_decay, factor)
                    print("Validating Experiment Number {} Starts".format(experiment_number))
                    best_experiment_number, best_bleu, best_cider, best_infer_time = validate(experiment_number, best_experiment_number, best_bleu, best_cider, best_infer_time, best_nhead, best_forward_size, best_num_layer)
                    if best_experiment_number == experiment_number:
                        best_momentum = factor
                    print('Best Experiment Number: {}'.format(best_experiment_number))
                    experiment_number += 1

    shutil.copy('logs/CaT/{}/train_log.txt'.format(best_experiment_number), 'logs/CaT/best/')
    shutil.copy('logs/CaT/{}/val_log.txt'.format(best_experiment_number), 'logs/CaT/best/')
    shutil.copy('logs/CaT/{}/hyperparams.txt'.format(best_experiment_number), 'logs/CaT/best/')