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
from models.NIC4Product import EncoderCNN, DecoderLSTM
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == '__main__':
    train_batch_size = 16
    val_batch_size = 16
    val_batch_size2 = 1
    max_length = 10

    best_embedding_size = 100
    best_hidden_size = 300
    best_batch_norm_momentum = 0.001
    best_learning_rate = 0.1
    best_weight_decay = 1e-08
    best_momentum = 0

    embedding_sizes = [100,200,300]
    hidden_sizes = [100,200,300]
    batch_norm_momentums = [0.1,0.01,0.001]
    learning_rates = [0.1,0.01,0.001]
    weight_decays = [0,0.0001,0.00000001]
    momentums = [0,0.001,0.01]
    one_factor_at_a_time = [embedding_sizes, hidden_sizes, batch_norm_momentums, learning_rates, weight_decays, momentums]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('dataset/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    transforms = transforms.Compose([transforms.Resize(224),transforms.RandomCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = zd.ZaloraDataset('dataset/train.csv', 'dataset/images', vocab, transforms)
    val_dataset = zd.ZaloraDataset('dataset/val.csv', 'dataset/images', vocab, transforms)
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2, collate_fn=zd.collate_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=2, collate_fn=zd.collate_fn)
    val_loader2 = DataLoader(dataset=val_dataset, batch_size=val_batch_size2, shuffle=False, num_workers=4, collate_fn=zd.collate_fn)

    def train(experiment_number, embedding_size, hidden_size, batch_norm_momentum, learning_rate, weight_decay, momentum):
        if not os.path.exists('logs/{}/'.format(experiment_number)):
            os.mkdir('logs/{}/'.format(experiment_number))

        encoder = EncoderCNN(embedding_size, batch_norm_momentum).to(device)
        decoder = DecoderLSTM(embedding_size, hidden_size, len(vocab), max_length).to(device)

        criterion = torch.nn.NLLLoss()
        nesterov = False
        if momentum > 0:
            nesterov = True
        encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
        decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)

        min_val_loss = float('inf')
        epoch = 1
        while epoch <= 50:
            epoch_start = time.time()

            train_loss_epoch = 0
            train_loss_count = 0
            encoder.train()
            decoder.train()

            for (_, images, titles, lengths) in train_loader:
                images = images.to(device)
                titles = titles.to(device)
                targets = pack_padded_sequence(titles, lengths, batch_first=True)[0]

                visual_features = encoder(images)
                outputs = decoder(visual_features, titles, lengths)

                loss = criterion(outputs, targets)
                decoder.zero_grad()
                encoder.zero_grad()
                
                loss.backward()
                decoder_optimizer.step()
                encoder_optimizer.step()

                train_loss_epoch += loss.item() * targets.size(0)
                train_loss_count += sum(lengths)

            epoch_end = time.time()
            
            val_loss_epoch = 0
            val_loss_count = 0
            with torch.no_grad():
                encoder.eval()
                decoder.eval()

                for (_, images, titles, lengths) in val_loader:
                    images = images.to(device)
                    titles = titles.to(device)
                    targets = pack_padded_sequence(titles, lengths, batch_first=True)[0]

                    visual_features = encoder(images)
                    outputs = decoder(visual_features, titles, lengths)

                    loss = criterion(outputs, targets)

                    val_loss_epoch += loss.item() * targets.size(0)
                    val_loss_count += sum(lengths)

            train_loss = train_loss_epoch/train_loss_count
            val_loss = val_loss_epoch/val_loss_count
            training_time = epoch_end-epoch_start
            with open('logs/NIC4Product/{}/train_log.txt'.format(experiment_number), 'a+') as f:
                f.write('Epoch: {}, Train Loss: {}, Validation Loss: {}, Training Time: {}\n'.format(epoch, train_loss, val_loss, training_time))
            print("Epoch: {}, Train Loss: {}, Validation Loss: {}, Training Time: {}".format(epoch, train_loss, val_loss, training_time))
            epoch += 1

            if val_loss <= min_val_loss:
                torch.save(decoder.state_dict(), 'checkpoints/NIC4Product/temp/decoder.ckpt')
                torch.save(encoder.state_dict(), 'checkpoints/NIC4Product/temp/encoder.ckpt')
                min_val_loss = val_loss
        
        with open('logs/NIC4Product/{}/hyperparams.txt'.format(experiment_number), 'a+') as f:
            f.write('embedding_size: {}\n'.format(embedding_size))
            f.write('hidden_size: {}\n'.format(hidden_size))
            f.write('batch_norm_momentum: {}\n'.format(batch_norm_momentum))
            f.write('learning_rate: {}\n'.format(learning_rate))
            f.write('weight_decay: {}\n'.format(weight_decay))
            f.write('momentum: {}\n'.format(momentum))

    def validate(experiment_number, best_experiment_number, best_bleu, best_cider, best_infer_time, embedding_size, hidden_size, batch_norm_momentum):
        encoder = EncoderCNN(embedding_size, batch_norm_momentum).eval().to(device)
        decoder = DecoderLSTM(embedding_size, hidden_size, len(vocab), max_length).eval().to(device)

        decoder.load_state_dict(torch.load('checkpoints/NIC4Product/temp/decoder.ckpt'))
        encoder.load_state_dict(torch.load('checkpoints/NIC4Product/temp/encoder.ckpt'))

        df = pd.read_csv('dataset/val.csv')

        results = []
        with torch.no_grad():
            for _, (ids, images, titles, _) in enumerate(val_loader2):
                start = time.time()

                images = images.to(device)
                visual_features = encoder(images)

                word_ids = decoder.predict(visual_features)
                word_ids = word_ids[0].cpu().numpy()

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

        with open('logs/NIC4Product/{}/val_log.txt'.format(experiment_number), 'a+') as f:
            f.write('BLEU Cumulative: {}\n'.format(current_bleu))
            f.write('CIDER: {}\n'.format(current_cider))
            f.write('Inference Time: {}\n'.format(current_infer_time))

        if (0.1*current_bleu + 0.9*current_cider/10) > (0.1*best_bleu + 0.9*best_cider/10):
            torch.save(decoder.state_dict(), 'checkpoints/NIC4Product/best/decoder.ckpt')
            torch.save(encoder.state_dict(), 'checkpoints/NIC4Product/best/encoder.ckpt')
            return experiment_number, current_bleu, current_cider, current_infer_time
        else:
            return best_experiment_number, best_bleu, best_cider, best_infer_time

    best_experiment_number = 10
    best_bleu = 0
    best_cider = 0
    best_infer_time = float('inf')

    experiment_number = 11
    print("Experiments Start")
    for i,factors in enumerate(one_factor_at_a_time):
        for j,factor in enumerate(factors):
            if i==0:
                print("Experiment Number {} Starts".format(experiment_number))
                print('Training Experiment Number {} Starts'.format(experiment_number))
                train(experiment_number, factor, best_hidden_size, best_batch_norm_momentum, best_learning_rate, best_weight_decay, best_momentum)
                print("Validating Experiment Number {} Starts".format(experiment_number))
                best_experiment_number, best_bleu, best_cider, best_infer_time = validate(experiment_number, best_experiment_number, best_bleu, best_cider, best_infer_time, factor, best_hidden_size, best_batch_norm_momentum)
                if best_experiment_number == experiment_number:
                    best_embedding_size = factor
                print('Best Experiment Number: {}'.format(best_experiment_number))
                experiment_number += 1
            elif i==1:
                if j>0:
                    print("Experiment Number {} Starts".format(experiment_number))
                    print('Training Experiment Number {} Starts'.format(experiment_number))
                    train(experiment_number, best_embedding_size, factor, best_batch_norm_momentum, best_learning_rate, best_weight_decay, best_momentum)
                    print("Validating Experiment Number {} Starts".format(experiment_number))
                    best_experiment_number, best_bleu, best_cider, best_infer_time = validate(experiment_number, best_experiment_number, best_bleu, best_cider, best_infer_time, best_embedding_size, factor, best_batch_norm_momentum)
                    if best_experiment_number == experiment_number:
                        best_hidden_size = factor
                    print('Best Experiment Number: {}'.format(best_experiment_number))
                    experiment_number += 1
            elif i==2:
                if j>0:
                    print("Experiment Number {} Starts".format(experiment_number))
                    print('Training Experiment Number {} Starts'.format(experiment_number))
                    train(experiment_number, best_embedding_size, best_hidden_size, factor, best_learning_rate, best_weight_decay, best_momentum)
                    print("Validating Experiment Number {} Starts".format(experiment_number))
                    best_experiment_number, best_bleu, best_cider, best_infer_time = validate(experiment_number, best_experiment_number, best_bleu, best_cider, best_infer_time, best_embedding_size, best_hidden_size, factor)
                    if best_experiment_number == experiment_number:
                        best_batch_norm_momentum = factor
                    print('Best Experiment Number: {}'.format(best_experiment_number))
                    experiment_number += 1
            elif i==3:
                if j>0:
                    print("Experiment Number {} Starts".format(experiment_number))
                    print('Training Experiment Number {} Starts'.format(experiment_number))
                    train(experiment_number, best_embedding_size, best_hidden_size, best_batch_norm_momentum, factor, best_weight_decay, best_momentum)
                    print("Validating Experiment Number {} Starts".format(experiment_number))
                    best_experiment_number, best_bleu, best_cider, best_infer_time = validate(experiment_number, best_experiment_number, best_bleu, best_cider, best_infer_time, best_embedding_size, best_hidden_size, best_batch_norm_momentum)
                    if best_experiment_number == experiment_number:
                        best_learning_rate = factor
                    print('Best Experiment Number: {}'.format(best_experiment_number))
                    experiment_number += 1
            elif i==4:
                if j>0:
                    print("Experiment Number {} Starts".format(experiment_number))
                    print('Training Experiment Number {} Starts'.format(experiment_number))
                    train(experiment_number, best_embedding_size, best_hidden_size, best_batch_norm_momentum, best_learning_rate, factor, best_momentum)
                    print("Validating Experiment Number {} Starts".format(experiment_number))
                    best_experiment_number, best_bleu, best_cider, best_infer_time = validate(experiment_number, best_experiment_number, best_bleu, best_cider, best_infer_time, best_embedding_size, best_hidden_size, best_batch_norm_momentum)
                    if best_experiment_number == experiment_number:
                        best_weight_decay = factor
                    print('Best Experiment Number: {}'.format(best_experiment_number))
                    experiment_number += 1
            elif i==5:
                if j>0:
                    print("Experiment Number {} Starts".format(experiment_number))
                    print('Training Experiment Number {} Starts'.format(experiment_number))
                    train(experiment_number, best_embedding_size, best_hidden_size, best_batch_norm_momentum, best_learning_rate, best_weight_decay, factor)
                    print("Validating Experiment Number {} Starts".format(experiment_number))
                    best_experiment_number, best_bleu, best_cider, best_infer_time = validate(experiment_number, best_experiment_number, best_bleu, best_cider, best_infer_time, best_embedding_size, best_hidden_size, best_batch_norm_momentum)
                    if best_experiment_number == experiment_number:
                        best_momentum = factor
                    print('Best Experiment Number: {}'.format(best_experiment_number))
                    experiment_number += 1

    shutil.copy('logs/NIC4Product/{}/train_log.txt'.format(best_experiment_number), 'logs/NIC4Product/best/')
    shutil.copy('logs/NIC4Product/{}/val_log.txt'.format(best_experiment_number), 'logs/NIC4Product/best/')
    shutil.copy('logs/NIC4Product/{}/hyperparams.txt'.format(best_experiment_number), 'logs/NIC4Product/best/')