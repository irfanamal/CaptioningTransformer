import dataset.ZaloraSemanticDataset as zd
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
from dataset.Semantic import Semantic
from models.NIC4ProductSem import EncoderCNN, DecoderLSTM
from models.DaE import DaE
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == '__main__':
    train_batch_size = 16
    val_batch_size = 16
    val_batch_size2 = 1
    max_length = 10

    embedding_size = 100
    hidden_size = 300
    batch_norm_momentum = 0.001
    learning_rate = 0.1
    weight_decay = 1e-08
    momentum = 0

    best_dim = 128

    dimensions = [128,256,512]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('dataset/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('dataset/semantic7.pkl', 'rb') as f:
        semantic = pickle.load(f)

    transforms = transforms.Compose([transforms.Resize(224),transforms.RandomCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = zd.ZaloraSemanticDataset('dataset/train.csv', 'dataset/images', vocab, semantic, transforms)
    val_dataset = zd.ZaloraSemanticDataset('dataset/val.csv', 'dataset/images', vocab, semantic, transforms)
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2, collate_fn=zd.collate_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=2, collate_fn=zd.collate_fn)
    val_loader2 = DataLoader(dataset=val_dataset, batch_size=val_batch_size2, shuffle=False, num_workers=4, collate_fn=zd.collate_fn)

    def train(dimension, encoder_state=None, decoder_state=None, encoder_local_optima=None, decoder_local_optima=None, temp_val_loss=float('inf'), start_epoch=0):
        if not os.path.exists('logs/NIC4Product/semantics/th7/{}/'.format(dimension)):
            os.mkdir('logs/NIC4Product/semantics/th7/{}/'.format(dimension))

        encoder = EncoderCNN(embedding_size, batch_norm_momentum).to(device)
        decoder = DecoderLSTM(embedding_size, hidden_size, len(vocab), max_length, len(semantic), dimension).to(device)

        criterion = torch.nn.NLLLoss()
        encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, nesterov=False)
        decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, nesterov=False)

        min_val_loss = temp_val_loss
        encoder_weights = encoder.state_dict()
        decoder_weights = decoder.state_dict()
        if encoder_local_optima != None:
            encoder_weigths = torch.load(encoder_local_optima)
            decoder_weights = torch.load(decoder_local_optima)

        epoch = start_epoch
        if encoder_state != None:
            encoder.load_state_dict(torch.load(encoder_state))
            decoder.load_state_dict(torch.load(decoder_state))

        while epoch <= 50:
            epoch_start = time.time()

            train_loss_epoch = 0
            train_loss_count = 0
            encoder.train()
            decoder.train()

            for _, images, titles, lengths, semantics in train_loader:
                images = images.to(device)
                titles = titles.to(device)
                sematics = semantics.to(device)
                targets = pack_padded_sequence(titles, lengths, batch_first=True)[0]

                visual_features = encoder(images)
                outputs = decoder(visual_features, titles, lengths, semantics)

                loss = criterion(outputs, targets)
                decoder.zero_grad()
                encoder.zero_grad()
                
                loss.backward()
                decoder_optimizer.step()
                encoder_optimizer.step()

                train_loss_epoch += loss.item() * targets.size(0)
                train_loss_count += sum(lengths)

            epoch_end = time.time()
            torch.save(encoder.state_dict(), 'checkpoint/NIC4Product/semantics/th7/state/encoder.ckpt')
            torch.save(decoder.state_dict(), 'checkpoint/NIC4Product/semantics/th7/state/decoder.ckpt')
            
            val_loss_epoch = 0
            val_loss_count = 0
            with torch.no_grad():
                encoder.eval()
                decoder.eval()

                for _, images, titles, lengths, semantics in val_loader:
                    images = images.to(device)
                    titles = titles.to(device)
                    semantics = semantics.to(device)
                    targets = pack_padded_sequence(titles, lengths, batch_first=True)[0]

                    visual_features = encoder(images)
                    outputs = decoder(visual_features, titles, lengths, semantics)

                    loss = criterion(outputs, targets)

                    val_loss_epoch += loss.item() * targets.size(0)
                    val_loss_count += sum(lengths)

            train_loss = train_loss_epoch/train_loss_count
            val_loss = val_loss_epoch/val_loss_count
            training_time = epoch_end-epoch_start
            with open('logs/NIC4Product/semantics/th7/{}/train_log.txt'.format(dimension), 'a+') as f:
                f.write('Epoch: {}, Train Loss: {}, Validation Loss: {}, Training Time: {}\n'.format(epoch, train_loss, val_loss, training_time))
            print("Epoch: {}, Train Loss: {}, Validation Loss: {}, Training Time: {}".format(epoch, train_loss, val_loss, training_time))
            epoch += 1

            if val_loss <= min_val_loss:
                torch.save(decoder.state_dict(), 'checkpoint/NIC4Product/semantics/th7/temp/decoder.ckpt')
                torch.save(encoder.state_dict(), 'checkpoint/NIC4Product/semantics/th7/temp/encoder.ckpt')
                min_val_loss = val_loss

                with open('checkpoint/NIC4Product/semantics/th7/temp/logs.txt'.format(vision), 'w+') as f:
                    f.write('Val Loss:{}\n'.format(min_val_loss))
                    f.write('Infer Time:{}\n'.format(infer_time))
        
        with open('logs/NIC4Product/semantics/th7/{}/hyperparams.txt'.format(dimension), 'a+') as f:
            f.write('embedding_size: {}\n'.format(embedding_size))
            f.write('hidden_size: {}\n'.format(hidden_size))
            f.write('batch_norm_momentum: {}\n'.format(batch_norm_momentum))
            f.write('learning_rate: {}\n'.format(learning_rate))
            f.write('weight_decay: {}\n'.format(weight_decay))
            f.write('momentum: {}\n'.format(momentum))
            f.write('proj_dim: {}\n'.format(dimension))

    def validate(dimension):
        encoder = EncoderCNN(embedding_size, batch_norm_momentum).eval().to(device)
        decoder = DecoderLSTM(embedding_size, hidden_size, len(vocab), max_length, len(semantics), dimension).eval().to(device)
        dae = DaE('vit', 2, 2048, len(semantics)).to_device

        decoder.load_state_dict(torch.load('checkpoint/NIC4Product/semantics/th7/temp/decoder.ckpt'))
        encoder.load_state_dict(torch.load('checkpoint/NIC4Product/semantics/th7/temp/encoder.ckpt'))
        dae.load_state_dict(torch.load('checkpoint/DaE/vit/th7/dae.ckpt'))

        df = pd.read_csv('dataset/val.csv')

        results = []
        with torch.no_grad():
            for ids, images, titles, _, semantics in val_loader2:
                start = time.time()

                images = images.to(device)
                semantics = dae(images)
                visual_features = encoder(images)

                word_ids = decoder.predict(visual_features, semantics)
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
        
        gts = {}
        res = {}
        for result in results:
            gts[result['id']] = [' '.join(result['ground_truth'][0])]
            res[result['id']] = [' '.join(result['generated'])]
        
        current_cider = Cider().compute_score(gts,res)[0]

        print('CIDER: {}'.format(current_cider))

        with open('logs/NIC4Product/semantics/th7/{}/val_log.txt'.format(dimension), 'a+') as f:
            f.write('CIDER: {}\n'.format(current_cider))

        return current_cider, encoder.state_dict(), decoder.state_dict()
        # if current_cider > best_cider:
        #     torch.save(decoder.state_dict(), 'checkpoint/NIC4Product/semantics/th7/best/decoder.ckpt')
        #     torch.save(encoder.state_dict(), 'checkpoint/NIC4Product/semantics/th7/best/encoder.ckpt')
        #     return dimension, current_cider
        # else:
        #     return best_experiment_number, best_bleu, best_cider, best_infer_time

    best_cider = 0

    print("Experiments Start")
    for dim in dimensions:
        print("Experiment Dimension {} Starts".format(dim))
        print('Training Dimension {} Starts'.format(dim))
        train(dim)
        # train(dim, 'checkpoint/NIC4Product/semantics/th7/state/encoder.ckpt', 'checkpoint/NIC4Product/semantics/th7/state/decoder.ckpt', 'checkpoint/NIC4Product/semantics/th7/temp/encoder.ckpt', 'checkpoint/NIC4Product/semantics/th7/temp/decoder.ckpt', 0, 0)
        print("Validating Dimension {} Starts".format(dim))
        current_cider, encoder_weight, decoder_weight = validate(dim)
        if current_cider > best_cider:
            best_cider = current_cider
            best_dim = dim
            torch.save(decoder_weight, 'checkpoint/NIC4Product/semantics/th7/best/decoder.ckpt')
            torch.save(encoder_weight, 'checkpoint/NIC4Product/semantics/th7/best/encoder.ckpt')

    shutil.copy('logs/NIC4Product/semantics/th7/{}/train_log.txt'.format(best_dim), 'logs/NIC4Product/semantics/th7/best/')
    shutil.copy('logs/NIC4Product/semantics/th7/{}/val_log.txt'.format(best_dim), 'logs/NIC4Product/semantics/th7/best/')
    shutil.copy('logs/NIC4Product/semantics/th7/{}/hyperparams.txt'.format(best_dim), 'logs/NIC4Product/semantics/th7/best/')