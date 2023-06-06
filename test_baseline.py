import dataset.ZaloraDataset as zd
import nltk
import numpy
import pandas as pd
import pickle
import time
import torch
from cider.cider import Cider
from dataset.Vocabulary import Vocabulary
from models.NIC4Product import EncoderCNN, DecoderLSTM
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == '__main__':
    embedding_size = 100
    hidden_size = 300
    max_length = 33
    batch_norm_momentum = 0.001
    test_batch_size = 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('dataset/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    transforms = transforms.Compose([transforms.Resize(224),transforms.RandomCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_dataset = zd.ZaloraDataset('dataset/test.csv', 'dataset/images', vocab, transforms)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4, collate_fn=zd.collate_fn)

    encoder = EncoderCNN(embedding_size, batch_norm_momentum).eval().to(device)
    decoder = DecoderLSTM(embedding_size, hidden_size, len(vocab), max_length).eval().to(device)

    encoder.load_state_dict(torch.load('checkpoints/NIC4Product/best/encoder.ckpt'))
    decoder.load_state_dict(torch.load('checkpoints/NIC4Product/best/decoder.ckpt'))

    df = pd.read_csv('dataset/test.csv')

    results = []
    with torch.no_grad():
        for i, (ids, images, titles, lengths) in enumerate(test_loader):
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
    
    bleu_1s = []
    bleu_2s = []
    bleu_3s = []
    bleu_4s = []
    times = []
    gts = {}
    res = {}
    for result in results:
        bleu_1s.append(nltk.translate.bleu_score.sentence_bleu(result['ground_truth'], result['generated'], weights=(1,0,0,0), smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method7))
        bleu_2s.append(nltk.translate.bleu_score.sentence_bleu(result['ground_truth'], result['generated'], weights=(0,1,0,0), smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method7))
        bleu_3s.append(nltk.translate.bleu_score.sentence_bleu(result['ground_truth'], result['generated'], weights=(0,0,1,0), smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method7))
        bleu_4s.append(nltk.translate.bleu_score.sentence_bleu(result['ground_truth'], result['generated'], weights=(0,0,0,1), smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method7))
        times.append(result['test_time'])
        res[result['id']] = [' '.join(result['generated'])]
        gts[result['id']] = [' '.join(result['ground_truth'][0])]
    
    cider_score = Cider().compute_score(gts,res)
    with open('logs/NIC4Product/best/all_test.txt', 'a+') as f:
        for i,result in enumerate(results):
            f.write('ID: {}\n'.format(result['id']))
            f.write('Ground Truth: {}\n'.format(gts[result['id']]))
            f.write('Generated: {}\n'.format(res[result['id']]))
            f.write('BLEU-1: {}\n'.format(bleu_1s[i]))
            f.write('BLEU-2: {}\n'.format(bleu_2s[i]))
            f.write('BLEU-3: {}\n'.format(bleu_3s[i]))
            f.write('BLEU-4: {}\n'.format(bleu_4s[i]))
            f.write('CIDER: {}\n'.format(cider_score[1][i]))
            f.write('Inference Time: {}\n\n'.format(times[i]))
    
    with open('logs/NIC4Product/best/summary_test.txt', 'a+') as f:
        f.write('BLEU-1: {}\n'.format(numpy.mean(bleu_1s)))
        f.write('BLEU-2: {}\n'.format(numpy.mean(bleu_2s)))
        f.write('BLEU-3: {}\n'.format(numpy.mean(bleu_3s)))
        f.write('BLEU-4: {}\n'.format(numpy.mean(bleu_4s)))
        f.write('CIDER: {}\n'.format(cider_score[0]))
        f.write('Inference Time: {}\n'.format(numpy.mean(times)))

    print('BLEU-1: {}'.format(numpy.mean(bleu_1s)))
    print('BLEU-2: {}'.format(numpy.mean(bleu_2s)))
    print('BLEU-3: {}'.format(numpy.mean(bleu_3s)))
    print('BLEU-4: {}'.format(numpy.mean(bleu_4s)))
    print('CIDER: {}'.format(cider_score[0]))
    print('Inference Time: {}'.format(numpy.mean(times)))