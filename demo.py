import numpy
import pickle
import torch
import urllib.request
from dataset.Vocabulary import Vocabulary
from dataset.Semantic import Semantic
from models.CaT import CaT
from models.CaTSem import CaT as CaTSem
from models.NIC4Product import EncoderCNN, DecoderLSTM
from models.NIC4ProductSem import EncoderCNN as EncoderCNNSem, DecoderLSTM as DecoderLSTMSem
from models.DaErev3 import DaE
from PIL import Image
from torchvision import transforms

def generate_using_baseline(image, vocab, device):
    encoder = EncoderCNN(100, 0.001).eval().to(device)
    encoder.load_state_dict(torch.load('checkpoint/NIC4Product/best/encoder.ckpt', map_location=device))
    decoder = DecoderLSTM(100, 300, len(vocab), 35).eval().to(device)
    decoder.load_state_dict(torch.load('checkpoint/NIC4Product/best/decoder.ckpt', map_location=device))

    with torch.no_grad():
        visual_features = encoder(image)
        word_ids = decoder.predict(visual_features)
        word_ids = word_ids[0].cpu().numpy()

    generated = []
    for id in word_ids:
        if id == 2:
            break
        elif id not in [0, 1, 3]:
            generated.append(vocab.getWord(id))
    
    print(' '.join(generated))

def generate_using_baseline_semantic(image, vocab, device, semantic):
    encoder = EncoderCNNSem(100, 0.001).eval().to(device)
    encoder.load_state_dict(torch.load('checkpoint/NIC4Product/sem/th5/rev3/encoder.ckpt', map_location=device))
    decoder = DecoderLSTMSem(100, 300, len(vocab), 35, len(semantic), 512).eval().to(device)
    decoder.load_state_dict(torch.load('checkpoint/NIC4Product/sem/th5/rev3/decoder.ckpt', map_location=device))
    dae = DaE('vit', 2, 2048, len(semantic)).eval().to(device)
    dae.load_state_dict(torch.load('checkpoint/DaE/vit/th5/rev3/dae.ckpt', map_location=device))

    with torch.no_grad():
        sem = dae.predict(image)
        visual_features = encoder(image)
        word_ids = decoder.predict(visual_features, sem)
        word_ids = word_ids[0].cpu().numpy()

    generated = []
    for id in word_ids:
        if id == 2:
            break
        elif id not in [0, 1, 3]:
            generated.append(vocab.getWord(id))
    
    print(' '.join(generated))

def generate_using_transformer(image, vocab, device):
    cat = CaT(len(vocab), 12, 6000, 12, 35).eval().to(device)
    cat.load_state_dict(torch.load('checkpoint/CaT/best/cat.ckpt', map_location=device))

    word_ids = [1]
    memory = None

    with torch.no_grad():
        for i in range(35):
            tgt = torch.tensor(word_ids).unsqueeze(0).to(device)
            tgt_masks = (torch.triu(torch.ones(i+1, i+1)) == 1).transpose(0,1)
            tgt_masks = tgt_masks.float().masked_fill(tgt_masks == 0, float('-inf')).masked_fill(tgt_masks == 1, float(0.0)).to(device)
            memory, output = cat.decode(image, tgt, tgt_masks, memory)
            word_ids.append(torch.argmax(output[-1]).item())
            if word_ids[-1] == 2:
                break
    
    generated = []
    for id in word_ids:
        if id == 2:
            break
        elif id not in [0, 1, 3]:
            generated.append(vocab.getWord(id))
    
    print(' '.join(generated))

def generate_using_transformer_semantic(image, vocab, device, semantic):
    cat = CaTSem(len(vocab), len(semantic), 12, 6000, 12, 35).eval().to(device)
    cat.load_state_dict(torch.load('checkpoint/CaT/sem/cat.ckpt', map_location=device))
    dae = DaE('vit', 2, 2048, len(semantic)).eval().to(device)
    dae.load_state_dict(torch.load('checkpoint/DaE/vit/th5/rev3/dae.ckpt', map_location=device))

    word_ids = [1]
    memory = None

    with torch.no_grad():
        sem = dae.predict(image)
        for i in range(35):
            tgt = torch.tensor(word_ids).unsqueeze(0).to(device)
            tgt_masks = (torch.triu(torch.ones(i+1, i+1)) == 1).transpose(0,1)
            tgt_masks = tgt_masks.float().masked_fill(tgt_masks == 0, float('-inf')).masked_fill(tgt_masks == 1, float(0.0)).to(device)
            memory, output = cat.decode(image, tgt, sem, tgt_masks, memory)
            word_ids.append(torch.argmax(output[-1]).item())
            if word_ids[-1] == 2:
                break
    
    generated = []
    for id in word_ids:
        if id == 2:
            break
        elif id not in [0, 1, 3]:
            generated.append(vocab.getWord(id))
    
    print(' '.join(generated))

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    image_dir = 'dataset/demo/'

    with open('dataset/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('dataset/semantic5.pkl', 'rb') as f:
        semantic = pickle.load(f)

    transforms = transforms.Compose([transforms.Resize(224),transforms.RandomCrop(224),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    opener = urllib.request.build_opener()
    opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
    urllib.request.install_opener(opener)

    main_flag = True
    while main_flag:
        source = str(input('Pilih sumber gambar (local/online): '))
        filename = None
        if source == 'online':
            image_url = str(input("Masukkan url gambar: "))
            filename = image_dir + image_url.split('/')[-1]
            urllib.request.urlretrieve(image_url, filename)
        else:
            filename = str(input("Masukkan nama file: "))
            filename = image_dir + filename

        image = Image.open(filename).convert('RGB')
        image = transforms(image).unsqueeze(0).to(device)

        generation_flag = True
        while generation_flag:
            selection_flag = True
            while selection_flag:
                print('Pilih model:')
                print('1. Baseline')
                print('2. Baseline + Semantik')
                print('3. Transformer')
                print('4. Transformer + Semantik\n')
                model_selection = int(input('Pilihan model: '))

                if model_selection == 1:
                    generate_using_baseline(image, vocab, device)
                    selection_flag = False
                elif model_selection == 2:
                    generate_using_baseline_semantic(image, vocab, device, semantic)
                    selection_flag = False
                elif model_selection == 3:
                    generate_using_transformer(image, vocab, device)
                    selection_flag = False
                elif model_selection == 4:
                    generate_using_transformer_semantic(image, vocab, device, semantic)
                    selection_flag = False
                else:
                    print('Silakan pilih di antara empat model\n')
            retry = str(input('Ingin mencoba model lain? (y/n): '))
            if retry == 'n':
                generation_flag = False
        
        retry = str(input('Ingin mencoba gambar lain? (y/n): '))
        if retry == 'n':
            main_flag = False