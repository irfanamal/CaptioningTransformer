import dataset.ZaloraDataset as zd
from dataset.Vocabulary import Vocabulary
import pickle
from torchvision import transforms
from torch.utils.data import DataLoader

with open('dataset/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

transforms = transforms.Compose([transforms.Resize(224),transforms.RandomCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_dataset = zd.ZaloraDataset('dataset/train.csv', 'dataset/images', vocab, transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True, num_workers=2, collate_fn=zd.collate_fn)

if __name__=='__main__':
    maximum = 0
    for _, _, _, lengths in train_loader:
        if max(lengths) > maximum:
            maximum = max(lengths)
    print(maximum)