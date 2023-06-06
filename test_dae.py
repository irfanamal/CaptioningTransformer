import pickle
import time
import torch
from dataset.DaEDataset import DaEDataset
from dataset.Semantic import Semantic
from models.DaE import DaE
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == '__main__':
    test_batch_size = 1

    vision = 'mobilenet'
    best = 'checkpoint/DaE/{}/best/dae.ckpt'.format(vision)
    num_layer = 2
    num_unit = 2048

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('dataset/semantic.pkl', 'rb') as f:
        semantic = pickle.load(f)
    keys = list(semantic.getSemantics())

    transforms = transforms.Compose([transforms.Resize(224),transforms.RandomCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_dataset = DaEDataset('dataset/train.csv', 'dataset/images', semantic, transforms)
    test_loader = DataLoader(test_dataset, test_batch_size, False, num_workers=4)

    dae = DaE(vision, num_layer, num_unit, len(semantic)).to(device)
    dae.load_state_dict(torch.load(best))

    criterion = torch.nn.MSELoss()

    test_loss = 0
    infer_time = 0
    with torch.no_grad():
        dae.eval()

        for id, images, targets in test_loader:
            start = time.time()

            images = images.to(device)
            targets = targets.to(device)

            outputs = dae(images)
            end = time.time()

            loss = criterion(outputs, targets)

            ground_truth = torch.topk(targets[0],5)
            prediction = torch.topk(outputs[0], 5)
            with open('logs/DaE/{}/best/all_train.txt'.format(vision), 'a+') as f:
                f.write('ID: {}\n'.format(id[0]))
                f.write('Top-5 Ground Truth Semantics: {}({}), {}({}), {}({}), {}({}), {}({})\n'.format(keys[ground_truth[1][0]], ground_truth[0][0], keys[ground_truth[1][1]], ground_truth[0][1], keys[ground_truth[1][2]], ground_truth[0][2], keys[ground_truth[1][3]], ground_truth[0][3], keys[ground_truth[1][4]], ground_truth[0][4]))
                f.write('Top-5 Prediction Semantics: {}({}), {}({}), {}({}), {}({}), {}({})\n'.format(keys[prediction[1][0]], prediction[0][0], keys[prediction[1][1]], prediction[0][1], keys[prediction[1][2]], prediction[0][2], keys[prediction[1][3]], prediction[0][3], keys[prediction[1][4]], prediction[0][4]))
                f.write('Prediction Relevant Semantics: {}({}), {}({}), {}({}), {}({}), {}({})\n'.format(keys[ground_truth[1][0]], outputs[0][ground_truth[1][0]], keys[ground_truth[1][1]], outputs[0][ground_truth[1][1]], keys[ground_truth[1][2]], outputs[0][ground_truth[1][2]], keys[ground_truth[1][3]], outputs[0][ground_truth[1][3]], keys[ground_truth[1][4]], outputs[0][ground_truth[1][4]]))
                f.write('MSE Loss: {}\n'.format(loss.item()))
                f.write('Inference Time: {}\n\n'.format(end-start))
            test_loss += loss.item()
            infer_time += end-start
    
    with open('logs/DaE/{}/best/summary_train.txt'.format(vision), 'a+') as f:
        f.write('MSE Loss: {}\n'.format(test_loss/len(test_loader)))
        f.write('Inference Time: {}\n'.format(infer_time/len(test_loader)))
    
    print('MSE Loss: {}'.format(test_loss/len(test_loader)))
    print('Inference Time: {}'.format(infer_time/len(test_loader)))