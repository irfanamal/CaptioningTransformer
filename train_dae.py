import os
import pickle
import shutil
import time
import torch
from dataset.DaEDataset import DaEDataset
from dataset.Semantic import Semantic
from models.DaE import DaE
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == '__main__':
    train_batch_size = 16
    val_batch_size = 1
    vision = 'mobilenet'
    state = 'checkpoint/DaE/mobilenet/state/dae.ckpt'
    temp = 'checkpoint/DaE/mobilenet/temp/dae.ckpt'

    best_num_layer = 2
    best_num_unit = 2048
    best_learning_rate = 0.1
    best_momentum = 0

    num_layers = [3, 2, 1]
    num_units = [2048, 1024, 512]
    learning_rates = [0.1, 0.01, 0.001]
    momentums = [0, 0.001, 0.01]
    one_factor_at_a_time = [num_layers, num_units, learning_rates, momentums]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('dataset/semantic.pkl', 'rb') as f:
        semantic = pickle.load(f)

    transforms = transforms.Compose([transforms.Resize(224),transforms.RandomCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = DaEDataset('dataset/train.csv', 'dataset/images', semantic, transforms)
    val_dataset = DaEDataset('dataset/val.csv', 'dataset/images', semantic, transforms)
    train_loader = DataLoader(train_dataset,train_batch_size,True,num_workers=2)
    val_loader = DataLoader(val_dataset,val_batch_size,False,num_workers=4)

    def train(experiment_number, num_layer, num_unit, vision, learning_rate, momentum, state=None, local_optima=None, temp_val_loss=float('inf'), temp_infer_time=0, start_epoch=1):
        if not os.path.exists('logs/DaE/{}/{}'.format(vision, experiment_number)):
            os.mkdir('logs/DaE/{}/{}'.format(vision, experiment_number))
        
        dae = DaE(vision, num_layer, num_unit, len(semantic)).to(device)

        criterion = torch.nn.MSELoss()
        nesterov = False
        if momentum > 0:
            nesterov = True
        optimizer = torch.optim.SGD(dae.parameters(), lr=learning_rate, momentum=momentum, nesterov=nesterov)

        min_val_loss = temp_val_loss
        infer_time = temp_infer_time
        weights = dae.state_dict()
        if local_optima != None:
            weights = torch.load(local_optima)

        epoch = start_epoch
        if state != None:
            dae.load_state_dict(torch.load(state))
            
        while epoch <= 50:
            start_train = time.time()

            train_loss_sum = 0
            dae.train()

            for _, images, targets in train_loader:
                images = images.to(device)
                targets = targets.to(device)

                outputs = dae(images)

                loss = criterion(outputs,targets)
                dae.zero_grad()

                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * targets.size(0)

            end_train = time.time()

            torch.save(dae.state_dict(), 'checkpoint/DaE/{}/state/dae.ckpt'.format(vision))

            val_loss_sum = 0
            infer_time_sum = 0
            with torch.no_grad():
                dae.eval()

                for _, images, targets in val_loader:
                    start_predict = time.time()
                    images = images.to(device)
                    targets = targets.to(device)

                    outputs = dae(images)

                    loss = criterion(outputs,targets)

                    val_loss_sum += loss.item() * targets.size(0)
                    end_predict = time.time()
                    infer_time_sum += end_predict - start_predict
            
            train_loss = train_loss_sum/len(train_dataset)
            val_loss = val_loss_sum/len(val_dataset)
            training_time = end_train - start_train

            with open('logs/DaE/{}/{}/train_log.txt'.format(vision, experiment_number), 'a+') as f:
                f.write('Epoch: {}, Train Loss: {}, Validation Loss: {}, Training Time: {}\n'.format(epoch, train_loss, val_loss, training_time))
            print("Epoch: {}, Train Loss: {}, Validation Loss: {}, Training Time: {}".format(epoch, train_loss, val_loss, training_time))
            epoch += 1

            if val_loss <= min_val_loss:
                torch.save(dae.state_dict(), 'checkpoint/DaE/{}/temp/dae.ckpt'.format(vision))
                weights = dae.state_dict()
                min_val_loss = val_loss
                infer_time = infer_time_sum/len(val_loader)
                with open('checkpoint/DaE/{}/temp/logs.txt'.format(vision), 'w+') as f:
                    f.write('Val Loss:{}\n'.format(min_val_loss))
                    f.write('Infer Time:{}\n'.format(infer_time))
        
        with open('logs/DaE/{}/{}/hyperparams.txt'.format(vision, experiment_number), 'a+') as f:
            f.write('num_layer: {}\n'.format(num_layer))
            f.write('num_unit: {}\n'.format(num_unit))
            f.write('learning_rate: {}\n'.format(learning_rate))
            f.write('momentum: {}\n'.format(momentum))

        with open('logs/DaE/{}/{}/val_log.txt'.format(vision,experiment_number), 'a+') as f:
            f.write('MSE: {}\n'.format(min_val_loss))
            f.write('Inference Time: {}\n'.format(infer_time))
        print('MSE: {}'.format(min_val_loss))
        print('Inference Time: {}'.format(infer_time))
        
        return min_val_loss, weights

    best_experiment_number = 2
    best_mse = 0.0003772205853660621

    experiment_number = 9
    for i,factors in enumerate(one_factor_at_a_time):
        for j,factor in enumerate(factors):
            if i==0:
                # if j==2:
                #     print("Experiment Number {}".format(experiment_number))
                #     print('Training Starts')
                #     mse, weights = train(experiment_number, factor, best_num_unit, vision, best_learning_rate, best_momentum, state, temp, 0.0012613250165211807, 0.00910241975922747, 48)
                #     if mse < best_mse:
                #         torch.save(weights, 'checkpoint/DaE/{}/best/dae.ckpt'.format(vision))
                #         best_experiment_number = experiment_number
                #         best_mse = mse
                #         best_num_layer = factor
                #     print('Best Experiment Number: {}'.format(best_experiment_number))
                #     experiment_number += 1
                pass
            elif i==1:
                # if j>0:
                #     print("Experiment Number {}".format(experiment_number))
                #     print('Training Starts')
                #     mse, weights = train(experiment_number, best_num_layer, factor, vision, best_learning_rate, best_momentum)
                #     if mse < best_mse:
                #         torch.save(weights, 'checkpoint/DaE/{}/best/dae.ckpt'.format(vision))
                #         best_experiment_number = experiment_number
                #         best_mse = mse
                #         best_num_unit = factor
                #     print('Best Experiment Number: {}'.format(best_experiment_number))
                #     experiment_number += 1
                pass
            elif i==2:
                # if j>0:
                #     print("Experiment Number {}".format(experiment_number))
                #     print('Training Starts')
                #     mse, weights = train(experiment_number, best_num_layer, best_num_unit, vision, factor, best_momentum)
                #     if mse < best_mse:
                #         torch.save(weights, 'checkpoint/DaE/{}/best/dae.ckpt'.format(vision))
                #         best_experiment_number = experiment_number
                #         best_mse = mse
                #         best_learning_rate = factor
                #     print('Best Experiment Number: {}'.format(best_experiment_number))
                #     experiment_number += 1
                pass
            elif i==3:
                if j==2:
                    print("Experiment Number {}".format(experiment_number))
                    print('Training Starts')
                    mse, weights = train(experiment_number, best_num_layer, best_num_unit, vision, best_learning_rate, factor, state, temp, 5.439461562994833e-05, 0.009400145858443029, 33)
                    if mse < best_mse:
                        torch.save(weights, 'checkpoint/DaE/{}/best/dae.ckpt'.format(vision))
                        best_experiment_number = experiment_number
                        best_mse = mse
                        best_momentum = factor
                    print('Best Experiment Number: {}'.format(best_experiment_number))
                    experiment_number += 1
    
    shutil.copy('logs/DaE/{}/{}/train_log.txt'.format(vision,best_experiment_number), 'logs/DaE/{}/best/'.format(vision))
    shutil.copy('logs/DaE/{}/{}/val_log.txt'.format(vision,best_experiment_number), 'logs/DaE/{}/best/'.format(vision))
    shutil.copy('logs/DaE/{}/{}/hyperparams.txt'.format(vision,best_experiment_number), 'logs/DaE/{}/best/'.format(vision))