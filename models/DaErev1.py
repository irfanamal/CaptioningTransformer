import torch
import torchvision
import timm

class DaE(torch.nn.Module):
    def __init__(self, vision, num_layers, num_nodes, num_semantics):
        super().__init__()
        cv = None
        if vision == 'mobilenet':
            cv = torchvision.models.mobilenet_v2(pretrained=True)
            cv.classifier[1] = torch.nn.Linear(cv.last_channel, num_nodes)
        elif vision == 'resnet':
            cv = torchvision.models.resnet152(pretrained=True)
            cv.fc = torch.nn.Linear(cv.fc.in_features, num_nodes)
        elif vision == 'vit':
            cv = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_nodes)
        self.vision = cv
        self.batchnorm1 = torch.nn.BatchNorm1d(num_nodes)
        self.activation1 = torch.nn.ReLU(inplace=True)
        hidden_block = torch.nn.Sequential(torch.nn.Linear(num_nodes, num_nodes), torch.nn.BatchNorm1d(num_nodes), torch.nn.ReLU(inplace=True))
        if num_layers > 1:
            self.fcs = torch.nn.Sequential(*[hidden_block for _ in range(1,num_layers)])
        else:
            self.fcs = torch.nn.Identity()
        self.regressor = torch.nn.Sequential(torch.nn.Linear(num_nodes, num_semantics), torch.nn.BatchNorm1d(num_semantics), torch.nn.ReLU(inplace=True))
    def forward(self, images):
        x = self.vision(images)
        x = self.batchnorm1(x)
        x = self.activation1(x)
        x = self.fcs(x)
        return self.regressor(x)