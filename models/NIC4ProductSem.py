import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import models

class EncoderCNN(nn.Module):
    def __init__(self, embedding_size, momentum):
        super(EncoderCNN, self).__init__()
        mobile_net = models.mobilenet_v2(pretrained=True)
        mobile_net.classifier[1] = nn.Linear(mobile_net.last_channel, embedding_size)
        self.mobile_net = mobile_net
        self.batchnorm = nn.BatchNorm1d(embedding_size, momentum=momentum)
    def forward(self, images):
        x = self.mobile_net(images)
        return self.batchnorm(x)

class DecoderLSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, max_length, semantic_size, input_proj_size):
        super(DecoderLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(input_proj_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.we = nn.Linear(embedding_size, input_proj_size)
        self.ws = nn.Linear(semantic_size, input_proj_size)
        self.max_length = max_length
    def forward(self, visual_features, titles, lengths, semantics):
        x1 = self.embedding(titles)
        x1 = torch.cat((visual_features.unsqueeze(1), x1), 1)
        x1 = self.we(x1)
        x2 = self.ws(semantics)
        x = torch.mul(x1,x2.unsqueeze(1))
        seq = pack_padded_sequence(x, lengths, batch_first=True)
        h, _ = self.lstm(seq)
        x = self.linear(h[0])
        return self.softmax(x)
    def predict(self, visual_features, semantics):
        state = None
        word_ids = []
        sem_input = semantics
        vis_input = visual_features
        sem_embed = self.ws(sem_input)
        vis_embed = self.we(vis_input)
        input = torch.mul(vis_embed, sem_embed)
        input = input.unsqueeze(1)
        for i in range(self.max_length):
            hidden, state = self.lstm(input, state)
            output = self.linear(hidden.squeeze(1))
            output = self.softmax(output)
            _, word_id = output.max(1)
            word_ids.append(word_id)
            if word_id[0] == 2:
                break
            else:
                tkn_input = self.embedding(word_id)
                tkn_embed = self.we(tkn_input)
                input = torch.mul(tkn_embed, sem_embed)
                input = input.unsqueeze(1)
        return torch.stack(word_ids, 1)
        
