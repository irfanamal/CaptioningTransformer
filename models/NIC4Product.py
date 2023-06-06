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
    def __init__(self, embedding_size, hidden_size, vocab_size, max_length):
        super(DecoderLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.max_length = max_length
    def forward(self, visual_features, titles, lengths):
        x = self.embedding(titles)
        x = torch.cat((visual_features.unsqueeze(1), x), 1)
        seq = pack_padded_sequence(x, lengths, batch_first=True)
        h, _ = self.lstm(seq)
        x = self.linear(h[0])
        return self.softmax(x)
    def predict(self, visual_features):
        state = None
        word_ids = []

        input = visual_features.unsqueeze(1)
        for i in range(self.max_length):
            hidden, state = self.lstm(input, state)
            output = self.linear(hidden.squeeze(1))
            output = self.softmax(output)
            _, word_id = output.max(1)
            word_ids.append(word_id)
            if word_id[0] == 2:
                break
            else:
                input = self.embedding(word_id)
                input = input.unsqueeze(1)
        return torch.stack(word_ids, 1)
        
