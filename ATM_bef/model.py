import torch
import torch.nn as nn
# MODEL
class Enhance(nn.Module):
    def __init__(self):
        super(Enhance,self).__init__()
        self.LSTM1 = nn.LSTM(
                input_size = 257, 
                hidden_size = 300,
                num_layers = 1,  
                batch_first=True)
        self.LSTM2 = nn.LSTM(
                input_size = 300, 
                hidden_size = 300,
                num_layers = 1,  
                batch_first=True)

        self.dense_spk1 = nn.Linear(256,300)
        self.dense_spk2 = nn.Linear(300,300)
        self.dense_rec = nn.Linear(300,257)
        self.relu = torch.nn.ReLU()
        self.leakyrelu = torch.nn.LeakyReLU()

    def forward(self, input, dense3_out):
        lstm1_out, (h_n,h_c) = self.LSTM1(input,None)


        if dense3_out == None:
            lstm2_out, (h_n,h_c) = self.LSTM2(lstm1_out.reshape(1,lstm1_out.shape[1],lstm1_out.shape[2]),None)
        else:
            dense3_out = self.relu(self.dense_spk1(dense3_out))
            dense3_out = self.relu(self.dense_spk2(dense3_out))
            lstm1_out = (torch.squeeze(lstm1_out) * dense3_out)
            lstm2_out, (h_n,h_c) = self.LSTM2(lstm1_out.reshape(1,lstm1_out.shape[0],lstm1_out.shape[1]),None)

        output_enh = self.dense_rec(lstm2_out)

        return output_enh, lstm2_out

class Speaker(nn.Module):
    def __init__(self):
        super(Speaker,self).__init__()
        self.dense1 = nn.Linear(3300,1024)
        self.dense2 = nn.Linear(1024,1024)
        self.dense3 = nn.Linear(1024,256)
        self.dense4 = nn.Linear(256,7)
        self.dense_lstm_out = nn.Linear(3300,1024)
        self.softmax = torch.nn.Softmax(dim=1)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, input):
        dense1_out = self.relu(self.drop(self.dense1(input)))

        dense2_out = self.relu(self.drop(self.dense2(dense1_out)))

        dense3_out = self.relu(self.drop(self.dense3(dense2_out)))
        output_speaker = self.dense4(dense3_out)

        return output_speaker , dense3_out
