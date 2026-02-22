import torch
import torch.nn as nn
import torch.nn.functional as F




class MultiLinearModel(nn.Module):
    def __init__(self, seq_len, pred_len, num_loops=2, dropout=0.4):
        super(MultiLinearModel, self).__init__()
        self.num_loops = num_loops
        self.pred_len = pred_len

        self.fc1 = nn.Linear(seq_len, pred_len * 4)
        self.fc2 = nn.Linear(pred_len * 4, pred_len * 2)
        self.fc3 = nn.Linear(pred_len * 2, pred_len)

        self.dropout = nn.Dropout(dropout)
        self.weighted_linear = nn.Linear(num_loops, 1, bias=True)

    def forward(self, input_data):
        transformed_data = [input_data.unsqueeze(-1)]
        for i in range(2, self.num_loops + 1):
            transformed = input_data.clone()
            transformed[:, 1, :] = torch.sign(input_data[:, 1, :]) * (torch.abs(input_data[:, 1, :]) ** (1 / i))
            transformed_data.append(transformed.unsqueeze(-1))

        concatenated_data = torch.cat(transformed_data, dim=-1)  # [B, L, C, num_loops]

        B, L, C, N = concatenated_data.shape
        x = concatenated_data.permute(0, 3, 2, 1).reshape(B * N * C, L)  # [B*N*C, L]

        # MLP
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)  # [B*N*C, pred_len]

        #  [B, N, C, pred_len]
        x = x.view(B, N, C, self.pred_len).permute(0, 3, 2, 1)  # [B, pred_len,C, N]


        output = self.weighted_linear(x).squeeze(-1)  # [B, pred_len, C]

        return output  # [B, pred_len, C]

