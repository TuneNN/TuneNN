import torch
import torch.nn as nn


class TuneEncoderFre(nn.Module):
    def __init__(self, d_model, nhead, n_layers=5, dropout=0.5):
        super(TuneEncoderFre, self).__init__()
        self.d_model = d_model
        self.attn_layer = nn.ModuleList([nn.MultiheadAttention(d_model, nhead) for _ in range(n_layers)])
        self.norm_layer = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, feature_dim)
        print(x.shape)
        _, feature_dim = x.shape
        x = x.view(x.shape[0], -1, self.d_model)
        for attn in self.attn_layer:
            x, _ = attn(x, x, x)
            x = self.norm_layer(x)
        x = self.dropout(x)

        x = x.view(-1, feature_dim)
        return x


class TuneEncoderTime(nn.Module):
    def __init__(self, d_model, nhead, n_layers=5, dropout=0.5):
        super(TuneEncoderTime, self).__init__()
        self.attn_layer = nn.ModuleList([nn.MultiheadAttention(d_model, nhead) for _ in range(n_layers)])
        self.norm_layer = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        for attn in self.attn_layer:
            x, _ = attn(x, x, x)
            x = self.norm_layer(x)
        x = self.dropout(x)

        return x


class TuneEncoder(nn.Module):
    def __init__(self, fre_d_model, fre_nhead, time_d_model, time_nhead, fre_layers=5, time_layers=5):
        super(TuneEncoder, self).__init__()
        self.fre_layer = TuneEncoderFre(d_model=fre_d_model, nhead=fre_nhead, n_layers=fre_layers, dropout=0.5)
        self.time_layer = TuneEncoderTime(d_model=time_d_model, nhead=time_nhead, n_layers=time_layers, dropout=0.5)

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        _, seq_len, feature_len = x.shape
        x1 = x[:, seq_len // 2, :]
        out_fre = self.fre_layer(x1)
        out_time = self.time_layer(x)
        return out_fre, out_time


class TuneDecoder(nn.Module):

    def __init__(self, d_model=512, nhead=8, n_layers=5, dropout=0.5):
        super(TuneDecoder, self).__init__()
        self.attn_layer = nn.ModuleList([nn.MultiheadAttention(d_model, nhead) for _ in range(n_layers)])
        self.norm_layer = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(1)
        x = torch.hstack((x1, x2))
        for attn in self.attn_layer:
            x, _ = attn(x, x, x)
            x = self.norm_layer(x)
        x = self.dropout(x)
        return x


class TuneTransformer(nn.Module):

    def __init__(self, n_fre, n_feature, fre_nhead=8, time_nhead=8, d_head=8, fre_layers=5, time_layers=5, d_layers=5):
        super().__init__()
        self.encoder = TuneEncoder(n_fre, fre_nhead, n_feature, time_nhead,
                                   fre_layers=fre_layers, time_layers=time_layers)
        self.decoder = TuneDecoder(n_feature, d_head, n_layers=d_layers, dropout=0.5)

    def forward(self, x):
        f_fre, f_time = self.encoder(x)
        x = self.decoder(f_fre, f_time)
        return x


class TuneNN(nn.Module):

    def __init__(self, n_stft_feature, n_bark_feature, n_ceq_feature, n_cqhc_feature, n_time,
                 n_stft_fre=8, n_bark_fre=8, n_ceq_fre=8, n_cqhc_fre=8,
                 stft_fre_nhead=8, stft_time_nhead=8, stft_d_head=8, stft_fre_layers=5, stft_time_layers=5,
                 stft_d_layers=5,
                 bark_fre_nhead=8, bark_time_nhead=8, bark_d_head=8, bark_fre_layers=5, bark_time_layers=5,
                 bark_d_layers=5,
                 ceq_fre_nhead=8, ceq_time_nhead=8, ceq_d_head=8, ceq_fre_layers=5, ceq_time_layers=5, ceq_d_layers=5,
                 cqhc_fre_nhead=8, cqhc_time_nhead=8, cqhc_d_head=8, cqhc_fre_layers=5, cqhc_time_layers=5,
                 cqhc_d_layers=5,
                 stft_weight=3, bark_weight=3, ceq_weight=2, cqhc_weight=2,
                 num_labels=336):
        super(TuneNN, self).__init__()

        self.stft_layer = TuneTransformer(n_stft_fre, n_stft_feature, fre_nhead=stft_fre_nhead,
                                          time_nhead=stft_time_nhead,
                                          d_head=stft_d_head, fre_layers=stft_fre_layers, time_layers=stft_time_layers,
                                          d_layers=stft_d_layers)
        self.bark_layer = TuneTransformer(n_bark_fre, n_bark_feature, fre_nhead=bark_fre_nhead,
                                          time_nhead=bark_time_nhead,
                                          d_head=bark_d_head, fre_layers=bark_fre_layers, time_layers=bark_time_layers,
                                          d_layers=bark_d_layers)
        self.cep_layer = TuneTransformer(n_ceq_fre, n_ceq_feature, fre_nhead=ceq_fre_nhead, time_nhead=ceq_time_nhead,
                                         d_head=ceq_d_head, fre_layers=ceq_fre_layers, time_layers=ceq_time_layers,
                                         d_layers=ceq_d_layers)
        self.cqhc_layer = TuneTransformer(n_cqhc_fre, n_cqhc_feature, fre_nhead=cqhc_fre_nhead,
                                          time_nhead=cqhc_time_nhead,
                                          d_head=cqhc_d_head, fre_layers=cqhc_fre_layers, time_layers=cqhc_time_layers,
                                          d_layers=cqhc_d_layers)
        self.stft_pool_layer = nn.AvgPool2d((1, 4))
        self.cep_pool_layer = nn.AvgPool2d((1, 4))

        n_feature = (n_stft_feature // 4 + n_bark_feature + n_ceq_feature // 4 + n_cqhc_feature) * (n_time + 1)
        print(n_feature)
        self.fc_layers = nn.Sequential(
            nn.Linear(n_feature, 2048, bias=True),
            nn.Linear(2048, 1024, bias=True),
            nn.Linear(1024, num_labels, bias=True),
        )

    def forward(self, x1, x2, x3, x4):
        stft_feature = self.stft_layer(x1)
        bark_feature = self.bark_layer(x2)
        cep_feature = self.cep_layer(x3)
        cqhc_feature = self.cqhc_layer(x4)

        stft_feature = self.stft_pool_layer(stft_feature)
        cep_feature = self.cep_pool_layer(cep_feature)

        stft_feature = stft_feature.view(stft_feature.shape[0], -1)
        bark_feature = bark_feature.view(bark_feature.shape[0], -1)
        cep_feature = cep_feature.view(cep_feature.shape[0], -1)
        cqhc_feature = cqhc_feature.view(cqhc_feature.shape[0], -1)

        x = torch.cat((stft_feature, bark_feature, cep_feature, cqhc_feature), dim=1)

        x = self.fc_layers(x)
        return x
