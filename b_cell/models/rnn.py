from torch import nn

class RNNModule(nn.Module) :
    def __init__(
        self,
        embedding_dim: int,
        total_padded_length: int,
        dropout: float,
        rnn_hidden_dim: int,
        rnn_num_layers: int,
        rnn_bidirectional: bool = True,
        return_hidden_state: bool = False,
    ) :
        super().__init__()

        self.total_padded_length = total_padded_length
        self.return_hidden_state = return_hidden_state

        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
            bias=True,
            dropout=dropout,
            bidirectional=rnn_bidirectional,
        )

    def forward(self, emb, lens) :
        emb_packed = nn.utils.rnn.pack_padded_sequence(emb, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, hn = self.rnn(emb_packed)
        out_padded, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=self.total_padded_length)
        return out_padded if not self.return_hidden_state else (out_padded, hn)
