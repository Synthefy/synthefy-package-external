import torch


class LinearDecoder(torch.nn.Module):
    def __init__(self, d_model, output_dim, dropout=0.1):
        super(LinearDecoder, self).__init__()
        self.linear = torch.nn.Linear(d_model, output_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, embeddings: torch.Tensor, x: torch.Tensor):
        x = self.dropout(embeddings)
        x = self.linear(x)

        return x


class MLPDecoder(torch.nn.Module):
    def __init__(
        self,
        d_model,
        output_dim,
        dropout=0.1,
        num_layers=1,
        hidden_dim=128,
        activation="gelu",
    ):
        super(MLPDecoder, self).__init__()
        self.num_layers = num_layers
        if self.num_layers > 1:
            self.first_linear = torch.nn.Sequential(
                torch.nn.Linear(d_model, hidden_dim),
                torch.nn.ReLU() if activation == "relu" else torch.nn.GELU(),
            )
        self.layers = torch.nn.Sequential(
            *[
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU()
                    if activation == "relu"
                    else torch.nn.GELU(),
                )
                for _ in range(num_layers - 2)
            ]
        )
        self.final_linear = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, embeddings: torch.Tensor, x: torch.Tensor):
        if self.num_layers > 1:
            embeddings = self.first_linear(embeddings)
        x = self.layers(embeddings)
        x = self.dropout(embeddings)
        x = self.final_linear(embeddings)

        return x
