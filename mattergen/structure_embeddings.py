import torch
from mattergen.common.data.chemgraph import ChemGraph

class SiteStructureEmbedding(torch.nn.Module):
    def __init__(
        self,
        name: str,
        embedding_module: torch.nn.Module,
        scaler: torch.nn.Module = torch.nn.Identity(),
        emb_size: int = 512,
    ):
        super().__init__()
        self.name = name
        self.embedding_module = embedding_module
        self.scaler = scaler
        self.emb_size = emb_size

    def forward(self, batch: ChemGraph) -> torch.Tensor:
        data = batch[self.name]
        if isinstance(data, torch.Tensor) and data.dim() == 2:
            data = data.squeeze(-1) # [B, 1] => [B,]

        # optionally apply normalization, eg unit standard deviation and zero mean
        data = self.scaler(data)
        return self.embedding_module(data)

    def fit_scaler(self, all_data):
        if isinstance(self.scaler, torch.nn.Identity):
            return
        self.scaler.fit(all_data)
