import argparse
import os
import time

import dgl
from dgl.nn import AvgPooling, GINEConv

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential
from torch.nn.parallel import DistributedDataParallel
from dgl.data import AsGraphPredDataset
from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import Evaluator
from ogb.graphproppred import DglGraphPropPredDataset as Dataset
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from tqdm import tqdm


class GIN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers=3,
                 dropout=0.5):
        super().__init__()

        self.dropout = dropout

        self.atom_encoder = AtomEncoder(hidden_channels)
        self.bond_encoder = BondEncoder(hidden_channels)

        self.convs = torch.nn.ModuleList()
        self.pool = AvgPooling()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(hidden_channels, 2 * hidden_channels),
                BatchNorm(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
                BatchNorm(hidden_channels),
                ReLU(),
            )
            self.convs.append(GINEConv(nn, learn_eps=True))

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, g, x, x_e):
        x = self.atom_encoder(x)
        xe = self.bond_encoder(x_e)

        for conv in self.convs:
            x = conv(g, x, xe)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.pool(g, x)
        x = self.lin(x)
        return x

@torch.no_grad()
def evaluate(dataloader, device, model, evaluator):
    model.eval()
    y_true = []
    y_pred = []
    for batched_graph, labels in tqdm(dataloader):
        batched_graph, labels = batched_graph.to(device), labels.to(device)
        node_feat, edge_feat = (
            batched_graph.ndata["feat"],
            batched_graph.edata["feat"],
        )
        y_hat = model(batched_graph, node_feat, edge_feat)
        y_true.append(labels.view(y_hat.shape).detach().cpu())
        y_pred.append(y_hat.detach().cpu())
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)

def run(rank, world_size: int, dataset_name: str, root: str, batch_size: int, epoch_num: int):
    dgl.utils.set_num_threads(11)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12347'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    
    torch.manual_seed(12345)
 
    dataset = AsGraphPredDataset(Dataset(dataset_name, root))
    train_loader = GraphDataLoader(
        dataset[dataset.train_idx], batch_size=batch_size, use_ddp=True, shuffle=True
    )

    model = GIN(300, dataset.num_tasks, num_layers=3, dropout=0.5).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    print("start!")
    s = time.perf_counter()
    for epoch in range(epoch_num):
        count = 0
        model.train()
        train_loader.set_epoch(epoch)
        for g, y in train_loader:
            g, y = g.to(rank), y.to(rank)
            optimizer.zero_grad()
            x, xe = (g.ndata["feat"], g.edata["feat"])
            logits = model(g, x, xe)
            loss = criterion(logits, y.to(torch.float))
            loss.backward()
            optimizer.step()
    if rank == 0:
        print(f"elpased time: {(time.perf_counter() - s) / float(epoch_num)} for {world_size} gpus for each epoch ")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    print(os.getpid())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbg-molhiv",
        choices=["ogbg-molhiv", "ogbg-molpcba"]
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=torch.cuda.device_count()
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=10
    ) 
    dataset_name = parser.parse_args().dataset
    root = "."
    
    # Download and process the dataset on main process.
    Dataset(dataset_name, root)

    world_size = parser.parse_args().world_size
    batch_size = parser.parse_args().batch_size
    epoch = parser.parse_args().epoch
    print("Let's use", world_size, "GPUs!")
    args = (world_size, dataset_name, root, batch_size, epoch)
    import torch.multiprocessing as mp

    mp.spawn(run, args=args, nprocs=world_size, join=True)
