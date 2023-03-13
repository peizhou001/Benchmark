import argparse
import os
import time

import torch

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from ogb.graphproppred import Evaluator
from ogb.graphproppred import PygGraphPropPredDataset as Dataset
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool


class GIN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers=3,
                 dropout=0.5):
        super().__init__()

        self.dropout = dropout

        self.atom_encoder = AtomEncoder(hidden_channels)
        self.bond_encoder = BondEncoder(hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(hidden_channels, 2 * hidden_channels),
                BatchNorm(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
                BatchNorm(hidden_channels),
                ReLU(),
            )
            self.convs.append(GINEConv(nn, train_eps=True))

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, adj_t, batch):
        x = self.atom_encoder(x)
        edge_attr = adj_t.coo()[2]
        adj_t = adj_t.set_value(self.bond_encoder(edge_attr), layout='coo')

        for conv in self.convs:
            x = conv(x, adj_t)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x


def run(rank, world_size: int, dataset_name: str, root: str, batch_size: int, epoch_num: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    
    torch.manual_seed(12345)
    
    dataset = Dataset(dataset_name, root,
                      pre_transform=T.ToSparseTensor(attr='edge_attr'))
    split_idx = dataset.get_idx_split()

    train_dataset = dataset[split_idx['train']]
    train_sampler = DistributedSampler(train_dataset, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=train_sampler)
    
    model = GIN(300, dataset.num_tasks, num_layers=3, dropout=0.5).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    s = time.time()
    to_time = 0.
    fw_time = 0.
    bw_time = 0.
    optim_time = 0.
    for epoch in range(epoch_num):
        model.train()
        train_sampler.set_epoch(epoch)
        for data in train_loader:
            se = time.time()
            data = data.to(rank)
            to_time += (time.time() - se)
            
            se = time.time()
            optimizer.zero_grad()
            logits = model(data.x, data.adj_t, data.batch)
            loss = criterion(logits, data.y.to(torch.float))
            fw_time += (time.time() - se)
            
            se = time.time()
            loss.backward()
            bw_time += (time.time() - se)
            se = time.time()
            optimizer.step()
            optim_time += (time.time() - se)
    print(f"{rank} to_time: {to_time / float(epoch_num)}")
    print(f"{rank} fw_time: {fw_time / float(epoch_num)} ")
    print(f"{rank} bw_time: {bw_time / float(epoch_num)}")
    print(f"{rank} optim_time: {optim_time / float(epoch_num)}")
    print(f"-------")
    if rank == 0:
        print(f"elpased time: {(time.time() - s) / float(epoch_num)} for {world_size} gpus for each epoch ")
    dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':
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
    root = '.'

    # Download and process the dataset on main process.
    Dataset(dataset_name, root,
            pre_transform=T.ToSparseTensor(attr='edge_attr'))

    world_size = parser.parse_args().world_size
    batch_size = parser.parse_args().batch_size
    epoch = parser.parse_args().epoch
    print('Let\'s use', world_size, 'GPUs!')
    args = (world_size, dataset_name, root, batch_size, epoch)

    mp.spawn(run, args=args, nprocs=world_size, join=True)
