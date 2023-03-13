## Environment
+ Model: GIN
+ Dataset : Ogbg-molhiv / ogbg-molpcba
+ Number of layers: 3
+ Metric: Training time per epoch.
+ Machine: Aws EC2 g4dn.metal(8 T4 GPUs)
+ DGL version: 1.1
+ PyG version: 2.2.0
+ OS: ubuntu


## Run

### DGL
 python dgl_distributed.py --world_size {} --epoch {} --dataset {} --batch_size {}

### PyG 
 python pyg_distributed.py --world_size {} --epoch {} --dataset {} --batch_size {}
