COL761 A3 Q2 Submission Notes

Kerberos: aib252562

Q2 uses the official interface:
python train.py --dataset A|B|C --task node|link --data_dir /path/to/datasets --model_dir /path/to/models --kerberos YOUR_KERBEROS

Extra dependency used:
- pyg_lib==0.5.0+pt27cu118

Reason:
Graph B uses torch_geometric.loader.NeighborLoader for mini-batch GNN training on the large graph. PyTorch Geometric requires either pyg-lib or torch-sparse for NeighborLoader sampling.

Install command used:
pip install pyg_lib -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
