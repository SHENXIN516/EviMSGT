import torch
from torch_sparse import SparseTensor # for propagation

def k_hop_subgraph(edge_index, num_nodes, num_hops):
    # return k-hop subgraphs for all nodes in the graph
    row, col = edge_index.to(torch.long)

    sparse_adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    hop_masks = [torch.eye(num_nodes, dtype=torch.bool, device=edge_index.device)] # each one contains <= i hop masks
    hop_indicator = row.new_full((num_nodes, num_nodes), -1)
    hop_indicator[hop_masks[0]] = 0
    for i in range(num_hops):
        next_mask = sparse_adj.matmul(hop_masks[i].float()) > 0
        hop_masks.append(next_mask)
        hop_indicator[(hop_indicator==-1) & next_mask] = i+1
    hop_indicator = hop_indicator.T  # N x N
    node_mask = (hop_indicator >= 0) # N x N dense mask matrix
    return node_mask, hop_indicator


from torch_cluster import random_walk
def random_walk_subgraph(edge_index, num_nodes, walk_length, p=1, q=1, repeat=1, cal_hops=True, max_hops=10):
    """
        p (float, optional): Likelihood of immediately revisiting a node in the
            walk. (default: :obj:`1`)  Setting it to a high value (> max(q, 1)) ensures
            that we are less likely to sample an already visited node in the following two steps.
        q (float, optional): Control parameter to interpolate between
            breadth-first strategy and depth-first strategy (default: :obj:`1`)
            if q > 1, the random walk is biased towards nodes close to node t.
            if q < 1, the walk is more inclined to visit nodes which are further away from the node t.
        p, q ∈ {0.25, 0.50, 1, 2, 4}.
        Typical values:
        Fix p and tune q

        repeat: restart the random walk many times and combine together for the result

    """
    row, col = edge_index
    start = torch.arange(num_nodes, device=edge_index.device)
    walks = [random_walk(row, col,
                         start=start,
                         walk_length=walk_length,
                         p=p, q=q,
                         num_nodes=num_nodes) for _ in range(repeat)]
    walk = torch.cat(walks, dim=-1)
    node_mask = row.new_empty((num_nodes, num_nodes), dtype=torch.bool)
    # print(walk.shape)
    node_mask.fill_(False)
    node_mask[start.repeat_interleave((walk_length+1)*repeat), walk.reshape(-1)] = True
    if cal_hops: # this is fast enough
        sparse_adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
        hop_masks = [torch.eye(num_nodes, dtype=torch.bool, device=edge_index.device)]
        hop_indicator = row.new_full((num_nodes, num_nodes), -1)
        hop_indicator[hop_masks[0]] = 0
        for i in range(max_hops):
            next_mask = sparse_adj.matmul(hop_masks[i].float())>0
            hop_masks.append(next_mask)
            hop_indicator[(hop_indicator==-1) & next_mask] = i+1
            if hop_indicator[node_mask].min() != -1:
                break
        return node_mask, hop_indicator
    return node_mask, None

from torch_sparse import mul
def ppr_topk(
    edge_index,
    num_nodes,
    k=10,
    alpha=0.1,
    t=5,
    chunk_size=256,
    adaptive_k=False,
    min_k=4,
    max_k=None,
):
    """
        k: keep top-k nodes for each center node (includes itself)
        t: number of power iterations
        alpha: restart probability (teleport probability)
        chunk_size: number of source nodes processed per chunk
        adaptive_k: if True, use degree-dependent k per node
        min_k/max_k: clamp range for adaptive per-node k

        Notes:
        - This implementation avoids materializing full NxN identity/PPR at once.
        - Time complexity is still dominated by repeated sparse-dense matmul.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    device = edge_index.device
    sparse_adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))

    # Row-normalize sparse adjacency.
    deg = sparse_adj.sum(-1).to(torch.float)
    deg_inv = deg.pow(-1)
    deg_inv[torch.isinf(deg_inv)] = 0
    sparse_adj = mul(sparse_adj, deg_inv.view(-1, 1))

    if max_k is None:
        max_k = k
    if adaptive_k:
        local_k = torch.ceil(torch.sqrt(deg + 1.0)).to(torch.long)
        local_k = local_k.clamp(min=min_k, max=max_k)
        topk_global = int(local_k.max().item())
    else:
        local_k = None
        topk_global = k

    node_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)

    for start_idx in range(0, num_nodes, chunk_size):
        end_idx = min(start_idx + chunk_size, num_nodes)
        src = torch.arange(start_idx, end_idx, device=device)
        bsz = src.numel()

        seed = torch.zeros((num_nodes, bsz), dtype=torch.float, device=device)
        seed[src, torch.arange(bsz, device=device)] = 1.0

        ppr = seed
        for _ in range(t):
            ppr = (1 - alpha) * sparse_adj.matmul(ppr) + alpha * seed

        scores = ppr.t().contiguous()  # [bsz, num_nodes]

        # Ensure the source node is always included in top-k.
        row_idx = torch.arange(bsz, device=device)
        row_max = scores.max(dim=-1).values
        scores[row_idx, src] = row_max + 1e-12

        top_idx = torch.topk(scores, topk_global, dim=-1).indices

        if adaptive_k:
            for i in range(bsz):
                ki = int(local_k[src[i]].item())
                node_mask[src[i], top_idx[i, :ki]] = True
        else:
            node_mask[src.repeat_interleave(topk_global), top_idx.reshape(-1)] = True

    return node_mask, None
