import torch as th
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch
from torch import nn
from torch_geometric.nn import global_mean_pool, global_max_pool
from .layer.subgraph import subgraph
from .transform_AK.transform import SubgraphsTransform
from .transform_AK.config import cfg, update_cfg
import math
import logging
import math
from torch.distributions import Normal

from .transform_AK.element import MLP, DiscreteEncoder, Identity, VNUpdate


def glorot_orthogonal(tensor, scale):
    """Initialize a tensor's values according to an orthogonal Glorot initialization scheme."""
    if tensor is not None:
        th.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()

class MultiHeadAttentionLayer(nn.Module):
    """Compute attention scores with a DGLGraph's node and edge (geometric) features."""

    def __init__(self, num_input_feats, num_output_feats,
                 num_heads, using_bias=False, update_edge_feats=True, update_coords=False):
        super(MultiHeadAttentionLayer, self).__init__()

        # Declare shared variables
        self.num_output_feats = num_output_feats
        self.num_heads = num_heads
        self.using_bias = using_bias
        self.update_edge_feats = update_edge_feats
        self.update_coords = update_coords

        # Define node features' query, key, and value tensors, and define edge features' projection tensors
        self.Q = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.K = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.V = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.edge_feats_projection = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads,
                                               bias=using_bias)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        if self.using_bias:
            glorot_orthogonal(self.Q.weight, scale=scale)
            self.Q.bias.data.fill_(0)

            glorot_orthogonal(self.K.weight, scale=scale)
            self.K.bias.data.fill_(0)

            glorot_orthogonal(self.V.weight, scale=scale)
            self.V.bias.data.fill_(0)

            glorot_orthogonal(self.edge_feats_projection.weight, scale=scale)
            self.edge_feats_projection.bias.data.fill_(0)
        else:
            glorot_orthogonal(self.Q.weight, scale=scale)
            glorot_orthogonal(self.K.weight, scale=scale)
            glorot_orthogonal(self.V.weight, scale=scale)
            glorot_orthogonal(self.edge_feats_projection.weight, scale=scale)

    def propagate_attention(self, edge_index, node_feats_q, node_feats_k, node_feats_v, edge_feats_projection, coords,
                            return_attention=False):
        row, col = edge_index
        e_out = None
        attention = None
        # Compute attention scores
        alpha = node_feats_k[row] * node_feats_q[col]
        # Scale and clip attention scores
        alpha = (alpha / np.sqrt(self.num_output_feats)).clamp(-5.0, 5.0)
        # Use available edge features to modify the attention scores
        alpha = alpha * edge_feats_projection
        # Copy edge features as e_out to be passed to edge_feats_MLP
        if self.update_edge_feats:
            e_out = alpha

        # Apply softmax to attention scores, followed by clipping
        alphax = th.exp((alpha.sum(-1, keepdim=True)).clamp(-5.0, 5.0))
        # Send weighted values to target nodes
        wV = scatter_add(node_feats_v[row] * alphax, col, dim=0, dim_size=node_feats_q.size(0))
        z = scatter_add(alphax, col, dim=0, dim_size=node_feats_q.size(0))
        if return_attention:
            eps = th.full_like(alphax, 1e-6)
            edge_attention = alphax / (z[col] + eps)
            attention = {
                'edge_index': edge_index,
                'edge_attention': edge_attention.squeeze(-1),
                'edge_attention_logits': alpha.sum(-1),
            }
        return wV, z, e_out, coords, attention

    def forward(self, x, edge_attr, edge_index, coords, return_attention=False):
        row, col = edge_index
        node_feats_q = self.Q(x).view(-1, self.num_heads, self.num_output_feats)
        node_feats_k = self.K(x).view(-1, self.num_heads, self.num_output_feats)
        node_feats_v = self.V(x).view(-1, self.num_heads, self.num_output_feats)
        edge_feats_projection = self.edge_feats_projection(edge_attr).view(-1, self.num_heads, self.num_output_feats)
        wV, z, e_out, coords, attention = self.propagate_attention(
            edge_index,
            node_feats_q,
            node_feats_k,
            node_feats_v,
            edge_feats_projection,
            coords,
            return_attention=return_attention,
        )

        h_out = wV / (z + th.full_like(z, 1e-6))

        if return_attention:
            return h_out, e_out, coords, attention
        return h_out, e_out, coords


class GraphTransformerModule(nn.Module):
    """A Graph Transformer module (equivalent to one layer of graph convolutions)."""

    def __init__(
            self,
            num_hidden_channels,
            activ_fn=nn.SiLU(),
            residual=True,
            num_attention_heads=4,
            norm_to_apply='batch',
            dropout_rate=0.1,
            num_layers=4,
    ):
        super(GraphTransformerModule, self).__init__()

        # Record parameters given
        self.activ_fn = activ_fn
        self.residual = residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        # --------------------
        # Transformer Module
        # --------------------
        # Define all modules related to a Geometric Transformer module
        self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()

        self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels
        if self.apply_layer_norm:
            self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:  # Otherwise, default to using batch normalization
            self.batch_norm1_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm1_edge_feats = nn.BatchNorm1d(self.num_output_feats)

        self.mha_module = MultiHeadAttentionLayer(
            self.num_hidden_channels,
            self.num_output_feats // self.num_attention_heads,
            self.num_attention_heads,
            self.num_hidden_channels != self.num_output_feats,  # Only use bias if a Linear() has to change sizes
            update_edge_feats=True
        )

        self.O_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)
        self.O_edge_feats = nn.Linear(self.num_output_feats, self.num_output_feats)

        # MLP for node features
        dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        self.node_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

        if self.apply_layer_norm:
            self.layer_norm2_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm2_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:  # Otherwise, default to using batch normalization
            self.batch_norm2_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm2_edge_feats = nn.BatchNorm1d(self.num_output_feats)

        # MLP for edge features
        self.edge_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        glorot_orthogonal(self.O_node_feats.weight, scale=scale)
        self.O_node_feats.bias.data.fill_(0)
        glorot_orthogonal(self.O_edge_feats.weight, scale=scale)
        self.O_edge_feats.bias.data.fill_(0)

        for layer in self.node_feats_MLP:
            if hasattr(layer, 'weight'):  # Skip initialization for activation functions
                glorot_orthogonal(layer.weight, scale=scale)

        for layer in self.edge_feats_MLP:
            if hasattr(layer, 'weight'):
                glorot_orthogonal(layer.weight, scale=scale)

    def run_gt_layer(self, data, node_feats, edge_feats, return_attention=False):
        """Perform a forward pass of geometric attention using a multi-head attention (MHA) module."""
        node_feats_in1 = node_feats  # Cache node representations for first residual connection
        edge_feats_in1 = edge_feats  # Cache edge representations for first residual connection

        # Apply first round of normalization before applying geometric attention, for performance enhancement
        if self.apply_layer_norm:
            node_feats = self.layer_norm1_node_feats(node_feats)
            edge_feats = self.layer_norm1_edge_feats(edge_feats)
        else:  # Otherwise, default to using batch normalization
            node_feats = self.batch_norm1_node_feats(node_feats)
            edge_feats = self.batch_norm1_edge_feats(edge_feats)

        # Get multi-head attention output using provided node and edge representations
        if return_attention:
            node_attn_out, edge_attn_out, data.pos, attention = self.mha_module(
                node_feats,
                edge_feats,
                data.edge_index,
                data.pos,
                return_attention=True,
            )
        else:
            node_attn_out, edge_attn_out, data.pos = self.mha_module(node_feats, edge_feats, data.edge_index, data.pos)
            attention = None

        node_feats = node_attn_out.view(-1, self.num_output_feats)
        edge_feats = edge_attn_out.view(-1, self.num_output_feats)

        node_feats = F.dropout(node_feats, self.dropout_rate, training=self.training)
        edge_feats = F.dropout(edge_feats, self.dropout_rate, training=self.training)

        node_feats = self.O_node_feats(node_feats)
        edge_feats = self.O_edge_feats(edge_feats)

        # Make first residual connection
        if self.residual:
            node_feats = node_feats_in1 + node_feats  # Make first node residual connection
            edge_feats = edge_feats_in1 + edge_feats  # Make first edge residual connection

        node_feats_in2 = node_feats  # Cache node representations for second residual connection
        edge_feats_in2 = edge_feats  # Cache edge representations for second residual connection

        # Apply second round of normalization after first residual connection has been made
        if self.apply_layer_norm:
            node_feats = self.layer_norm2_node_feats(node_feats)
            edge_feats = self.layer_norm2_edge_feats(edge_feats)
        else:  # Otherwise, default to using batch normalization
            node_feats = self.batch_norm2_node_feats(node_feats)
            edge_feats = self.batch_norm2_edge_feats(edge_feats)

        # Apply MLPs for node and edge features
        for layer in self.node_feats_MLP:
            node_feats = layer(node_feats)
        for layer in self.edge_feats_MLP:
            edge_feats = layer(edge_feats)

        # Make second residual connection
        if self.residual:
            node_feats = node_feats_in2 + node_feats  # Make second node residual connection
            edge_feats = edge_feats_in2 + edge_feats  # Make second edge residual connection

        # Return edge representations along with node representations (for tasks other than interface prediction)
        if return_attention:
            return node_feats, edge_feats, attention
        return node_feats, edge_feats

    def forward(self, data, node_feats, edge_feats, return_attention=False):
        """Perform a forward pass of a Geometric Transformer to get intermediate node and edge representations."""
        if return_attention:
            node_feats, edge_feats, attention = self.run_gt_layer(data, node_feats, edge_feats, return_attention=True)
            return node_feats, edge_feats, attention
        node_feats, edge_feats = self.run_gt_layer(data, node_feats, edge_feats)
        return node_feats, edge_feats


class FinalGraphTransformerModule(nn.Module):
    """A (final layer) Graph Transformer module that combines node and edge representations using self-attention."""

    def __init__(self,
                 num_hidden_channels,
                 activ_fn=nn.SiLU(),
                 residual=True,
                 num_attention_heads=4,
                 norm_to_apply='batch',
                 dropout_rate=0.1,
                 num_layers=4):
        super(FinalGraphTransformerModule, self).__init__()

        # Record parameters given
        self.activ_fn = activ_fn
        self.residual = residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        # --------------------
        # Transformer Module
        # --------------------
        # Define all modules related to a Geometric Transformer module
        self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()

        self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels
        if self.apply_layer_norm:
            self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:  # Otherwise, default to using batch normalization
            self.batch_norm1_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm1_edge_feats = nn.BatchNorm1d(self.num_output_feats)

        self.mha_module = MultiHeadAttentionLayer(
            self.num_hidden_channels,
            self.num_output_feats // self.num_attention_heads,
            self.num_attention_heads,
            self.num_hidden_channels != self.num_output_feats,  # Only use bias if a Linear() has to change sizes
            update_edge_feats=False)

        self.O_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)

        # MLP for node features
        dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        self.node_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

        if self.apply_layer_norm:
            self.layer_norm2_node_feats = nn.LayerNorm(self.num_output_feats)
        else:  # Otherwise, default to using batch normalization
            self.batch_norm2_node_feats = nn.BatchNorm1d(self.num_output_feats)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        glorot_orthogonal(self.O_node_feats.weight, scale=scale)
        self.O_node_feats.bias.data.fill_(0)

        for layer in self.node_feats_MLP:
            if hasattr(layer, 'weight'):  # Skip initialization for activation functions
                glorot_orthogonal(layer.weight, scale=scale)

    # glorot_orthogonal(self.conformation_module.weight, scale=scale)

    def run_gt_layer(self, data, node_feats, edge_feats, return_attention=False):
        """Perform a forward pass of geometric attention using a multi-head attention (MHA) module."""
        node_feats_in1 = node_feats  # Cache node representations for first residual connection
        # edge_feats = self.conformation_module(edge_feats)

        # Apply first round of normalization before applying geometric attention, for performance enhancement
        if self.apply_layer_norm:
            node_feats = self.layer_norm1_node_feats(node_feats)
            edge_feats = self.layer_norm1_edge_feats(edge_feats)
        else:  # Otherwise, default to using batch normalization
            node_feats = self.batch_norm1_node_feats(node_feats)
            edge_feats = self.batch_norm1_edge_feats(edge_feats)

        # Get multi-head attention output using provided node and edge representations
        if return_attention:
            node_attn_out, _, data.pos, attention = self.mha_module(
                node_feats,
                edge_feats,
                data.edge_index,
                data.pos,
                return_attention=True,
            )
        else:
            node_attn_out, _, data.pos = self.mha_module(node_feats, edge_feats, data.edge_index, data.pos)
            attention = None
        node_feats = node_attn_out.view(-1, self.num_output_feats)
        node_feats = F.dropout(node_feats, self.dropout_rate, training=self.training)
        node_feats = self.O_node_feats(node_feats)

        # Make first residual connection
        if self.residual:
            node_feats = node_feats_in1 + node_feats  # Make first node residual connection

        node_feats_in2 = node_feats  # Cache node representations for second residual connection

        # Apply second round of normalization after first residual connection has been made
        if self.apply_layer_norm:
            node_feats = self.layer_norm2_node_feats(node_feats)
        else:  # Otherwise, default to using batch normalization
            node_feats = self.batch_norm2_node_feats(node_feats)

        # Apply MLP for node features
        for layer in self.node_feats_MLP:
            node_feats = layer(node_feats)

        # Make second residual connection
        if self.residual:
            node_feats = node_feats_in2 + node_feats  # Make second node residual connection

        # Return node representations
        if return_attention:
            return node_feats, attention
        return node_feats

    def forward(self, data, node_feats, edge_feats, return_attention=False):
        """Perform a forward pass of a Geometric Transformer to get final node representations."""
        if return_attention:
            node_feats, attention = self.run_gt_layer(data, node_feats, edge_feats, return_attention=True)
            return node_feats, attention
        node_feats = self.run_gt_layer(data, node_feats, edge_feats)
        return node_feats


class EvidentialClassificationHead(nn.Module):
    """Dirichlet evidential head for classification.

    Returns logits by default for backward compatibility.
    Set return_dict=True to also get probabilities and uncertainty.
    """

    def __init__(self, in_dim, hidden_dim, num_classes=2, dropout_rate=0.1):
        super(EvidentialClassificationHead, self).__init__()
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, return_dict=False):
        logits = self.mlp(x)
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        s = th.sum(alpha, dim=-1, keepdim=True)
        probs = alpha / s
        uncertainty = float(self.num_classes) / s.squeeze(-1)

        if return_dict:
            return {
                'logits': logits,
                'evidence': evidence,
                'alpha': alpha,
                'probs': probs,
                'uncertainty': uncertainty,
            }
        return logits

class GraphTransformer(nn.Module):
    """A graph transformer
	"""

    def __init__(
            self,
            in_channels,
            edge_features=10,
            num_hidden_channels=128,
            activ_fn=nn.SiLU(),
            transformer_residual=True,
            num_attention_heads=4,
            norm_to_apply='batch',
            dropout_rate=0.1,
            num_layers=4,
                use_evidential=False,
            **kwargs
    ):
        super(GraphTransformer, self).__init__()

        # Initialize model parameters
        self.activ_fn = activ_fn
        self.transformer_residual = transformer_residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.use_evidential = use_evidential

        # --------------------
        # Initializer Modules
        # --------------------
        # Define all modules related to edge and node initialization
        self.node_encoder = nn.Linear(in_channels, num_hidden_channels)
        self.edge_encoder = nn.Linear(edge_features, num_hidden_channels)
        # --------------------
        # Transformer Module
        # --------------------
        # Define all modules related to a variable number of Geometric Transformer modules
        num_intermediate_layers = max(0, num_layers - 1)
        gt_block_modules = [GraphTransformerModule(
            num_hidden_channels=num_hidden_channels,
            activ_fn=activ_fn,
            residual=transformer_residual,
            num_attention_heads=num_attention_heads,
            norm_to_apply=norm_to_apply,
            dropout_rate=dropout_rate,
            num_layers=num_layers) for _ in range(num_intermediate_layers)]
        if num_layers > 0:
            gt_block_modules.extend([FinalGraphTransformerModule(
                num_hidden_channels=num_hidden_channels,
                activ_fn=activ_fn,
                residual=transformer_residual,
                num_attention_heads=num_attention_heads,
                norm_to_apply=norm_to_apply,
                dropout_rate=dropout_rate,
                num_layers=num_layers)])
        self.gt_block = nn.ModuleList(gt_block_modules)

        self.readout_layer = nn.Sequential(
            nn.Linear(num_hidden_channels, num_hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(num_hidden_channels // 2, 2),
        )
        self.evidential_head = EvidentialClassificationHead(
            in_dim=num_hidden_channels,
            hidden_dim=num_hidden_channels // 2,
            num_classes=2,
            dropout_rate=dropout_rate,
        )

    def forward(self, data, return_attention=False, return_evidential=False):
        node_feats = self.node_encoder(data.x)
        edge_feats = self.edge_encoder(data.edge_attr)
        attentions = []

        # Apply a given number of intermediate geometric attention layers to the node and edge features given
        for gt_layer in self.gt_block[:-1]:
            if return_attention:
                node_feats, edge_feats, attention = gt_layer(data, node_feats, edge_feats, return_attention=True)
                attentions.append(attention)
            else:
                node_feats, edge_feats = gt_layer(data, node_feats, edge_feats)

        # Apply final layer to update node representations by merging current node and edge representations
        if return_attention:
            node_feats, attention = self.gt_block[-1](data, node_feats, edge_feats, return_attention=True)
            attentions.append(attention)
        else:
            node_feats = self.gt_block[-1](data, node_feats, edge_feats)

        prop = global_mean_pool(node_feats, data.batch)

        if self.use_evidential:
            cls = self.evidential_head(prop, return_dict=return_evidential)
        else:
            cls = self.readout_layer(prop)

        if return_attention:
            return cls, attentions
        return cls


class MultiScaleGraphTransformer(nn.Module):
    """Two-scale model: atom-level graph transformer + residue-level self-attention + cross-scale write-back.

    Expected optional field on `data`:
    - `atom_residue_index`: LongTensor[num_atoms], residue id local to each graph.
      If absent, each atom is treated as an individual residue.
    """

    def __init__(
            self,
            in_channels,
            edge_features=10,
            num_hidden_channels=128,
            activ_fn=nn.SiLU(),
            transformer_residual=True,
            num_attention_heads=4,
            norm_to_apply='batch',
            dropout_rate=0.1,
            num_layers=4,
            num_residue_layers=2,
                use_evidential=False,
            **kwargs
    ):
        super(MultiScaleGraphTransformer, self).__init__()

        self.activ_fn = activ_fn
        self.transformer_residual = transformer_residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.num_residue_layers = num_residue_layers
        self.use_evidential = use_evidential

        self.node_encoder = nn.Linear(in_channels, num_hidden_channels)
        self.edge_encoder = nn.Linear(edge_features, num_hidden_channels)

        num_intermediate_layers = max(0, num_layers - 1)
        gt_block_modules = [GraphTransformerModule(
            num_hidden_channels=num_hidden_channels,
            activ_fn=activ_fn,
            residual=transformer_residual,
            num_attention_heads=num_attention_heads,
            norm_to_apply=norm_to_apply,
            dropout_rate=dropout_rate,
            num_layers=num_layers) for _ in range(num_intermediate_layers)]
        if num_layers > 0:
            gt_block_modules.extend([FinalGraphTransformerModule(
                num_hidden_channels=num_hidden_channels,
                activ_fn=activ_fn,
                residual=transformer_residual,
                num_attention_heads=num_attention_heads,
                norm_to_apply=norm_to_apply,
                dropout_rate=dropout_rate,
                num_layers=num_layers)])
        self.gt_block = nn.ModuleList(gt_block_modules)

        self.residue_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=num_hidden_channels,
                num_heads=num_attention_heads,
                dropout=dropout_rate,
                batch_first=True,
            )
            for _ in range(num_residue_layers)
        ])
        self.residue_norm1 = nn.ModuleList([nn.LayerNorm(num_hidden_channels) for _ in range(num_residue_layers)])
        self.residue_norm2 = nn.ModuleList([nn.LayerNorm(num_hidden_channels) for _ in range(num_residue_layers)])
        self.residue_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_hidden_channels, num_hidden_channels * 2),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(num_hidden_channels * 2, num_hidden_channels),
            )
            for _ in range(num_residue_layers)
        ])

        # Topology-aware explicit attention bias on residue layers.
        # edge_type: 0=none, 1=residue adjacency (covalent path), 2=disulfide-like bridge.
        self.topology_distance_scale = nn.Parameter(th.tensor(0.3, dtype=th.float))
        self.topology_edge_bias = nn.Embedding(3, 1)
        nn.init.zeros_(self.topology_edge_bias.weight)

        self.atom_from_residue = nn.Linear(num_hidden_channels, num_hidden_channels)
        # Gate now sees atom feat, residue message, and relative geometry (dx,dy,dz,||d||).
        self.cross_scale_gate = nn.Linear(num_hidden_channels * 2 + 4, num_hidden_channels)
        self.cross_scale_norm = nn.LayerNorm(num_hidden_channels)
        self.readout_layer = nn.Sequential(
            nn.Linear(num_hidden_channels * 4, num_hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_hidden_channels, 2),
        )
        self.evidential_head = EvidentialClassificationHead(
            in_dim=num_hidden_channels * 4,
            hidden_dim=num_hidden_channels,
            num_classes=2,
            dropout_rate=dropout_rate,
        )

    def _build_residue_index(self, data, batch):
        if hasattr(data, 'atom_residue_index') and data.atom_residue_index is not None:
            atom_residue_index = data.atom_residue_index.long()
        else:
            atom_residue_index = th.arange(batch.size(0), device=batch.device, dtype=th.long)

        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        offsets = []
        running = 0
        for gid in range(num_graphs):
            mask = batch == gid
            if mask.any():
                local_max = int(atom_residue_index[mask].max().item()) + 1
            else:
                local_max = 0
            offsets.append(running)
            running += local_max

        offsets = th.tensor(offsets, device=batch.device, dtype=th.long)
        global_residue_ids = atom_residue_index + offsets[batch]
        unique_ids, atom_to_residue = th.unique(global_residue_ids, sorted=True, return_inverse=True)

        residue_batch = th.zeros(unique_ids.size(0), device=batch.device, dtype=batch.dtype)
        residue_batch[atom_to_residue] = batch
        return atom_to_residue, residue_batch

    def _atom_to_residue_pool(self, atom_feats, atom_to_residue):
        num_residues = int(atom_to_residue.max().item()) + 1 if atom_to_residue.numel() > 0 else 0
        residue_sum = scatter_add(atom_feats, atom_to_residue, dim=0, dim_size=num_residues)
        ones = th.ones((atom_feats.size(0), 1), dtype=atom_feats.dtype, device=atom_feats.device)
        residue_cnt = scatter_add(ones, atom_to_residue, dim=0, dim_size=num_residues).clamp_min(1.0)
        residue_feats = residue_sum / residue_cnt
        return residue_feats

    def _relative_atom_residue_geometry(self, data, atom_to_residue):
        # Build per-atom relative coordinates to residue centroid: [dx, dy, dz, ||d||].
        if hasattr(data, 'pos') and data.pos is not None and data.pos.dim() == 2 and data.pos.size(1) >= 3:
            pos = data.pos[:, :3]
        else:
            pos = th.zeros((atom_to_residue.size(0), 3), dtype=th.float, device=atom_to_residue.device)

        pos = pos.to(dtype=th.float, device=atom_to_residue.device)
        num_residues = int(atom_to_residue.max().item()) + 1 if atom_to_residue.numel() > 0 else 0

        centroid_sum = scatter_add(pos, atom_to_residue, dim=0, dim_size=num_residues)
        ones = th.ones((pos.size(0), 1), dtype=pos.dtype, device=pos.device)
        centroid_cnt = scatter_add(ones, atom_to_residue, dim=0, dim_size=num_residues).clamp_min(1.0)
        centroid = centroid_sum / centroid_cnt

        rel = pos - centroid[atom_to_residue]
        rel_norm = th.norm(rel, dim=-1, keepdim=True)
        return th.cat([rel, rel_norm], dim=-1)

    def _build_residue_topology(self, data, atom_to_residue, residue_batch):
        num_residues = int(residue_batch.size(0))
        if num_residues == 0:
            return (
                th.zeros((0, 0), dtype=th.float, device=atom_to_residue.device),
                th.zeros((0, 0), dtype=th.long, device=atom_to_residue.device),
            )

        dist = th.full((num_residues, num_residues), float('inf'), dtype=th.float, device=atom_to_residue.device)
        dist[th.arange(num_residues), th.arange(num_residues)] = 0.0
        edge_type = th.zeros((num_residues, num_residues), dtype=th.long, device=atom_to_residue.device)

        row, col = data.edge_index
        res_u = atom_to_residue[row]
        res_v = atom_to_residue[col]
        cross = res_u != res_v
        if cross.any():
            ru = res_u[cross]
            rv = res_v[cross]
            dist[ru, rv] = 1.0
            dist[rv, ru] = 1.0

            base_type = th.ones_like(ru, dtype=th.long)
            if hasattr(data, 'atom_z') and data.atom_z is not None:
                z_u = data.atom_z[row[cross]].long()
                z_v = data.atom_z[col[cross]].long()
                disulfide_like = (z_u == 16) & (z_v == 16)
                base_type[disulfide_like] = 2

            edge_type[ru, rv] = th.maximum(edge_type[ru, rv], base_type)
            edge_type[rv, ru] = th.maximum(edge_type[rv, ru], base_type)

        # Per-graph shortest path distance on residue graph.
        num_graphs = int(residue_batch.max().item()) + 1 if residue_batch.numel() > 0 else 1
        for gid in range(num_graphs):
            nodes = th.where(residue_batch == gid)[0]
            n = int(nodes.numel())
            if n <= 1:
                continue

            # Build adjacency list for this graph.
            global_to_local = {int(nodes[i].item()): i for i in range(n)}
            adj = [[] for _ in range(n)]
            sub_edge = edge_type[nodes][:, nodes]
            e_row, e_col = th.where(sub_edge > 0)
            for i, j in zip(e_row.tolist(), e_col.tolist()):
                if i != j:
                    adj[i].append(j)

            # BFS from each source residue.
            local_dist = th.full((n, n), float('inf'), dtype=th.float, device=dist.device)
            local_dist[th.arange(n), th.arange(n)] = 0.0
            for src in range(n):
                q = [src]
                head = 0
                while head < len(q):
                    u = q[head]
                    head += 1
                    du = float(local_dist[src, u].item())
                    for v in adj[u]:
                        if local_dist[src, v] == float('inf'):
                            local_dist[src, v] = du + 1.0
                            q.append(v)

            # Unreachable pairs get a capped penalty distance.
            max_reachable = float(th.max(local_dist[local_dist < float('inf')]).item()) if (local_dist < float('inf')).any() else 1.0
            fill_value = max_reachable + 1.0
            local_dist = th.where(local_dist == float('inf'), th.full_like(local_dist, fill_value), local_dist)
            dist[nodes[:, None], nodes[None, :]] = local_dist

        return dist, edge_type

    def _build_topology_attn_mask(self, residue_ids_dense, mask, residue_dist, residue_edge_type):
        bsz, max_len = residue_ids_dense.size()
        dense_bias = th.zeros((bsz, max_len, max_len), dtype=residue_dist.dtype, device=residue_dist.device)

        for b in range(bsz):
            valid = mask[b]
            n = int(valid.sum().item())
            if n == 0:
                continue
            ids = residue_ids_dense[b, valid].long()
            d = residue_dist[ids][:, ids]
            et = residue_edge_type[ids][:, ids]
            topo_bias = -self.topology_distance_scale * th.log1p(d.clamp_min(0.0))
            topo_bias = topo_bias + self.topology_edge_bias(et).squeeze(-1)
            dense_bias[b, :n, :n] = topo_bias

        # MultiheadAttention expects [B*num_heads, L, L] for per-sample float masks.
        return dense_bias.repeat_interleave(self.num_attention_heads, dim=0)

    def _run_atom_encoder(self, data, return_attention=False):
        node_feats = self.node_encoder(data.x)
        edge_feats = self.edge_encoder(data.edge_attr)
        atom_attentions = []

        for gt_layer in self.gt_block[:-1]:
            if return_attention:
                node_feats, edge_feats, attention = gt_layer(data, node_feats, edge_feats, return_attention=True)
                atom_attentions.append(attention)
            else:
                node_feats, edge_feats = gt_layer(data, node_feats, edge_feats)

        if return_attention:
            node_feats, attention = self.gt_block[-1](data, node_feats, edge_feats, return_attention=True)
            atom_attentions.append(attention)
            return node_feats, atom_attentions

        node_feats = self.gt_block[-1](data, node_feats, edge_feats)
        return node_feats, atom_attentions

    def _run_residue_encoder(self, residue_feats, residue_batch, residue_dist=None, residue_edge_type=None, return_attention=False):
        dense_residue, mask = to_dense_batch(residue_feats, residue_batch)
        key_padding_mask = ~mask
        residue_ids = th.arange(residue_feats.size(0), dtype=th.long, device=residue_feats.device)
        residue_ids_dense, _ = to_dense_batch(residue_ids, residue_batch)

        attn_mask = None
        if residue_dist is not None and residue_edge_type is not None:
            attn_mask = self._build_topology_attn_mask(
                residue_ids_dense=residue_ids_dense,
                mask=mask,
                residue_dist=residue_dist,
                residue_edge_type=residue_edge_type,
            ).to(dtype=dense_residue.dtype)
        residue_attentions = []

        for attn_layer, norm1, ffn, norm2 in zip(
                self.residue_attn_layers,
                self.residue_norm1,
                self.residue_ffn,
                self.residue_norm2,
        ):
            if return_attention:
                attn_out, attn_weights = attn_layer(
                    dense_residue,
                    dense_residue,
                    dense_residue,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=True,
                    average_attn_weights=False,
                )
                residue_attentions.append(attn_weights)
            else:
                attn_out, _ = attn_layer(
                    dense_residue,
                    dense_residue,
                    dense_residue,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )

            dense_residue = norm1(dense_residue + F.dropout(attn_out, self.dropout_rate, training=self.training))
            ffn_out = ffn(dense_residue)
            dense_residue = norm2(dense_residue + F.dropout(ffn_out, self.dropout_rate, training=self.training))

        residue_context = dense_residue[mask]
        return residue_context, residue_attentions

    def forward(self, data, return_attention=False, return_evidential=False):
        if not hasattr(data, 'pos') or data.pos is None:
            data.pos = th.zeros((data.x.size(0), 3), dtype=data.x.dtype, device=data.x.device)

        atom_feats, atom_attentions = self._run_atom_encoder(data, return_attention=return_attention)
        atom_to_residue, residue_batch = self._build_residue_index(data, data.batch)
        residue_feats = self._atom_to_residue_pool(atom_feats, atom_to_residue)
        residue_dist, residue_edge_type = self._build_residue_topology(data, atom_to_residue, residue_batch)
        residue_context, residue_attentions = self._run_residue_encoder(
            residue_feats,
            residue_batch,
            residue_dist=residue_dist,
            residue_edge_type=residue_edge_type,
            return_attention=return_attention,
        )

        residue_message = self.atom_from_residue(residue_context[atom_to_residue])
        rel_geo = self._relative_atom_residue_geometry(data, atom_to_residue)
        cross_scale_gate = th.sigmoid(self.cross_scale_gate(th.cat([atom_feats, residue_message, rel_geo], dim=-1)))
        atom_feats = self.cross_scale_norm(atom_feats + cross_scale_gate * residue_message)

        atom_graph = th.cat([
            global_mean_pool(atom_feats, data.batch),
            global_max_pool(atom_feats, data.batch),
        ], dim=-1)
        residue_graph = th.cat([
            global_mean_pool(residue_context, residue_batch),
            global_max_pool(residue_context, residue_batch),
        ], dim=-1)
        graph_repr = th.cat([atom_graph, residue_graph], dim=-1)
        if self.use_evidential:
            cls = self.evidential_head(graph_repr, return_dict=return_evidential)
        else:
            cls = self.readout_layer(graph_repr)

        if return_attention:
            attn = {
                'atom_attentions': atom_attentions,
                'residue_attentions': residue_attentions,
                'atom_to_residue': atom_to_residue,
            }
            return cls, attn
        return cls


class SubGT(nn.Module):
    def __init__(self,
                 in_channels,
                 edge_features=10,
                 num_hidden_channels=128,
                 activ_fn=nn.SiLU(),
                 transformer_residual=True,
                 num_attention_heads=4,
                 norm_to_apply='batch',
                 dropout_rate=0.1,
                 num_layers=4,
                 **kwargs
                 ):
        super(SubGT, self).__init__()

        # Initialize model parameters
        self.activ_fn = activ_fn
        self.transformer_residual = transformer_residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        # --------------------
        # Initializer Modules
        # --------------------
        # Define all modules related to edge and node initialization
        self.node_encoder = nn.Linear(in_channels, num_hidden_channels)
        self.edge_encoder = nn.Linear(edge_features, num_hidden_channels)
        # --------------------
        # Transformer Module
        # --------------------
        # Define all modules related to a variable number of Geometric Transformer modules
        num_intermediate_layers = max(0, num_layers - 1)
        gt_block_modules = [GraphTransformerModule(
            num_hidden_channels=num_hidden_channels,
            activ_fn=activ_fn,
            residual=transformer_residual,
            num_attention_heads=num_attention_heads,
            norm_to_apply=norm_to_apply,
            dropout_rate=dropout_rate,
            num_layers=num_layers) for _ in range(num_intermediate_layers)]
        if num_layers > 0:
            gt_block_modules.extend([FinalGraphTransformerModule(
                num_hidden_channels=num_hidden_channels,
                activ_fn=activ_fn,
                residual=transformer_residual,
                num_attention_heads=num_attention_heads,
                norm_to_apply=norm_to_apply,
                dropout_rate=dropout_rate,
                num_layers=num_layers)])
        self.gt_block = nn.ModuleList(gt_block_modules)

        self.transform = SubgraphsTransform(cfg.subgraph.hops,
                                            walk_length=cfg.subgraph.walk_length,
                                            p=cfg.subgraph.walk_p,
                                            q=cfg.subgraph.walk_q,
                                            repeat=cfg.subgraph.walk_repeat,
                                            sampling_mode=cfg.sampling.mode,
                                            minimum_redundancy=cfg.sampling.redundancy,
                                            shortest_path_mode_stride=cfg.sampling.stride,
                                            random_mode_sampling_rate=cfg.sampling.random_rate,
                                            random_init=True)

        self.transform_eval = SubgraphsTransform(cfg.subgraph.hops,
                                                 walk_length=cfg.subgraph.walk_length,
                                                 p=cfg.subgraph.walk_p,
                                                 q=cfg.subgraph.walk_q,
                                                 repeat=cfg.subgraph.walk_repeat,
                                                 sampling_mode=None,
                                                 random_init=False)

        self.sub_GT = subgraph(self.gt_block, num_hidden_channels)

        self.readout_layer = nn.Sequential(
            nn.Linear(num_hidden_channels, num_hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(num_hidden_channels // 2, 2),
        )

    def forward(self, data, model_type=None):
        edge_index = data.edge_index
        data.x = self.node_encoder(data.x)
        data.edge_attr = self.edge_encoder(data.edge_attr)

        sample = dict()

        sample['subgraphs_batch'], sample['subgraphs_nodes_mapper'], sample['subgraphs_edges_mapper'], sample[
            'combined_subgraphs'], sample['hop_indicator'], sample['num_nodes'] = self.transform_eval(data.edge_index,
                                                                                                      data.x.size()[0])

        # Apply a given number of intermediate geometric attention layers to the node and edge features given

        node_feats = self.sub_GT(data, sample, model_type)

        prop = global_mean_pool(node_feats, data.batch)

        cls = self.readout_layer(prop)

        return cls



