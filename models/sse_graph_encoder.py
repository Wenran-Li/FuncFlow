import torch
import torch.nn as nn


class SSEGraphLayer(nn.Module):
    """
    Simple invariant message-passing layer over padded SSE graphs.

    Inputs
    ------
    h : [B, K, D]
        SSE node embeddings.

    edge_feat : [B, K, K, De]
        Pair / edge embeddings.

    edge_mask : [B, K, K]
        1 where an actual graph edge exists.

    node_mask : [B, K]
        1 for real SSE nodes, 0 for padding.
    """

    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim,
    ):
        super().__init__()

        self.message_mlp = nn.Sequential(
            nn.Linear(
                2 * node_dim + edge_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                node_dim,
            ),
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(
                2 * node_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                node_dim,
            ),
        )

        self.norm = nn.LayerNorm(
            node_dim
        )

    def forward(
        self,
        h,
        edge_feat,
        edge_mask,
        node_mask,
    ):
        B, K, D = h.shape

        # ----------------------------------------------------
        # Build h_i and h_j for every ordered pair.
        # ----------------------------------------------------

        h_i = h[:, :, None, :].expand(
            B,
            K,
            K,
            D,
        )

        h_j = h[:, None, :, :].expand(
            B,
            K,
            K,
            D,
        )

        # ----------------------------------------------------
        # Message input:
        #
        # [h_i, h_j, edge_ij]
        # ----------------------------------------------------

        msg_input = torch.cat(
            [
                h_i,
                h_j,
                edge_feat,
            ],
            dim=-1,
        )

        messages = self.message_mlp(
            msg_input
        )

        # ----------------------------------------------------
        # Mask nonexistent graph edges.
        # ----------------------------------------------------

        edge_mask_f = (
            edge_mask
            .unsqueeze(-1)
            .to(messages.dtype)
        )

        messages = (
            messages
            * edge_mask_f
        )

        # ----------------------------------------------------
        # Aggregate incoming messages over j.
        # ----------------------------------------------------

        denom = (
            edge_mask_f.sum(
                dim=2
            )
            .clamp(min=1.0)
        )

        agg = (
            messages.sum(
                dim=2
            )
            / denom
        )

        # ----------------------------------------------------
        # Node update
        # ----------------------------------------------------

        update_input = torch.cat(
            [
                h,
                agg,
            ],
            dim=-1,
        )

        delta = self.update_mlp(
            update_input
        )

        h_new = self.norm(
            h + delta
        )

        # ----------------------------------------------------
        # Zero padded nodes
        # ----------------------------------------------------

        h_new = (
            h_new
            * node_mask
            .unsqueeze(-1)
            .to(h_new.dtype)
        )

        return h_new


class SSEGraphEncoder(nn.Module):
    """
    Encode padded SSE graph features.

    Node inputs
    -----------
    sse_type:
        0 = H
        1 = E
        2 = PAD

    sse_length:
        integer SSE length

    linker_before:
        number of residues before current SSE

    Edge inputs
    -----------
    sse_edge_type:
        0 = no edge
        1 = sequence edge
        2 = spatial edge

    sse_orientation:
        0 = other
        1 = parallel
        2 = antiparallel
        3 = unknown
        4 = PAD/no edge
    """

    def __init__(
        self,
        node_dim=64,
        edge_dim=32,
        hidden_dim=128,
        num_layers=2,
        max_sse_length=512,
        max_linker_length=128,
    ):
        super().__init__()

        self.node_dim = node_dim

        # ----------------------------------------------------
        # Node embeddings
        # ----------------------------------------------------

        self.type_embedding = nn.Embedding(
            3,
            node_dim,
        )

        self.length_embedding = nn.Embedding(
            max_sse_length + 1,
            node_dim,
        )

        self.linker_embedding = nn.Embedding(
            max_linker_length + 1,
            node_dim,
        )

        # ----------------------------------------------------
        # Edge embeddings
        # ----------------------------------------------------

        self.edge_type_embedding = nn.Embedding(
            3,
            edge_dim,
        )

        self.orientation_embedding = nn.Embedding(
            5,
            edge_dim,
        )

        self.edge_projection = nn.Linear(
            2 * edge_dim,
            edge_dim,
        )

        # ----------------------------------------------------
        # Message-passing layers
        # ----------------------------------------------------

        self.layers = nn.ModuleList(
            [
                SSEGraphLayer(
                    node_dim=node_dim,
                    edge_dim=edge_dim,
                    hidden_dim=hidden_dim,
                )
                for _ in range(
                    num_layers
                )
            ]
        )

    def forward(
        self,
        sse_type,
        sse_length,
        linker_before,
        sse_mask,
        sse_edge_type,
        sse_orientation,
    ):
        """
        Returns
        -------
        h : [B, K, node_dim]
            Encoded SSE-node representations.
        """

        # ----------------------------------------------------
        # Clip integer features to embedding ranges.
        # ----------------------------------------------------

        sse_length = torch.clamp(
            sse_length,
            min=0,
            max=self.length_embedding.num_embeddings - 1,
        )

        linker_before = torch.clamp(
            linker_before,
            min=0,
            max=self.linker_embedding.num_embeddings - 1,
        )

        # ----------------------------------------------------
        # Initial node representation
        # ----------------------------------------------------

        h = (
            self.type_embedding(
                sse_type
            )
            +
            self.length_embedding(
                sse_length
            )
            +
            self.linker_embedding(
                linker_before
            )
        )

        h = (
            h
            * sse_mask
            .unsqueeze(-1)
            .to(h.dtype)
        )

        # ----------------------------------------------------
        # Edge representation
        # ----------------------------------------------------

        edge_type_emb = (
            self.edge_type_embedding(
                sse_edge_type
            )
        )

        orientation_emb = (
            self.orientation_embedding(
                sse_orientation
            )
        )

        edge_feat = self.edge_projection(
            torch.cat(
                [
                    edge_type_emb,
                    orientation_emb,
                ],
                dim=-1,
            )
        )

        # Actual graph edge:
        #
        # edge_type 1 or 2
        edge_mask = (
            sse_edge_type > 0
        )

        # ----------------------------------------------------
        # Message passing
        # ----------------------------------------------------

        for layer in self.layers:

            h = layer(
                h,
                edge_feat,
                edge_mask,
                sse_mask,
            )

        return h


def broadcast_sse_to_residues(
    sse_repr,
    residue_to_sse,
):
    """
    Broadcast SSE-node representations to residues.

    Parameters
    ----------
    sse_repr : [B, K, D]

    residue_to_sse : [B, L]
        -1 for coil residues
        0..K-1 for H/E residues.

    Returns
    -------
    residue_repr : [B, L, D]

    residue_sse_mask : [B, L]
        1 where residue belongs to an SSE,
        0 for coil / unassigned residue.
    """

    B, K, D = sse_repr.shape

    mapping = residue_to_sse

    residue_sse_mask = (
        mapping >= 0
    )

    # Avoid negative gather indices.
    safe_mapping = torch.clamp(
        mapping,
        min=0,
        max=K - 1,
    )

    gather_idx = (
        safe_mapping
        .unsqueeze(-1)
        .expand(
            -1,
            -1,
            D,
        )
    )

    residue_repr = torch.gather(
        sse_repr,
        dim=1,
        index=gather_idx,
    )

    residue_repr = (
        residue_repr
        * residue_sse_mask
        .unsqueeze(-1)
        .to(residue_repr.dtype)
    )

    return (
        residue_repr,
        residue_sse_mask,
    )
