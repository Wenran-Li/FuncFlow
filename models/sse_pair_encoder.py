import torch
import torch.nn as nn
import os

class SSEPairEncoder(nn.Module):
    """
    Encode SSE-level pair relations and broadcast them to residue pairs.

    Inputs
    ------
    sse_edge_type:
        [B, K, K]
        0 = none
        1 = sequential
        2 = spatial

    sse_orientation:
        [B, K, K]
        0 = other
        1 = parallel
        2 = antiparallel
        3 = unknown
        4 = PAD / no-edge

    residue_to_sse:
        [B, L]
        SSE index for H/E residues.
        -1 for coil residues.

    sse_mask:
        [B, K]

    Output
    ------
    residue_pair_emb:
        [B, L, L, pair_dim]
    """

    def __init__(
        self,
        pair_dim=32,
        edge_emb_dim=16,
        orientation_emb_dim=16,
    ):
        super().__init__()

        # 0 none, 1 sequential, 2 spatial
        self.edge_embedding = nn.Embedding(
            3,
            edge_emb_dim,
        )

        # 0 other
        # 1 parallel
        # 2 antiparallel
        # 3 unknown
        # 4 PAD / no-edge
        self.orientation_embedding = nn.Embedding(
            5,
            orientation_emb_dim,
        )

        self.proj = nn.Sequential(
            nn.Linear(
                edge_emb_dim + orientation_emb_dim,
                pair_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                pair_dim,
                pair_dim,
            ),
        )

    def _shuffle_pair_relations(
        self,
        sse_edge_type,
        sse_orientation,
        sse_mask,
    ):
        """Deterministically permute SSE identities for pair relations only."""

        edge = sse_edge_type.clone()
        orient = sse_orientation.clone()

        B = edge.shape[0]

        for b in range(B):
            valid_idx = torch.where(
                sse_mask[b].bool()
            )[0]

            K = valid_idx.numel()

            if K <= 1:
                continue

            # Deterministic pseudo-random permutation.
            # Fixed across every reverse-flow step.
            g = torch.Generator(device="cpu")
            g.manual_seed(1000 + K)

            perm_local = torch.randperm(
                K,
                generator=g,
            ).to(valid_idx.device)

            source_idx = valid_idx[perm_local]

            old_edge = sse_edge_type[b].clone()
            old_orient = sse_orientation[b].clone()

            edge[b][
                valid_idx[:, None],
                valid_idx[None, :]
            ] = old_edge[
                source_idx[:, None],
                source_idx[None, :]
            ]

            orient[b][
                valid_idx[:, None],
                valid_idx[None, :]
            ] = old_orient[
                source_idx[:, None],
                source_idx[None, :]
            ]

        return edge, orient

    def forward(
        self,
        sse_edge_type,
        sse_orientation,
        residue_to_sse,
        sse_mask,
        res_mask=None,
    ):
        """
        Returns [B, L, L, pair_dim].
        """

        B, K, _ = sse_edge_type.shape
        _, L = residue_to_sse.shape

        # -----------------------------------------------
        # 1. SSE-pair embeddings
        # -----------------------------------------------

        # FlowModel performs the SSE pair permutation.
        # The encoder must not shuffle the same input again.
        # Retain pair_mode for the existing 'none' masking branch.
        pair_mode = os.environ.get(
            "SSE_PAIR_MODE", "correct"
        ).lower()

        if pair_mode not in {"correct", "shuffled", "none"}:
            raise ValueError(
                f"Unknown SSE_PAIR_MODE={pair_mode}"
            )

        edge_emb = self.edge_embedding(
            sse_edge_type.long()
        )

        orient_emb = self.orientation_embedding(
            sse_orientation.long()
        )

        sse_pair = torch.cat(
            [edge_emb, orient_emb],
            dim=-1,
        )

        sse_pair = self.proj(
            sse_pair
        )
        # [B, K, K, pair_dim]

        # -----------------------------------------------
        # 2. Mask invalid SSE pairs
        # -----------------------------------------------

        pair_mask = (
            sse_mask[:, :, None]
            * sse_mask[:, None, :]
        ).to(sse_pair.dtype)

        sse_pair = (
            sse_pair
            * pair_mask[..., None]
        )

        # -----------------------------------------------
        # 3. Broadcast SSE pair -> residue pair
        #
        # residue_to_sse:
        #   -1 for coil
        #
        # Clamp only for indexing, then explicitly mask.
        # -----------------------------------------------

        valid_res_sse = (
            residue_to_sse >= 0
        )

        idx = residue_to_sse.clamp(
            min=0,
            max=K - 1,
        ).long()

        # First gather SSE index for residue i.
        #
        # sse_pair: [B,K,K,D]
        # idx:      [B,L]
        #
        # -> [B,L,K,D]

        D = sse_pair.shape[-1]

        idx_i = idx[
            :, :, None, None
        ].expand(
            B,
            L,
            K,
            D,
        )

        residue_i_sse_pairs = torch.gather(
            sse_pair,
            dim=1,
            index=idx_i,
        )

        # Now gather SSE index for residue j.
        #
        # -> [B,L,L,D]

        idx_j = idx[
            :, None, :, None
        ].expand(
            B,
            L,
            L,
            D,
        )

        residue_pair = torch.gather(
            residue_i_sse_pairs,
            dim=2,
            index=idx_j,
        )

        # -----------------------------------------------
        # 4. Mask coil / invalid residues
        # -----------------------------------------------

        valid_pair = (
            valid_res_sse[:, :, None]
            & valid_res_sse[:, None, :]
        )

        if res_mask is not None:
            valid_pair = (
                valid_pair
                & res_mask[:, :, None].bool()
                & res_mask[:, None, :].bool()
            )

        residue_pair = (
            residue_pair
            * valid_pair[..., None].to(
                residue_pair.dtype
            )
        )

        if pair_mode == "none":
            residue_pair = torch.zeros_like(
                residue_pair
            )

        return residue_pair
