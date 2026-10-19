import os
import torch
from torch import nn

from models.utils import (
    get_index_embedding,
    calc_distogram,
)


class EdgeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(EdgeFeatureNet, self).__init__()

        self._cfg = module_cfg

        self.c_s = self._cfg.c_s
        self.c_p = self._cfg.c_p
        self.feat_dim = self._cfg.feat_dim

        # ----------------------------------------------------
        # Original FrameFlow edge features
        # ----------------------------------------------------

        self.linear_s_p = nn.Linear(
            self.c_s,
            self.feat_dim,
        )

        self.linear_relpos = nn.Linear(
            self.feat_dim,
            self.feat_dim,
        )

        total_edge_feats = (
            self.feat_dim * 3
            + self._cfg.num_bins * 2
        )

        if self._cfg.embed_chain:
            total_edge_feats += 1

        if self._cfg.embed_diffuse_mask:
            total_edge_feats += 2

        self.edge_embedder = nn.Sequential(
            nn.Linear(
                total_edge_feats,
                self.c_p,
            ),
            nn.ReLU(),
            nn.Linear(
                self.c_p,
                self.c_p,
            ),
            nn.ReLU(),
            nn.Linear(
                self.c_p,
                self.c_p,
            ),
            nn.LayerNorm(
                self.c_p
            ),
        )

        # ----------------------------------------------------
        # NEW:
        # SSE pair conditioning
        #
        # e_ij^SSE: [B,L,L,c_sse_pair_emb]
        #
        # Project directly into the existing pair-space c_p:
        #
        # z_ij = z_ij^FrameFlow + W_pair e_ij^SSE
        #
        # Zero initialization ensures exact M2 behaviour
        # before M3 training.
        # ----------------------------------------------------

        self.c_sse_pair_emb = self._cfg.get(
            "c_sse_pair_emb",
            None,
        )

        # Backward compatibility:
        # M2 has no pair-conditioning projection.
        if self.c_sse_pair_emb is not None:
            self.sse_pair_proj = nn.Linear(
                self.c_sse_pair_emb,
                self.c_p,
                bias=False,
            )

            nn.init.zeros_(
                self.sse_pair_proj.weight
            )
        else:
            self.sse_pair_proj = None

    def embed_relpos(self, r):
        """
        AlphaFold2 Algorithms 4 & 5.

        Parameters
        ----------
        r : [B,L]

        Returns
        -------
        [B,L,L,feat_dim]
        """

        d = (
            r[:, :, None]
            - r[:, None, :]
        )

        pos_emb = get_index_embedding(
            d,
            self._cfg.feat_dim,
            max_len=2056,
        )

        return self.linear_relpos(
            pos_emb
        )

    def _cross_concat(
        self,
        feats_1d,
        num_batch,
        num_res,
    ):

        return torch.cat(
            [
                torch.tile(
                    feats_1d[
                        :, :, None, :
                    ],
                    (
                        1,
                        1,
                        num_res,
                        1,
                    ),
                ),
                torch.tile(
                    feats_1d[
                        :, None, :, :
                    ],
                    (
                        1,
                        num_res,
                        1,
                        1,
                    ),
                ),
            ],
            dim=-1,
        ).float().reshape(
            [
                num_batch,
                num_res,
                num_res,
                -1,
            ]
        )

    def forward(
        self,
        s,
        t,
        sc_t,
        p_mask,
        diffuse_mask,
        sse_pair_feats=None,
    ):
        """
        Parameters
        ----------
        s
            [B,L,c_s]

        t
            [B,L,3]

        sc_t
            [B,L,3]

        p_mask
            [B,L,L]

        diffuse_mask
            [B,L]

        sse_pair_feats
            Optional [B,L,L,c_sse_pair_emb]

        Returns
        -------
        edge_feats
            [B,L,L,c_p]
        """

        num_batch, num_res, _ = s.shape

        # ----------------------------------------------------
        # Original FrameFlow pair representation
        # ----------------------------------------------------

        p_i = self.linear_s_p(
            s
        )

        cross_node_feats = self._cross_concat(
            p_i,
            num_batch,
            num_res,
        )

        r = torch.arange(
            num_res,
            device=s.device,
        ).unsqueeze(0).repeat(
            num_batch,
            1,
        )

        relpos_feats = self.embed_relpos(
            r
        )

        dist_feats = calc_distogram(
            t,
            min_bin=1e-3,
            max_bin=20.0,
            num_bins=self._cfg.num_bins,
        )

        sc_feats = calc_distogram(
            sc_t,
            min_bin=1e-3,
            max_bin=20.0,
            num_bins=self._cfg.num_bins,
        )

        all_edge_feats = [
            cross_node_feats,
            relpos_feats,
            dist_feats,
            sc_feats,
        ]

        if self._cfg.embed_diffuse_mask:

            diff_feat = self._cross_concat(
                diffuse_mask[..., None],
                num_batch,
                num_res,
            )

            all_edge_feats.append(
                diff_feat
            )

        edge_feats = self.edge_embedder(
            torch.concat(
                all_edge_feats,
                dim=-1,
            )
        )

        # ----------------------------------------------------
        # NEW:
        # Explicit SSE pair conditioning
        # ----------------------------------------------------

        if (
            sse_pair_feats is not None
            and self.sse_pair_proj is not None
        ):

            if (
                sse_pair_feats.shape[:3]
                != edge_feats.shape[:3]
            ):
                raise ValueError(
                    "SSE pair feature shape mismatch: "
                    f"{sse_pair_feats.shape} vs "
                    f"{edge_feats.shape}"
                )

            if (
                sse_pair_feats.shape[-1]
                != self.c_sse_pair_emb
            ):
                raise ValueError(
                    "Unexpected SSE pair embedding "
                    f"dimension "
                    f"{sse_pair_feats.shape[-1]}, "
                    f"expected "
                    f"{self.c_sse_pair_emb}"
                )

            pair_update = (
                self.sse_pair_proj(
                    sse_pair_feats
                )
            )

            # Diagnostic gate for SSE pair conditioning.
            # Default 1.0 exactly reproduces the trained M3/M3.1 model.
            pair_scale = float(
                os.environ.get("SSE_PAIR_SCALE", "1.0")
            )

            pair_update = (
                pair_scale
                * pair_update
            )

            edge_feats = (
                edge_feats
                + pair_update
            )

        # ----------------------------------------------------
        # Pair mask
        # ----------------------------------------------------

        edge_feats = (
            edge_feats
            * p_mask.unsqueeze(-1)
        )

        return edge_feats