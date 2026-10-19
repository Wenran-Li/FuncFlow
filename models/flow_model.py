import torch
from torch import nn

from models.node_feature_net import NodeFeatureNet
from models.edge_feature_net import EdgeFeatureNet
from models import ipa_pytorch
from data import utils as du

from models.sse_graph_encoder import (
    SSEGraphEncoder,
    broadcast_sse_to_residues,
)
import os
from models.sse_pair_encoder import SSEPairEncoder


class FlowModel(nn.Module):

    def __init__(self, model_conf):
        super(FlowModel, self).__init__()

        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa

        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(
            lambda x: x * du.ANG_TO_NM_SCALE
        )

        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(
            lambda x: x * du.NM_TO_ANG_SCALE
        )

        # ============================================================
        # Node feature network
        # ============================================================

        self.node_feature_net = NodeFeatureNet(
            model_conf.node_features
        )

        # ============================================================
        # SSE node-level graph conditioning
        #
        # SSE graph
        #   -> SSEGraphEncoder
        #   -> SSE node embeddings
        #   -> broadcast to residues
        #   -> NodeFeatureNet
        # ============================================================

        self.sse_graph_encoder = SSEGraphEncoder(
            node_dim=model_conf.node_features.c_sse_emb,
            edge_dim=32,
            hidden_dim=128,
            num_layers=2,
        )

        # ============================================================
        # SSE pair-level conditioning
        #
        # SSE graph pair relation
        #   -> SSEPairEncoder
        #   -> residue-pair embeddings
        #   -> EdgeFeatureNet
        #
        # .get() keeps old M2 checkpoint configs compatible.
        # ============================================================

        pair_dim = model_conf.edge_features.get(
            "c_sse_pair_emb",
            None,
        )

        # Backward compatibility:
        # M2 checkpoints do not contain c_sse_pair_emb and must not
        # instantiate the M3 pair-conditioning pathway.
        if pair_dim is not None:
            self.sse_pair_encoder = SSEPairEncoder(
                pair_dim=pair_dim,
            )
        else:
            self.sse_pair_encoder = None

        # ============================================================
        # Edge feature network
        # ============================================================

        self.edge_feature_net = EdgeFeatureNet(
            model_conf.edge_features
        )

        # ============================================================
        # Attention trunk
        self.trunk = nn.ModuleDict()

        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(
                self._ipa_conf
            )

            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(
                self._ipa_conf.c_s
            )

            tfmr_in = self._ipa_conf.c_s

            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False,
            )

            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer,
                self._ipa_conf.seq_tfmr_num_layers,
                enable_nested_tensor=False,
            )

            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in,
                self._ipa_conf.c_s,
                init="final",
            )

            self.trunk[f'node_transition_{b}'] = (
                ipa_pytorch.StructureModuleTransition(
                    c=self._ipa_conf.c_s
                )
            )

            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s,
                use_rot_updates=True,
            )

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size

                self.trunk[f'edge_transition_{b}'] = (
                    ipa_pytorch.EdgeTransition(
                        node_embed_size=self._ipa_conf.c_s,
                        edge_embed_in=edge_in,
                        edge_embed_out=self._model_conf.edge_embed_size,
                    )
                )

    def forward(self, input_feats):

        # ============================================================
        # Basic FrameFlow inputs
        # ============================================================

        node_mask = input_feats["res_mask"]

        edge_mask = (
            node_mask[:, None]
            * node_mask[:, :, None]
        )

        diffuse_mask = input_feats[
            "diffuse_mask"
        ]

        res_index = input_feats[
            "res_idx"
        ]

        so3_t = input_feats[
            "so3_t"
        ]

        r3_t = input_feats[
            "r3_t"
        ]

        trans_t = input_feats[
            "trans_t"
        ]

        rotmats_t = input_feats[
            "rotmats_t"
        ]

        # ============================================================
        # Level-2 SSE node conditioning
        # ============================================================

        sse_node_emb = self.sse_graph_encoder(
            sse_type=input_feats[
                "sse_type"
            ],
            sse_length=input_feats[
                "sse_length"
            ],
            linker_before=input_feats[
                "linker_before"
            ],
            sse_mask=input_feats[
                "sse_mask"
            ],
            sse_edge_type=input_feats[
                "sse_edge_type"
            ],
            sse_orientation=input_feats[
                "sse_orientation"
            ],
        )

        # [B, K, 64]
        # ->
        # [B, L, 64]

        sse_res_emb, _ = (
            broadcast_sse_to_residues(
                sse_node_emb,
                input_feats[
                    "residue_to_sse"
                ],
            )
        )

        # Mask padded residues.
        sse_res_emb = (
            sse_res_emb
            * input_feats["res_mask"]
            .unsqueeze(-1)
            .to(sse_res_emb.dtype)
        )

        # ============================================================
        # NEW:
        # Level-2 SSE pair conditioning
        #
        # SSE pair relation
        # ->
        # residue-pair relation
        #
        # Output:
        # [B, L, L, 32]
        # ============================================================

        if self.sse_pair_encoder is not None:

            pair_mode = os.environ.get(
                "SSE_PAIR_MODE",
                "correct",
            ).lower()

            if pair_mode not in {
                "correct",
                "shuffled",
                "none",
            }:
                raise ValueError(
                    f"Unknown SSE_PAIR_MODE={pair_mode}. "
                    "Expected correct, shuffled, or none."
                )

            pair_edge_type = input_feats["sse_edge_type"]
            pair_orientation = input_feats["sse_orientation"]

            # --------------------------------------------------------
            # Pair-condition negative control.
            #
            # IMPORTANT:
            # Only the tensors entering SSEPairEncoder are shuffled.
            # The SSEGraphEncoder above still receives the original,
            # correct graph.
            #
            # A deterministic permutation is used so that the same
            # wrong graph is presented at every reverse-flow step.
            # --------------------------------------------------------
            if pair_mode == "shuffled":

                pair_edge_type = pair_edge_type.clone()
                pair_orientation = pair_orientation.clone()

                original_edge_type = input_feats[
                    "sse_edge_type"
                ]
                original_orientation = input_feats[
                    "sse_orientation"
                ]

                sse_mask = input_feats["sse_mask"]

                batch_size = pair_edge_type.shape[0]

                for b in range(batch_size):

                    valid_idx = torch.where(
                        sse_mask[b].bool()
                    )[0]

                    num_sse = valid_idx.numel()

                    if num_sse <= 1:
                        continue

                    # Deterministic pseudo-random permutation.
                    # Same protein size -> same permutation.
                    # Most importantly, it remains fixed across
                    # reverse-flow integration steps.
                    generator = torch.Generator(
                        device="cpu"
                    )
                    generator.manual_seed(
                        1000 + int(num_sse)
                    )

                    perm = torch.randperm(
                        num_sse,
                        generator=generator,
                    ).to(valid_idx.device)

                    source_idx = valid_idx[perm]

                    pair_edge_type[b][
                        valid_idx[:, None],
                        valid_idx[None, :]
                    ] = original_edge_type[b][
                        source_idx[:, None],
                        source_idx[None, :]
                    ]

                    pair_orientation[b][
                        valid_idx[:, None],
                        valid_idx[None, :]
                    ] = original_orientation[b][
                        source_idx[:, None],
                        source_idx[None, :]
                    ]

            sse_pair_feats = self.sse_pair_encoder(
                sse_edge_type=pair_edge_type,
                sse_orientation=pair_orientation,
                residue_to_sse=input_feats[
                    "residue_to_sse"
                ],
                sse_mask=input_feats[
                    "sse_mask"
                ],
                res_mask=input_feats[
                    "res_mask"
                ],
            )

            if pair_mode == "none":
                sse_pair_feats = torch.zeros_like(
                    sse_pair_feats
                )

        else:
            # M2 backward-compatible path.
            sse_pair_feats = None

        # ============================================================
        # Initialize node embedding
        #
        # Includes:
        #   original FrameFlow node features
        #   residue SS conditioning
        #   SSE node conditioning
        # ============================================================

        init_node_embed = (
            self.node_feature_net(
                so3_t,
                r3_t,
                node_mask,
                diffuse_mask,
                res_index,
                input_feats["ss"],
                input_feats["ss_mask"],
                sse_res_emb,
            )
        )

        # ============================================================
        # Self-conditioning translation
        # ============================================================

        if "trans_sc" not in input_feats:

            trans_sc = torch.zeros_like(
                trans_t
            )

        else:

            trans_sc = input_feats[
                "trans_sc"
            ]

        # ============================================================
        # Initialize edge embedding
        #
        # NEW:
        # sse_pair_feats enters EdgeFeatureNet directly.
        #
        # z_ij =
        #     z_ij(FrameFlow)
        #     +
        #     W_pair e_ij(SSE)
        # ============================================================

        init_edge_embed = (
            self.edge_feature_net(
                init_node_embed,
                trans_t,
                trans_sc,
                edge_mask,
                diffuse_mask,
                sse_pair_feats=sse_pair_feats,
            )
        )

        # ============================================================
        # Initial rigid frames
        # ============================================================

        curr_rigids = du.create_rigid(
            rotmats_t,
            trans_t,
        )

        # ============================================================
        # Main IPA trunk
        # ============================================================

        curr_rigids = (
            self.rigids_ang_to_nm(
                curr_rigids
            )
        )

        init_node_embed = (
            init_node_embed
            * node_mask[..., None]
        )

        node_embed = (
            init_node_embed
            * node_mask[..., None]
        )

        edge_embed = (
            init_edge_embed
            * edge_mask[..., None]
        )

        for b in range(
            self._ipa_conf.num_blocks
        ):

            # --------------------------------------------------------
            # IPA
            # --------------------------------------------------------

            ipa_embed = (
                self.trunk[
                    f"ipa_{b}"
                ](
                    node_embed,
                    edge_embed,
                    curr_rigids,
                    node_mask,
                )
            )

            ipa_embed *= (
                node_mask[..., None]
            )

            node_embed = (
                self.trunk[
                    f"ipa_ln_{b}"
                ](
                    node_embed
                    + ipa_embed
                )
            )

            # --------------------------------------------------------
            # Sequence transformer
            # --------------------------------------------------------

            seq_tfmr_out = (
                self.trunk[
                    f"seq_tfmr_{b}"
                ](
                    node_embed,
                    src_key_padding_mask=(
                        1 - node_mask
                    ).to(
                        torch.bool
                    ),
                )
            )

            node_embed = (
                node_embed
                + self.trunk[
                    f"post_tfmr_{b}"
                ](
                    seq_tfmr_out
                )
            )

            # --------------------------------------------------------
            # Node transition
            # --------------------------------------------------------

            node_embed = (
                self.trunk[
                    f"node_transition_{b}"
                ](
                    node_embed
                )
            )

            node_embed = (
                node_embed
                * node_mask[..., None]
            )

            # --------------------------------------------------------
            # Backbone update
            # --------------------------------------------------------

            rigid_update = (
                self.trunk[
                    f"bb_update_{b}"
                ](
                    node_embed
                    * node_mask[
                        ..., None
                    ]
                )
            )

            curr_rigids = (
                curr_rigids
                .compose_q_update_vec(
                    rigid_update,
                    (
                        node_mask
                        * diffuse_mask
                    )[..., None],
                )
            )

            # --------------------------------------------------------
            # Edge transition
            # --------------------------------------------------------

            if (
                b
                < self._ipa_conf.num_blocks
                - 1
            ):

                edge_embed = (
                    self.trunk[
                        f"edge_transition_{b}"
                    ](
                        node_embed,
                        edge_embed,
                    )
                )

                edge_embed *= (
                    edge_mask[..., None]
                )

        # ============================================================
        # Convert back to Angstrom
        # ============================================================

        curr_rigids = (
            self.rigids_nm_to_ang(
                curr_rigids
            )
        )

        pred_trans = (
            curr_rigids.get_trans()
        )

        pred_rotmats = (
            curr_rigids
            .get_rots()
            .get_rot_mats()
        )

        return {
            "pred_trans": pred_trans,
            "pred_rotmats": pred_rotmats,
        }