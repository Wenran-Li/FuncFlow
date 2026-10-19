import torch
from torch import nn
from models.utils import get_index_embedding, get_time_embedding


class NodeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(NodeFeatureNet, self).__init__()
        self._cfg = module_cfg

        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb

        # Backward compatible:
        # original FrameFlow config has neither of these fields.
        self.c_ss_emb = self._cfg.get("c_ss_emb", 0)
        self.c_sse_emb = self._cfg.get("c_sse_emb", 0)

        # Optional residue-level secondary-structure conditioning.
        if self.c_ss_emb > 0:
            # 0=H, 1=E, 2=C, 3=UNK
            self.ss_embedding = nn.Embedding(
                num_embeddings=4,
                embedding_dim=self.c_ss_emb,
            )
        else:
            self.ss_embedding = None

        embed_size = (
            self._cfg.c_pos_emb
            + self._cfg.c_timestep_emb * 2
            + 1
            + self.c_ss_emb
            + self.c_sse_emb
        )

        if self._cfg.embed_chain:
            embed_size += self._cfg.c_pos_emb

        self.linear = nn.Linear(embed_size, self.c_s)

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(
            1,
            mask.shape[1],
            1
        )

        return timestep_emb * mask.unsqueeze(-1)

    def forward(
        self,
        so3_t,
        r3_t,
        res_mask,
        diffuse_mask,
        pos,
        ss=None,
        ss_mask=None,
        sse_res_emb=None,
    ):
        # Position embedding
        pos_emb = get_index_embedding(
            pos,
            self.c_pos_emb,
            max_len=2056
        )
        pos_emb = pos_emb * res_mask.unsqueeze(-1)

        input_feats = [
            pos_emb,
            diffuse_mask[..., None],
            self.embed_t(so3_t, res_mask),
            self.embed_t(r3_t, res_mask),
        ]

        # ------------------------------------------------------------
        # Optional residue-level secondary-structure conditioning
        # ------------------------------------------------------------
        if self.c_ss_emb > 0:
            if ss is None:
                # No SS condition supplied -> UNK / zero contribution.
                ss = torch.full_like(pos, 3, dtype=torch.long)

            ss_tokens = ss.long().clone()

            # dataset: 0=H, 1=E, 2=C, -1=invalid
            # embedding: 0=H, 1=E, 2=C, 3=UNK
            ss_tokens = torch.where(
                ss_tokens >= 0,
                ss_tokens,
                torch.full_like(ss_tokens, 3),
            )

            ss_emb = self.ss_embedding(ss_tokens)

            if ss_mask is None:
                ss_mask = torch.zeros_like(res_mask)

            ss_emb = (
                ss_emb
                * ss_mask.unsqueeze(-1).to(ss_emb.dtype)
                * res_mask.unsqueeze(-1).to(ss_emb.dtype)
            )

            input_feats.append(ss_emb)

        # ------------------------------------------------------------
        # Optional SSE graph-to-node conditioning
        # ------------------------------------------------------------
        if self.c_sse_emb > 0:
            if sse_res_emb is None:
                sse_res_emb = torch.zeros(
                    *res_mask.shape,
                    self.c_sse_emb,
                    device=res_mask.device,
                    dtype=pos_emb.dtype,
                )

            sse_res_emb = (
                sse_res_emb
                * res_mask.unsqueeze(-1).to(sse_res_emb.dtype)
            )

            input_feats.append(sse_res_emb)

        return self.linear(
            torch.cat(input_feats, dim=-1)
        )