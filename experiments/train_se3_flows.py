import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Pytorch lightning imports
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from data.datasets import ScopeDataset, PdbDataset
from data.protein_dataloader import ProteinData
from models.flow_module import FlowModule
from experiments import utils as eu
import wandb

log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision('high')


class Experiment:

    def __init__(self, *, cfg: DictConfig):
        self._cfg = cfg
        self._data_cfg = cfg.data
        self._exp_cfg = cfg.experiment
        self._task = self._data_cfg.task
        self._setup_dataset()
        self._datamodule: LightningDataModule = ProteinData(
            data_cfg=self._data_cfg,
            train_dataset=self._train_dataset,
            valid_dataset=self._valid_dataset
        )
        self._train_device_ids = list(range(self._exp_cfg.num_devices))
        log.info(f"Training with devices: {self._train_device_ids}")
        self._module: LightningModule = FlowModule(self._cfg)

        if self._exp_cfg.pretrained_ckpt is not None:
            if self._exp_cfg.warm_start is not None:
                raise ValueError(
                    "pretrained_ckpt and warm_start cannot be used together."
                )

            self._load_pretrained_model(
                self._exp_cfg.pretrained_ckpt
            )

    def _setup_dataset(self):
        if self._data_cfg.dataset == 'scope':
            self._train_dataset, self._valid_dataset = eu.dataset_creation(
                ScopeDataset, self._cfg.scope_dataset, self._task)
        elif self._data_cfg.dataset == 'pdb':
            self._train_dataset, self._valid_dataset = eu.dataset_creation(
                PdbDataset, self._cfg.pdb_dataset, self._task)
        else:
            raise ValueError(f'Unrecognized dataset {self._data_cfg.dataset}') 

    def _load_pretrained_model(self, ckpt_path):
        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
        )

        state = ckpt["state_dict"]
        linear_key = "model.node_feature_net.linear.weight"

        old_in_dim = state[linear_key].shape[1]
        new_in_dim = self._module.state_dict()[linear_key].shape[1]

        print(
            f"[Pretrained] Node projection: "
            f"{old_in_dim} -> {new_in_dim}"
        )

        if old_in_dim == 385 and new_in_dim == 417:
            return self._load_pretrained_frameflow(
                ckpt_path
            )

        if old_in_dim == 417 and new_in_dim == 481:
            return self._load_pretrained_ss_frameflow(
                ckpt_path
            )

        # M2 hierarchical model -> M3 pair-conditioned model.
        #
        # NodeFeatureNet dimensions are unchanged because
        # the new conditioning pathway is injected directly
        # into EdgeFeatureNet.
        if old_in_dim == 481 and new_in_dim == 481:
            return self._load_pretrained_hierarchical_frameflow(
                ckpt_path
            )

        raise RuntimeError(
            f"Unsupported pretrained transition: "
            f"{old_in_dim} -> {new_in_dim}"
        )

    def _load_pretrained_frameflow(self, ckpt_path):
        log.info(f"Loading pretrained FrameFlow from {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["state_dict"]

        current = self._module.state_dict()

        linear_key = "model.node_feature_net.linear.weight"
        ss_key = "model.node_feature_net.ss_embedding.weight"

        old_w = state[linear_key]
        new_w = current[linear_key]

        print(
            f"[Pretrained] Expanding {linear_key}: "
            f"{tuple(old_w.shape)} -> {tuple(new_w.shape)}"
        )

        assert old_w.shape == (256, 385)
        assert new_w.shape == (256, 417)

        # [original FrameFlow features | new SS features]
        expanded_w = torch.zeros_like(new_w)
        expanded_w[:, :old_w.shape[1]] = old_w

        state[linear_key] = expanded_w

        # Everything shared must now have identical shape.
        for key, value in state.items():
            if key not in current:
                raise RuntimeError(
                    f"Unexpected checkpoint key: {key}"
                )

            if value.shape != current[key].shape:
                raise RuntimeError(
                    f"Shape mismatch for {key}: "
                    f"{tuple(value.shape)} vs "
                    f"{tuple(current[key].shape)}"
                )

        missing, unexpected = self._module.load_state_dict(
            state,
            strict=False,
        )

        if set(missing) != {ss_key}:
            raise RuntimeError(
                f"Unexpected missing keys: {missing}"
            )

        if unexpected:
            raise RuntimeError(
                f"Unexpected keys: {unexpected}"
            )

        # Final verification
        loaded_w = self._module.state_dict()[linear_key]

        assert torch.equal(
            loaded_w[:, :old_w.shape[1]],
            old_w,
        )

        assert torch.count_nonzero(
            loaded_w[:, old_w.shape[1]:]
        ).item() == 0

        print("[Pretrained] Original 385 columns loaded exactly.")
        print("[Pretrained] New 32 SS columns initialized to zero.")
        print("[Pretrained] SS embedding remains newly initialized.")
        print("[Pretrained] INITIALIZATION PASSED.")

    def _load_pretrained_ss_frameflow(self, ckpt_path):
        log.info(
            f"Loading pretrained SS-FrameFlow from {ckpt_path}"
        )

        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
        )

        state = ckpt["state_dict"]
        current = self._module.state_dict()

        linear_key = (
            "model.node_feature_net.linear.weight"
        )

        old_w = state[linear_key]
        new_w = current[linear_key]

        print(
            f"[Hierarchical] Expanding {linear_key}: "
            f"{tuple(old_w.shape)} -> "
            f"{tuple(new_w.shape)}"
        )

        assert old_w.shape == (
            256,
            417,
        )

        assert new_w.shape == (
            256,
            481,
        )

        # -----------------------------------------------------
        # Preserve complete trained SS-FrameFlow.
        #
        # [trained 417 columns | new 64 SSE graph columns]
        # -----------------------------------------------------

        expanded_w = torch.zeros_like(
            new_w
        )

        expanded_w[
            :,
            :417,
        ] = old_w

        state[linear_key] = expanded_w

        # -----------------------------------------------------
        # Shared checkpoint parameters must match exactly.
        #
        # New SSEGraphEncoder parameters are allowed to be
        # absent from the old checkpoint.
        # -----------------------------------------------------

        for key, value in state.items():

            if key not in current:
                raise RuntimeError(
                    f"Unexpected checkpoint key: {key}"
                )

            if value.shape != current[key].shape:
                raise RuntimeError(
                    f"Shape mismatch for {key}: "
                    f"{tuple(value.shape)} vs "
                    f"{tuple(current[key].shape)}"
                )

        missing, unexpected = (
            self._module.load_state_dict(
                state,
                strict=False,
            )
        )

        allowed_missing_prefix = (
            "model.sse_graph_encoder."
        )

        bad_missing = [
            key
            for key in missing
            if not key.startswith(
                allowed_missing_prefix
            )
        ]

        if bad_missing:
            raise RuntimeError(
                "Unexpected missing keys: "
                f"{bad_missing}"
            )

        if unexpected:
            raise RuntimeError(
                f"Unexpected keys: {unexpected}"
            )

        # -----------------------------------------------------
        # Exact initialization verification
        # -----------------------------------------------------

        loaded_w = (
            self._module
            .state_dict()[linear_key]
        )

        assert torch.equal(
            loaded_w[:, :417],
            old_w,
        )

        assert (
            torch.count_nonzero(
                loaded_w[:, 417:]
            ).item()
            == 0
        )

        print(
            "[Hierarchical] Existing 417 SS-FrameFlow "
            "columns loaded exactly."
        )

        print(
            "[Hierarchical] New 64 SSE graph columns "
            "initialized to zero."
        )

        print(
            "[Hierarchical] SSEGraphEncoder newly initialized."
        )

        print(
            "[Hierarchical] INITIALIZATION PASSED."
        )

    def _load_pretrained_hierarchical_frameflow(
            self,
            ckpt_path,
        ):
            """
            Load the trained M2 hierarchical model into M3.

            M2 already contains:
                - residue SS conditioning
                - SSEGraphEncoder node conditioning
                - 481-dim NodeFeatureNet input

            M3 adds only:
                - SSEPairEncoder
                - zero-initialized pair projection in EdgeFeatureNet

            Therefore all M2 parameters must be copied exactly.
            No NodeFeatureNet expansion is required.
            """

            log.info(
                "Loading pretrained hierarchical FrameFlow "
                f"from {ckpt_path}"
            )

            ckpt = torch.load(
                ckpt_path,
                map_location="cpu",
            )

            state = ckpt["state_dict"]
            current = self._module.state_dict()

            linear_key = (
                "model.node_feature_net.linear.weight"
            )

            old_w = state[linear_key]
            new_w = current[linear_key]

            print(
                "[M2 -> M3] Node projection unchanged: "
                f"{tuple(old_w.shape)} -> "
                f"{tuple(new_w.shape)}"
            )

            assert old_w.shape == (256, 481)
            assert new_w.shape == (256, 481)

            # -----------------------------------------------------
            # Every parameter present in M2 must also exist in M3
            # with exactly the same shape.
            # -----------------------------------------------------

            for key, value in state.items():

                if key not in current:
                    raise RuntimeError(
                        "Unexpected M2 checkpoint key: "
                        f"{key}"
                    )

                if value.shape != current[key].shape:
                    raise RuntimeError(
                        f"Shape mismatch for {key}: "
                        f"M2={tuple(value.shape)} "
                        f"M3={tuple(current[key].shape)}"
                    )

            # -----------------------------------------------------
            # Load all trained M2 parameters exactly.
            # -----------------------------------------------------

            missing, unexpected = (
                self._module.load_state_dict(
                    state,
                    strict=False,
                )
            )

            # -----------------------------------------------------
            # ONLY the newly introduced M3 pair-conditioning
            # parameters may be absent.
            # -----------------------------------------------------

            allowed_missing_prefixes = (
                "model.sse_pair_encoder.",
                "model.edge_feature_net.sse_pair_proj.",
            )

            bad_missing = [
                key
                for key in missing
                if not key.startswith(
                    allowed_missing_prefixes
                )
            ]

            if bad_missing:
                raise RuntimeError(
                    "Unexpected missing keys in M2 -> M3: "
                    f"{bad_missing}"
                )

            if unexpected:
                raise RuntimeError(
                    "Unexpected checkpoint keys in M2 -> M3: "
                    f"{unexpected}"
                )

            print(
                f"[M2 -> M3] New M3 parameters: "
                f"{len(missing)}"
            )

            for key in missing:
                print(
                    f"[M2 -> M3]   {key}"
                )

            # -----------------------------------------------------
            # Verify EVERY M2 tensor was copied exactly.
            # -----------------------------------------------------

            loaded = self._module.state_dict()

            verified = 0

            for key, old_tensor in state.items():

                new_tensor = loaded[key]

                if not torch.equal(
                    old_tensor.cpu(),
                    new_tensor.cpu(),
                ):
                    max_diff = (
                        old_tensor.cpu()
                        - new_tensor.cpu()
                    ).abs().max().item()

                    raise RuntimeError(
                        "M2 parameter changed during "
                        f"M3 initialization: {key}; "
                        f"max_diff={max_diff}"
                    )

                verified += 1

            print(
                f"[M2 -> M3] Verified {verified} "
                "M2 tensors copied exactly."
            )

            # -----------------------------------------------------
            # Verify the new pair projection is EXACTLY zero.
            #
            # This guarantees:
            #
            # z_M3 = z_M2 + W_pair e_pair
            #      = z_M2
            #
            # at initialization.
            # -----------------------------------------------------

            pair_proj_key = (
                "model.edge_feature_net."
                "sse_pair_proj.weight"
            )

            pair_w = loaded[pair_proj_key]

            pair_nonzero = (
                torch.count_nonzero(
                    pair_w
                ).item()
            )

            pair_norm = (
                pair_w
                .float()
                .norm()
                .item()
            )

            print(
                "[M2 -> M3] SSE pair projection "
                f"shape: {tuple(pair_w.shape)}"
            )

            print(
                "[M2 -> M3] SSE pair projection "
                f"norm: {pair_norm}"
            )

            print(
                "[M2 -> M3] SSE pair projection "
                f"nonzero: {pair_nonzero}"
            )

            if pair_nonzero != 0:
                raise RuntimeError(
                    "M3 SSE pair projection is not "
                    "zero-initialized."
                )

            # -----------------------------------------------------
            # We expect exactly the new pair-path parameters.
            # Current architecture = 7 new tensors.
            # -----------------------------------------------------

            expected_missing = {
                "model.sse_pair_encoder.edge_embedding.weight",
                "model.sse_pair_encoder.orientation_embedding.weight",
                "model.sse_pair_encoder.proj.0.weight",
                "model.sse_pair_encoder.proj.0.bias",
                "model.sse_pair_encoder.proj.2.weight",
                "model.sse_pair_encoder.proj.2.bias",
                "model.edge_feature_net.sse_pair_proj.weight",
            }

            if set(missing) != expected_missing:
                raise RuntimeError(
                    "M3 missing-key set differs from "
                    "the expected pair branch.\n"
                    f"Expected: {sorted(expected_missing)}\n"
                    f"Actual:   {sorted(missing)}"
                )

            print(
                "[M2 -> M3] Exact missing-key set: PASS."
            )

            print(
                "[M2 -> M3] Pair projection zero-init: PASS."
            )

            print(
                "[M2 -> M3] INITIALIZATION PASSED."
            )

    def train(self):
        callbacks = []
        if self._exp_cfg.debug:
            log.info("Debug mode.")
            logger = None
            self._train_device_ids = [self._train_device_ids[0]]
            self._data_cfg.loader.num_workers = 0
        else:
            logger = WandbLogger(
                **self._exp_cfg.wandb,
            )
            
            # Checkpoint directory.
            ckpt_dir = self._exp_cfg.checkpointer.dirpath
            os.makedirs(ckpt_dir, exist_ok=True)
            log.info(f"Checkpoints saved to {ckpt_dir}")
            
            # Model checkpoints
            callbacks.append(ModelCheckpoint(**self._exp_cfg.checkpointer))
            
            # Save config only for main process.
            local_rank = os.environ.get('LOCAL_RANK', 0)
            if local_rank == 0:
                cfg_path = os.path.join(ckpt_dir, 'config.yaml')
                with open(cfg_path, 'w') as f:
                    OmegaConf.save(config=self._cfg, f=f.name)
                cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
                flat_cfg = dict(eu.flatten_dict(cfg_dict))
                if isinstance(logger.experiment.config, wandb.sdk.wandb_config.Config):
                    logger.experiment.config.update(flat_cfg)
        trainer = Trainer(
            **self._exp_cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=self._train_device_ids,
        )
        trainer.fit(
            model=self._module,
            datamodule=self._datamodule,
            ckpt_path=self._exp_cfg.warm_start
        )


@hydra.main(version_base=None, config_path="../configs", config_name="base.yaml")
def main(cfg: DictConfig):

    if cfg.experiment.warm_start is not None and cfg.experiment.warm_start_cfg_override:
        # Loads warm start config.
        warm_start_cfg_path = os.path.join(
            os.path.dirname(cfg.experiment.warm_start), 'config.yaml')
        warm_start_cfg = OmegaConf.load(warm_start_cfg_path)

        # Warm start config may not have latest fields in the base config.
        # Add these fields to the warm start config.
        OmegaConf.set_struct(cfg.model, False)
        OmegaConf.set_struct(warm_start_cfg.model, False)
        cfg.model = OmegaConf.merge(cfg.model, warm_start_cfg.model)
        OmegaConf.set_struct(cfg.model, True)
        log.info(f'Loaded warm start config from {warm_start_cfg_path}')

    exp = Experiment(cfg=cfg)
    exp.train()

if __name__ == "__main__":
    main()
