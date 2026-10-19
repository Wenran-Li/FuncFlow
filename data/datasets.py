import abc
import numpy as np
import pandas as pd
import logging
import tree
import torch
import random

import os

from torch.utils.data import Dataset
from data import utils as du
from openfold.data import data_transforms
from openfold.utils import rigid_utils
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from data.sse_utils import (
    ss_tensor_to_string,
    build_padded_sse_graph,
)


def _rog_filter(df, quantile):
    y_quant = pd.pivot_table(
        df,
        values='radius_gyration', 
        index='modeled_seq_len',
        aggfunc=lambda x: np.quantile(x, quantile)
    )
    x_quant = y_quant.index.to_numpy()
    y_quant = y_quant.radius_gyration.to_numpy()

    # Fit polynomial regressor
    poly = PolynomialFeatures(degree=4, include_bias=True)
    poly_features = poly.fit_transform(x_quant[:, None])
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y_quant)

    # Calculate cutoff for all sequence lengths
    max_len = df.modeled_seq_len.max()
    pred_poly_features = poly.fit_transform(np.arange(max_len)[:, None])
    # Add a little more.
    pred_y = poly_reg_model.predict(pred_poly_features) + 0.1

    row_rog_cutoffs = df.modeled_seq_len.map(lambda x: pred_y[x-1])
    return df[df.radius_gyration < row_rog_cutoffs]


def _length_filter(data_csv, min_res, max_res):
    return data_csv[
        (data_csv.modeled_seq_len >= min_res)
        & (data_csv.modeled_seq_len <= max_res)
    ]


def _plddt_percent_filter(data_csv, min_plddt_percent):
    return data_csv[data_csv.num_confident_plddt > min_plddt_percent]


def _max_coil_filter(data_csv, max_coil_percent):
    return data_csv[data_csv.coil_percent <= max_coil_percent]

def _process_csv_row(processed_file_path):
    processed_feats = du.read_pkl(processed_file_path)
    processed_feats = du.parse_chain_feats(processed_feats)

    # Only take modeled residues.
    modeled_idx = processed_feats['modeled_idx']
    min_idx = np.min(modeled_idx)
    max_idx = np.max(modeled_idx)

    # ---------------------------------------------------------
    # Load DSSP secondary-structure labels.
    #
    # processed_pdb/6v/16vp.pkl
    #        ->
    # dssp_labels/6v/16vp.npz
    # ---------------------------------------------------------
    dssp_path = processed_file_path.replace(
        'processed_pdb', 'dssp_labels'
    )
    dssp_path = dssp_path.rsplit('.', 1)[0] + '.npz'

    if not os.path.exists(dssp_path):
        raise FileNotFoundError(
            f'DSSP label not found for {processed_file_path}: {dssp_path}'
        )

    with np.load(dssp_path) as dssp_data:
        ss_full = dssp_data['ss']
        ss_mask_full = dssp_data['ss_mask']

    # Sanity check BEFORE cropping.
    if len(ss_full) != len(processed_feats['aatype']):
        raise ValueError(
            f'DSSP/PDB length mismatch for {processed_file_path}: '
            f'ss={len(ss_full)}, '
            f'aatype={len(processed_feats["aatype"])}'
        )

    # Use exactly the same crop as FrameFlow backbone features.
    ss = ss_full[min_idx:(max_idx + 1)]
    ss_mask = ss_mask_full[min_idx:(max_idx + 1)]

    del processed_feats['modeled_idx']

    processed_feats = tree.map_structure(
        lambda x: x[min_idx:(max_idx + 1)],
        processed_feats
    )

    # Run through OpenFold data transforms.
    chain_feats = {
        'aatype': torch.tensor(processed_feats['aatype']).long(),
        'all_atom_positions': torch.tensor(
            processed_feats['atom_positions']
        ).double(),
        'all_atom_mask': torch.tensor(
            processed_feats['atom_mask']
        ).double()
    }

    chain_feats = data_transforms.atom37_to_frames(chain_feats)

    rigids_1 = rigid_utils.Rigid.from_tensor_4x4(
        chain_feats['rigidgroups_gt_frames']
    )[:, 0]

    rotmats_1 = rigids_1.get_rots().get_rot_mats()
    trans_1 = rigids_1.get_trans()

    res_plddt = processed_feats['b_factors'][:, 1]
    res_mask = torch.tensor(processed_feats['bb_mask']).int()

    # Re-number residue indices for each chain such that it starts from 1.
    # Randomize chain indices.
    chain_idx = processed_feats['chain_index']
    res_idx = processed_feats['residue_index']

    new_res_idx = np.zeros_like(res_idx)
    new_chain_idx = np.zeros_like(res_idx)

    all_chain_idx = np.unique(chain_idx).tolist()

    shuffled_chain_idx = np.array(
        random.sample(all_chain_idx, len(all_chain_idx))
    ) - np.min(all_chain_idx) + 1

    for i, chain_id in enumerate(all_chain_idx):
        chain_mask = (chain_idx == chain_id).astype(int)

        chain_min_idx = np.min(
            res_idx + (1 - chain_mask) * 1e3
        ).astype(int)

        new_res_idx = (
            new_res_idx
            + (res_idx - chain_min_idx + 1) * chain_mask
        )

        replacement_chain_id = shuffled_chain_idx[i]

        new_chain_idx = (
            new_chain_idx
            + replacement_chain_id * chain_mask
        )

    if torch.isnan(trans_1).any() or torch.isnan(rotmats_1).any():
        raise ValueError(
            f'Found NaNs in {processed_file_path}'
        )

    # Final alignment sanity checks.
    num_res = len(processed_feats['aatype'])

    if len(ss) != num_res:
        raise ValueError(
            f'Post-crop DSSP length mismatch for {processed_file_path}: '
            f'ss={len(ss)}, backbone={num_res}'
        )

    if len(ss_mask) != num_res:
        raise ValueError(
            f'Post-crop DSSP mask mismatch for {processed_file_path}: '
            f'ss_mask={len(ss_mask)}, backbone={num_res}'
        )

    # =========================================================
    # Hierarchical SSE graph conditioning
    # =========================================================
    #
    # Convert residue-level DSSP labels into an H/E/C string.
    #
    # DSSP encoding:
    #   0 = H
    #   1 = E
    #   2 = C
    #  -1 = unavailable
    #
    # Invalid / unavailable residues are treated as coil here so
    # that they do not accidentally create H/E SSE nodes.
    # ---------------------------------------------------------

    ss_chars = []

    for label, valid in zip(ss, ss_mask):

        if not valid:
            ss_chars.append('C')

        elif label == 0:
            ss_chars.append('H')

        elif label == 1:
            ss_chars.append('E')

        elif label == 2:
            ss_chars.append('C')

        else:
            ss_chars.append('C')

    ss_string = ''.join(ss_chars)

    # trans_1 is already cropped to exactly the same residue range
    # as ss / ss_mask.
    ca_pos = (
        trans_1
        .detach()
        .cpu()
        .numpy()
    )

    if ca_pos.shape != (num_res, 3):
        raise ValueError(
            f'Unexpected trans_1 shape for {processed_file_path}: '
            f'{ca_pos.shape}, expected ({num_res}, 3)'
        )

    sse_graph = build_padded_sse_graph(
        ss_string,
        ca_pos,
        max_sse=48,
        spatial_cutoff=8.0,
        orientation_cutoff=0.7,
    )

    # Additional graph sanity check.
    if len(sse_graph['residue_to_sse']) != num_res:
        raise ValueError(
            f'SSE graph/backbone length mismatch for '
            f'{processed_file_path}: '
            f'residue_to_sse={len(sse_graph["residue_to_sse"])}, '
            f'backbone={num_res}'
        )
 

    return {
        'res_plddt': res_plddt,
        'aatype': chain_feats['aatype'],
        'rotmats_1': rotmats_1,
        'trans_1': trans_1,
        'res_mask': res_mask,
        'chain_idx': new_chain_idx,
        'res_idx': new_res_idx,

        # -----------------------------------------------------
        # Level 1:
        # residue-level secondary-structure conditioning
        #
        # 0 = H
        # 1 = E
        # 2 = C
        # -1 = unavailable
        # -----------------------------------------------------
        'ss': torch.tensor(
            ss
        ).long(),

        'ss_mask': torch.tensor(
            ss_mask
        ).int(),

        # -----------------------------------------------------
        # Level 2:
        # hierarchical SSE graph conditioning
        # -----------------------------------------------------

        # Scalar number of real SSE nodes.
        'num_sse': torch.tensor(
            sse_graph['num_sse']
        ).long(),

        # Node features [48].
        #
        # sse_type:
        #   0 = H
        #   1 = E
        #   2 = PAD
        'sse_type': torch.from_numpy(
            sse_graph['sse_type']
        ).long(),

        'sse_length': torch.from_numpy(
            sse_graph['sse_length']
        ).long(),

        'linker_before': torch.from_numpy(
            sse_graph['linker_before']
        ).long(),

        'sse_mask': torch.from_numpy(
            sse_graph['sse_mask']
        ).long(),

        # -----------------------------------------------------
        # Pair / graph features [48, 48].
        #
        # edge_type:
        #   0 = no edge
        #   1 = sequence edge
        #   2 = spatial edge
        # -----------------------------------------------------

        'sse_edge_type': torch.from_numpy(
            sse_graph['sse_edge_type']
        ).long(),

        'sse_spatial_contact': torch.from_numpy(
            sse_graph['sse_spatial_contact']
        ).long(),

        # orientation:
        #   0 = other
        #   1 = parallel
        #   2 = antiparallel
        #   3 = unknown / invalid axis
        #   4 = PAD / no edge
        'sse_orientation': torch.from_numpy(
            sse_graph['sse_orientation']
        ).long(),

        'sse_orientation_valid': torch.from_numpy(
            sse_graph['sse_orientation_valid']
        ).long(),

        'sse_pair_mask': torch.from_numpy(
            sse_graph['sse_pair_mask']
        ).long(),

        # -----------------------------------------------------
        # Residue -> SSE mapping [L].
        #
        # -1 = coil / not assigned to an H/E SSE
        # 0..K-1 = SSE node index
        # -----------------------------------------------------
        'residue_to_sse': torch.from_numpy(
            sse_graph['residue_to_sse']
        ).long(),
    }


# def _process_csv_row(processed_file_path):
#     processed_feats = du.read_pkl(processed_file_path)
#     processed_feats = du.parse_chain_feats(processed_feats)

#     # Only take modeled residues.
#     modeled_idx = processed_feats['modeled_idx']
#     min_idx = np.min(modeled_idx)
#     max_idx = np.max(modeled_idx)
#     del processed_feats['modeled_idx']
#     processed_feats = tree.map_structure(
#         lambda x: x[min_idx:(max_idx+1)], processed_feats)

#     # Run through OpenFold data transforms.
#     chain_feats = {
#         'aatype': torch.tensor(processed_feats['aatype']).long(),
#         'all_atom_positions': torch.tensor(processed_feats['atom_positions']).double(),
#         'all_atom_mask': torch.tensor(processed_feats['atom_mask']).double()
#     }
#     chain_feats = data_transforms.atom37_to_frames(chain_feats)
#     rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]
#     rotmats_1 = rigids_1.get_rots().get_rot_mats()
#     trans_1 = rigids_1.get_trans()
#     res_plddt = processed_feats['b_factors'][:, 1]
#     res_mask = torch.tensor(processed_feats['bb_mask']).int()

#     # Re-number residue indices for each chain such that it starts from 1.
#     # Randomize chain indices.
#     chain_idx = processed_feats['chain_index']
#     res_idx = processed_feats['residue_index']
#     new_res_idx = np.zeros_like(res_idx)
#     new_chain_idx = np.zeros_like(res_idx)
#     all_chain_idx = np.unique(chain_idx).tolist()
#     shuffled_chain_idx = np.array(
#         random.sample(all_chain_idx, len(all_chain_idx))) - np.min(all_chain_idx) + 1
#     for i,chain_id in enumerate(all_chain_idx):
#         chain_mask = (chain_idx == chain_id).astype(int)
#         chain_min_idx = np.min(res_idx + (1 - chain_mask) * 1e3).astype(int)
#         new_res_idx = new_res_idx + (res_idx - chain_min_idx + 1) * chain_mask

#         # Shuffle chain_index
#         replacement_chain_id = shuffled_chain_idx[i]
#         new_chain_idx = new_chain_idx + replacement_chain_id * chain_mask
#     if torch.isnan(trans_1).any() or torch.isnan(rotmats_1).any():
#         raise ValueError(f'Found NaNs in {processed_file_path}')
#     return {
#         'res_plddt': res_plddt,
#         'aatype': chain_feats['aatype'],
#         'rotmats_1': rotmats_1,
#         'trans_1': trans_1,
#         'res_mask': res_mask,
#         'chain_idx': new_chain_idx,
#         'res_idx': new_res_idx,
#     }


def _add_plddt_mask(feats, plddt_threshold):
    feats['plddt_mask'] = torch.tensor(
        feats['res_plddt'] > plddt_threshold).int()


def _read_clusters(cluster_path):
    pdb_to_cluster = {}
    with open(cluster_path, "r") as f:
        for i,line in enumerate(f):
            for chain in line.split(' '):
                pdb = chain.split('_')[0]
                pdb_to_cluster[pdb.upper()] = i
    return pdb_to_cluster


class BaseDataset(Dataset):
    def __init__(
            self,
            *,
            dataset_cfg,
            is_training,
            task,
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self.task = task
        self.raw_csv = pd.read_csv(self.dataset_cfg.csv_path)
        metadata_csv = self._filter_metadata(self.raw_csv)
        metadata_csv = metadata_csv.sort_values(
            'modeled_seq_len', ascending=False)
        self._create_split(metadata_csv)
        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)

    @property
    def is_training(self):
        return self._is_training

    @property
    def dataset_cfg(self):
        return self._dataset_cfg
    
    def __len__(self):
        return len(self.csv)

    @abc.abstractmethod
    def _filter_metadata(self, raw_csv: pd.DataFrame) -> pd.DataFrame:
        pass

    def _create_split(self, data_csv):
        # Training or validation specific logic.
        if self.is_training:
            self.csv = data_csv
            self._log.info(
                f'Training: {len(self.csv)} examples')
        else:
            if self._dataset_cfg.max_eval_length is None:
                eval_lengths = data_csv.modeled_seq_len
            else:
                eval_lengths = data_csv.modeled_seq_len[
                    data_csv.modeled_seq_len <= self._dataset_cfg.max_eval_length
                ]
            all_lengths = np.sort(eval_lengths.unique())
            length_indices = (len(all_lengths) - 1) * np.linspace(
                0.0, 1.0, self.dataset_cfg.num_eval_lengths)
            length_indices = length_indices.astype(int)
            eval_lengths = all_lengths[length_indices]
            eval_csv = data_csv[data_csv.modeled_seq_len.isin(eval_lengths)]

            # Fix a random seed to get the same split each time.
            eval_csv = eval_csv.groupby('modeled_seq_len').sample(
                self.dataset_cfg.samples_per_eval_length,
                replace=True,
                random_state=123
            )
            eval_csv = eval_csv.sort_values('modeled_seq_len', ascending=False)
            self.csv = eval_csv
            self._log.info(
                f'Validation: {len(self.csv)} examples with lengths {eval_lengths}')
        self.csv['index'] = list(range(len(self.csv)))

    def process_csv_row(self, csv_row):
        path = csv_row['processed_path']
        seq_len = csv_row['modeled_seq_len']
        # Large protein files are slow to read. Cache them.
        use_cache = seq_len > self._dataset_cfg.cache_num_res
        if use_cache and path in self._cache:
            return self._cache[path]
        processed_row = _process_csv_row(path)
        if use_cache:
            self._cache[path] = processed_row
        return processed_row
    
    def _sample_scaffold_mask(self, batch, rng):
        trans_1 = batch['trans_1']
        num_res = trans_1.shape[0]
        min_motif_size = int(self._dataset_cfg.min_motif_percent * num_res)
        max_motif_size = int(self._dataset_cfg.max_motif_percent * num_res)

        # Sample the total number of residues that will be used as the motif.
        total_motif_size = self._rng.integers(
            low=min_motif_size,
            high=max_motif_size
        )

        # Sample motifs at different locations.
        num_motifs = rng.integers(low=1, high=total_motif_size)

        # Attempt to sample
        attempt = 0
        while attempt < 100:
            # Sample lengths of each motif.
            motif_lengths = np.sort(
                rng.integers(
                    low=1,
                    high=max_motif_size,
                    size=(num_motifs,)
                )
            )

            # Truncate motifs to not go over the motif length.
            cumulative_lengths = np.cumsum(motif_lengths)
            motif_lengths = motif_lengths[cumulative_lengths < total_motif_size]
            if len(motif_lengths) == 0:
                attempt += 1
            else:
                break
        if len(motif_lengths) == 0:
            motif_lengths = [total_motif_size]

        # Sample start location of each motif.
        seed_residues = rng.integers(
            low=0,
            high=num_res-1,
            size=(len(motif_lengths),)
        )

        # Construct the motif mask.
        motif_mask = torch.zeros(num_res)
        for motif_seed, motif_len in zip(seed_residues, motif_lengths):
            motif_mask[motif_seed:min(motif_seed+motif_len, num_res)] = 1.0
        scaffold_mask = 1 - motif_mask
        return scaffold_mask * batch['res_mask']
    
    def setup_inpainting(self, feats, rng):
        diffuse_mask = self._sample_scaffold_mask(feats, rng)
        if 'plddt_mask' in feats:
            diffuse_mask = diffuse_mask * feats['plddt_mask']
        if torch.sum(diffuse_mask) < 1:
            # Should only happen rarely.
            diffuse_mask = torch.ones_like(diffuse_mask)
        feats['diffuse_mask'] = diffuse_mask

    def __getitem__(self, row_idx):
        # Process data example.
        csv_row = self.csv.iloc[row_idx]
        feats = self.process_csv_row(csv_row)

        if self._dataset_cfg.add_plddt_mask:
            _add_plddt_mask(feats, self._dataset_cfg.min_plddt_threshold)
        else:
            feats['plddt_mask'] = torch.ones_like(feats['res_mask'])

        if self.task == 'hallucination':
            feats['diffuse_mask'] = torch.ones_like(feats['res_mask']).bool()
        elif self.task == 'inpainting':
            if self._dataset_cfg.inpainting_percent < random.random():
                feats['diffuse_mask'] = torch.ones_like(feats['res_mask'])    
            else:
                rng = self._rng if self.is_training else np.random.default_rng(seed=123)
                self.setup_inpainting(feats, rng)
                # Center based on motif locations
                motif_mask = 1 - feats['diffuse_mask']
                trans_1 = feats['trans_1']
                motif_1 = trans_1 * motif_mask[:, None]
                motif_com = torch.sum(motif_1, dim=0) / (torch.sum(motif_mask) + 1)
                trans_1 -= motif_com[None, :]
                feats['trans_1'] = trans_1
        else:
            raise ValueError(f'Unknown task {self.task}')
        feats['diffuse_mask'] = feats['diffuse_mask'].int()

        # Storing the csv index is helpful for debugging.
        feats['csv_idx'] = torch.ones(1, dtype=torch.long) * row_idx
        return feats


class ScopeDataset(BaseDataset):

    def _filter_metadata(self, raw_csv):
        filter_cfg = self.dataset_cfg.filter
        data_csv = _length_filter(
            raw_csv,
            filter_cfg.min_num_res,
            filter_cfg.max_num_res
        )
        data_csv['oligomeric_detail'] = 'monomeric'
        return data_csv


class PdbDataset(BaseDataset):

    def __init__(
            self,
            *,
            dataset_cfg,
            is_training,
            task,
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self.task = task
        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)

        # Process clusters
        self.raw_csv = pd.read_csv(self.dataset_cfg.csv_path)
        metadata_csv = self._filter_metadata(self.raw_csv)
        metadata_csv = metadata_csv.sort_values(
            'modeled_seq_len', ascending=False)

        self._pdb_to_cluster = _read_clusters(self._dataset_cfg.cluster_path)
        self._max_cluster = max(self._pdb_to_cluster.values())
        self._missing_pdbs = 0
        def cluster_lookup(pdb):
            pdb = pdb.upper()
            if pdb not in self._pdb_to_cluster:
                self._pdb_to_cluster[pdb] = self._max_cluster + 1
                self._max_cluster += 1
                self._missing_pdbs += 1
            return self._pdb_to_cluster[pdb]
        metadata_csv['cluster'] = metadata_csv['pdb_name'].map(cluster_lookup)
        self._create_split(metadata_csv)
        self._all_clusters = dict(
            enumerate(self.csv['cluster'].unique().tolist()))
        self._num_clusters = len(self._all_clusters)

    def _filter_metadata(self, raw_csv):
        """Filter metadata."""
        filter_cfg = self.dataset_cfg.filter
        data_csv = raw_csv[
            raw_csv.oligomeric_detail.isin(filter_cfg.oligomeric)]
        data_csv = data_csv[
            data_csv.num_chains.isin(filter_cfg.num_chains)]
        data_csv = _length_filter(
            data_csv, filter_cfg.min_num_res, filter_cfg.max_num_res)
        data_csv = _max_coil_filter(data_csv, filter_cfg.max_coil_percent)
        data_csv = _rog_filter(data_csv, filter_cfg.rog_quantile)
        return data_csv
