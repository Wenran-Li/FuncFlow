from pathlib import Path
import os
import glob
import argparse
import numpy as np
import torch
import mdtraj as md

from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from data.datasets import PdbDataset


SS_TO_INT = {
    "H": 0,
    "E": 1,
    "C": 2,
}

INT_TO_SS = {
    0: "H",
    1: "E",
    2: "C",
}


def dssp_from_pdb(pdb_path):
    """
    Compute simplified DSSP:
        H -> 0
        E -> 1
        C -> 2
        unknown -> -1
    """
    traj = md.load(pdb_path)

    dssp = md.compute_dssp(
        traj,
        simplified=True
    )[0]

    out = np.full(len(dssp), -1, dtype=np.int64)

    for i, s in enumerate(dssp):
        if s in SS_TO_INT:
            out[i] = SS_TO_INT[s]

    return out


def get_ca_coords(pdb_path):
    """
    Return CA coordinates in Angstrom.
    MDTraj stores nm, so multiply by 10.
    """
    traj = md.load(pdb_path)

    topology = traj.topology

    ca_indices = [
        atom.index
        for atom in topology.atoms
        if atom.name == "CA"
    ]

    xyz = traj.xyz[0, ca_indices, :] * 10.0

    return xyz


def axis_from_points(points):
    """
    Principal axis of an SSE.

    Sign is fixed N -> C by requiring the axis
    to point roughly from first residue to last residue.

    Returns
    -------
    axis : np.ndarray shape (3,) or None
    """
    if len(points) < 2:
        return None

    center = points.mean(axis=0)
    x = points - center

    try:
        _, _, vh = np.linalg.svd(x, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    axis = vh[0]

    n_to_c = points[-1] - points[0]

    if np.dot(axis, n_to_c) < 0:
        axis = -axis

    norm = np.linalg.norm(axis)

    if norm < 1e-8:
        return None

    return axis / norm


def min_distance(group_i, group_j):
    """
    Minimum CA-CA distance between two residue groups.
    """
    diff = group_i[:, None, :] - group_j[None, :, :]
    d = np.linalg.norm(diff, axis=-1)

    return float(d.min())


def orientation_class(axis_i, axis_j):
    """
    Same convention used in the SSE graph:

      0 = other
      1 = parallel
      2 = antiparallel
      3 = unknown

    Threshold |cos| >= 0.7.
    """
    if axis_i is None or axis_j is None:
        return 3

    cos = float(np.dot(axis_i, axis_j))

    if cos >= 0.7:
        return 1
    elif cos <= -0.7:
        return 2
    else:
        return 0


def extract_generated_graph_on_target_segments(
    ca_coords,
    residue_to_sse,
    sse_mask,
    contact_cutoff=8.0,
):
    """
    Evaluate generated structure using the TARGET SSE residue assignment.

    This intentionally avoids SSE node-matching ambiguity.

    Returns
    -------
    contact : [K, K] bool
    orient  : [K, K] int
    valid_axis : [K] bool
    """

    residue_to_sse = np.asarray(residue_to_sse)
    sse_mask = np.asarray(sse_mask).astype(bool)

    K = len(sse_mask)

    groups = []
    axes = []

    for k in range(K):

        if not sse_mask[k]:
            groups.append(None)
            axes.append(None)
            continue

        residues = np.where(residue_to_sse == k)[0]

        residues = residues[residues < len(ca_coords)]

        if len(residues) == 0:
            groups.append(None)
            axes.append(None)
            continue

        pts = ca_coords[residues]

        groups.append(pts)
        axes.append(axis_from_points(pts))

    contact = np.zeros((K, K), dtype=bool)
    orient = np.full((K, K), 3, dtype=np.int64)

    for i in range(K):
        if not sse_mask[i] or groups[i] is None:
            continue

        for j in range(i + 1, K):
            if not sse_mask[j] or groups[j] is None:
                continue

            # Ignore sequential neighbors for spatial topology.
            if abs(i - j) <= 1:
                continue

            dmin = min_distance(groups[i], groups[j])

            if dmin <= contact_cutoff:
                contact[i, j] = True
                contact[j, i] = True

            o = orientation_class(
                axes[i],
                axes[j],
            )

            orient[i, j] = o
            orient[j, i] = o

    valid_axis = np.array(
        [a is not None for a in axes],
        dtype=bool
    )

    return contact, orient, valid_axis


def safe_prf(y_true, y_pred):
    if len(y_true) == 0:
        return np.nan, np.nan, np.nan

    p, r, f, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    return float(p), float(r), float(f)


def find_samples(run_dir):
    """
    Expected structure:

      val_000/length_256/sample_0/sample.pdb
      val_000/length_256/sample_1/sample.pdb
      ...
    """
    pattern = os.path.join(
        run_dir,
        "val_*",
        "length_*",
        "sample_*",
        "sample.pdb",
    )

    files = sorted(glob.glob(pattern))

    samples = []

    for path in files:

        parts = path.split(os.sep)

        val_part = [
            x for x in parts
            if x.startswith("val_")
        ][-1]

        sample_part = [
            x for x in parts
            if x.startswith("sample_")
        ][-1]

        val_idx = int(val_part.split("_")[1])
        sample_id = int(sample_part.split("_")[1])

        samples.append(
            (val_idx, sample_id, path)
        )

    return samples


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_dir",
        required=True,
    )

    parser.add_argument(
        "--train_config",
        default=None,
        help="Evaluation config YAML. Defaults to <run_dir>/config.yaml.",
    )

    parser.add_argument(
        "--contact_cutoff",
        type=float,
        default=8.0,
    )

    args = parser.parse_args()

    config_path = (
        Path(args.train_config)
        if args.train_config is not None
        else Path(args.run_dir) / "config.yaml"
    )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Evaluation config not found: {config_path}. "
            "Provide --train_config explicitly if necessary."
        )

    cfg = OmegaConf.load(config_path)

    print("Evaluation config:", config_path)
    print("max_num_res:", cfg.pdb_dataset.filter.max_num_res)
    print("metadata CSV:", cfg.pdb_dataset.csv_path)

    val_dataset = PdbDataset(
        dataset_cfg=cfg.pdb_dataset,
        task=cfg.data.task,
        is_training=False,
    )

    samples = find_samples(args.run_dir)

    print("=" * 70)
    print("Evaluation")
    print("=" * 70)
    print("Run:", args.run_dir)
    print("Samples found:", len(samples))
    print()

    if len(samples) == 0:
        raise RuntimeError(
            f"No sample.pdb files found in {args.run_dir}"
        )

    all_target_ss = []
    all_pred_ss = []

    sample_ss_acc = []
    sample_ss_f1 = []

    all_contact_true = []
    all_contact_pred = []

    all_orientation_true = []
    all_orientation_pred = []

    sample_contact_f1 = []
    sample_orientation_acc = []

    for n, (val_idx, sample_id, pdb_path) in enumerate(samples):

        x = val_dataset[val_idx]

        target_ss = x["ss"].cpu().numpy()
        target_ss_mask = x["ss_mask"].cpu().numpy().astype(bool)

        residue_to_sse = (
            x["residue_to_sse"]
            .cpu()
            .numpy()
        )

        sse_mask = (
            x["sse_mask"]
            .cpu()
            .numpy()
            .astype(bool)
        )

        target_edge_type = (
            x["sse_edge_type"]
            .cpu()
            .numpy()
        )

        target_orientation = (
            x["sse_orientation"]
            .cpu()
            .numpy()
        )

        # ------------------------------------------------
        # Level 1: DSSP
        # ------------------------------------------------

        pred_ss = dssp_from_pdb(
            pdb_path
        )

        L = min(
            len(target_ss),
            len(pred_ss),
        )

        valid = (
            target_ss_mask[:L]
            & (target_ss[:L] >= 0)
            & (pred_ss[:L] >= 0)
        )

        yt = target_ss[:L][valid]
        yp = pred_ss[:L][valid]

        all_target_ss.extend(
            yt.tolist()
        )

        all_pred_ss.extend(
            yp.tolist()
        )

        if len(yt) > 0:

            acc = accuracy_score(
                yt,
                yp,
            )

            mf1 = f1_score(
                yt,
                yp,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )

            sample_ss_acc.append(acc)
            sample_ss_f1.append(mf1)

        # ------------------------------------------------
        # Level 2: SSE topology
        # ------------------------------------------------

        ca = get_ca_coords(
            pdb_path
        )

        pred_contact, pred_orientation, valid_axis = (
            extract_generated_graph_on_target_segments(
                ca_coords=ca,
                residue_to_sse=residue_to_sse,
                sse_mask=sse_mask,
                contact_cutoff=args.contact_cutoff,
            )
        )

        K = len(sse_mask)

        contact_true_sample = []
        contact_pred_sample = []

        orient_true_sample = []
        orient_pred_sample = []

        for i in range(K):

            if not sse_mask[i]:
                continue

            for j in range(i + 1, K):

                if not sse_mask[j]:
                    continue

                # Spatial graph excludes adjacent SSEs.
                if abs(i - j) <= 1:
                    continue

                target_contact = (
                    target_edge_type[i, j] == 2
                )

                predicted_contact = bool(
                    pred_contact[i, j]
                )

                contact_true_sample.append(
                    int(target_contact)
                )

                contact_pred_sample.append(
                    int(predicted_contact)
                )

                all_contact_true.append(
                    int(target_contact)
                )

                all_contact_pred.append(
                    int(predicted_contact)
                )

                # Orientation is only defined meaningfully
                # for TARGET spatial-contact pairs.
                if (
                    target_contact
                    and valid_axis[i]
                    and valid_axis[j]
                    and target_orientation[i, j] in [0, 1, 2]
                ):

                    orient_true_sample.append(
                        int(
                            target_orientation[i, j]
                        )
                    )

                    orient_pred_sample.append(
                        int(
                            pred_orientation[i, j]
                        )
                    )

                    all_orientation_true.append(
                        int(
                            target_orientation[i, j]
                        )
                    )

                    all_orientation_pred.append(
                        int(
                            pred_orientation[i, j]
                        )
                    )

        _, _, cf1 = safe_prf(
            contact_true_sample,
            contact_pred_sample,
        )

        if not np.isnan(cf1):
            sample_contact_f1.append(
                cf1
            )

        if len(orient_true_sample) > 0:

            oa = accuracy_score(
                orient_true_sample,
                orient_pred_sample,
            )

            sample_orientation_acc.append(
                oa
            )

        print(
            f"[{n+1:02d}/{len(samples):02d}] "
            f"val={val_idx:03d} "
            f"sample={sample_id} "
            f"L={L}"
        )

    # ====================================================
    # Aggregate Level 1
    # ====================================================

    all_target_ss = np.asarray(
        all_target_ss
    )

    all_pred_ss = np.asarray(
        all_pred_ss
    )

    ss_acc = accuracy_score(
        all_target_ss,
        all_pred_ss,
    )

    ss_macro_f1 = f1_score(
        all_target_ss,
        all_pred_ss,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )

    ss_class_f1 = f1_score(
        all_target_ss,
        all_pred_ss,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )

    # ====================================================
    # Aggregate Level 2
    # ====================================================

    cp, cr, cf1 = safe_prf(
        all_contact_true,
        all_contact_pred,
    )

    if len(all_orientation_true) > 0:
        orientation_acc = accuracy_score(
            all_orientation_true,
            all_orientation_pred,
        )
    else:
        orientation_acc = np.nan

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print()
    print("Residue-level secondary structure")
    print("---------------------------------")
    print(
        f"Accuracy       : {ss_acc:.4f}"
    )
    print(
        f"Macro-F1       : {ss_macro_f1:.4f}"
    )

    for i, name in enumerate(["H", "E", "C"]):
        print(
            f"{name} F1           : "
            f"{ss_class_f1[i]:.4f}"
        )

    print(
        f"Per-sample Acc : "
        f"{np.mean(sample_ss_acc):.4f} "
        f"± {np.std(sample_ss_acc):.4f}"
    )

    print(
        f"Per-sample F1  : "
        f"{np.mean(sample_ss_f1):.4f} "
        f"± {np.std(sample_ss_f1):.4f}"
    )

    print()
    print("SSE spatial contact topology")
    print("----------------------------")
    print(
        f"Precision      : {cp:.4f}"
    )
    print(
        f"Recall         : {cr:.4f}"
    )
    print(
        f"F1             : {cf1:.4f}"
    )
    print(
        f"Pairs evaluated: "
        f"{len(all_contact_true)}"
    )
    print(
        f"Positive target pairs: "
        f"{sum(all_contact_true)}"
    )

    print(
        f"Per-sample F1  : "
        f"{np.mean(sample_contact_f1):.4f} "
        f"± {np.std(sample_contact_f1):.4f}"
    )

    print()
    print("SSE orientation")
    print("---------------")
    print(
        f"Accuracy       : "
        f"{orientation_acc:.4f}"
    )
    print(
        f"Pairs evaluated: "
        f"{len(all_orientation_true)}"
    )

    if len(sample_orientation_acc) > 0:
        print(
            f"Per-sample Acc : "
            f"{np.mean(sample_orientation_acc):.4f} "
            f"± {np.std(sample_orientation_acc):.4f}"
        )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
