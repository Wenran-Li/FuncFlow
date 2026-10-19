#!/usr/bin/env python3

import argparse
import pickle
import tempfile
from pathlib import Path

import mdtraj as md
import numpy as np

from openfold.np import residue_constants


# ============================================================
# Secondary structure encoding
# ============================================================

SS_TO_INT = {
    "H": 0,   # Helix
    "E": 1,   # Extended / beta strand
    "C": 2,   # Coil
}

INT_TO_SS = {
    0: "H",
    1: "E",
    2: "C",
}


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_atom_name(atom_idx):
    """
    FrameFlow/OpenFold atom37 index -> atom name.
    """
    return residue_constants.atom_types[atom_idx]


def get_residue_name(aatype):
    """
    OpenFold aatype integer -> 3-letter residue name.
    Unknown residues become UNK.
    """
    if 0 <= int(aatype) < 20:
        aa1 = residue_constants.restypes[int(aatype)]
        return residue_constants.restype_1to3[aa1]

    return "UNK"


def write_modeled_pdb(feats, output_path):
    """
    Reconstruct a temporary PDB from FrameFlow atom37 features.

    Only residues listed in modeled_idx are written.

    Returns
    -------
    written_indices : np.ndarray
        Original FrameFlow residue indices that were successfully written.
    """

    atom_positions = np.asarray(feats["atom_positions"])
    atom_mask = np.asarray(feats["atom_mask"])
    aatype = np.asarray(feats["aatype"])
    residue_index = np.asarray(feats["residue_index"])
    chain_index = np.asarray(feats["chain_index"])
    modeled_idx = np.asarray(feats["modeled_idx"], dtype=np.int64)

    # Simple chain naming for PDB output.
    chain_chars = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789"
    )

    unique_chains = sorted(np.unique(chain_index).tolist())

    if len(unique_chains) > len(chain_chars):
        raise ValueError(
            f"Too many chains ({len(unique_chains)}) for temporary PDB."
        )

    chain_map = {
        chain: chain_chars[i]
        for i, chain in enumerate(unique_chains)
    }

    serial = 1
    written_indices = []

    with open(output_path, "w") as f:

        for idx in modeled_idx:

            aa = int(aatype[idx])

            # Unknown/unmodeled residue.
            if aa >= 20:
                continue

            resname = get_residue_name(aa)
            chain_id = chain_map[int(chain_index[idx])]

            # Use a safe sequential numbering in the temporary PDB.
            # DSSP ordering is what matters here.
            resseq = len(written_indices) + 1

            atoms_written = 0

            for atom_idx, atom_name in enumerate(residue_constants.atom_types):

                if atom_mask[idx, atom_idx] < 0.5:
                    continue

                xyz = atom_positions[idx, atom_idx]

                if not np.all(np.isfinite(xyz)):
                    continue

                x, y, z = xyz

                element = atom_name[0]

                line = (
                    f"ATOM  {serial:5d} "
                    f"{atom_name:^4s}"
                    f" {resname:>3s} "
                    f"{chain_id:1s}"
                    f"{resseq:4d}"
                    f"    "
                    f"{x:8.3f}"
                    f"{y:8.3f}"
                    f"{z:8.3f}"
                    f"{1.00:6.2f}"
                    f"{0.00:6.2f}"
                    f"          "
                    f"{element:>2s}\n"
                )

                f.write(line)

                serial += 1
                atoms_written += 1

            if atoms_written > 0:
                written_indices.append(idx)

        f.write("END\n")

    return np.asarray(written_indices, dtype=np.int64)


def annotate_one(pkl_path, input_root, output_root, verbose=True):
    """
    Annotate one FrameFlow pickle with simplified DSSP labels.

    Output:
        output_root / relative_path / protein.npz
    """

    pkl_path = Path(pkl_path)
    input_root = Path(input_root)
    output_root = Path(output_root)

    relative = pkl_path.relative_to(input_root)
    output_path = output_root / relative.with_suffix(".npz")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        if verbose:
            print(f"[SKIP] {pkl_path} -> already exists")
        return "skip"

    feats = load_pickle(pkl_path)

    required = [
        "atom_positions",
        "atom_mask",
        "aatype",
        "modeled_idx",
        "residue_index",
        "chain_index",
    ]

    for key in required:
        if key not in feats:
            raise KeyError(f"{pkl_path}: missing key '{key}'")

    seq_len = len(feats["aatype"])
    modeled_idx = np.asarray(feats["modeled_idx"], dtype=np.int64)

    if len(modeled_idx) == 0:
        raise ValueError(f"{pkl_path}: no modeled residues")

    # Full-length labels.
    # -1 means residue has no DSSP label / not modeled.
    ss_full = np.full(seq_len, -1, dtype=np.int8)

    with tempfile.TemporaryDirectory() as tmpdir:

        tmp_pdb = Path(tmpdir) / "structure.pdb"

        written_indices = write_modeled_pdb(
            feats,
            tmp_pdb,
        )

        if len(written_indices) == 0:
            raise ValueError(
                f"{pkl_path}: no residues written to temporary PDB"
            )

        traj = md.load(str(tmp_pdb))

        # simplified=True returns H / E / C
        dssp = md.compute_dssp(
            traj,
            simplified=True,
        )[0]

    if len(dssp) != len(written_indices):
        raise ValueError(
            f"{pkl_path}: DSSP length mismatch: "
            f"DSSP={len(dssp)}, "
            f"written residues={len(written_indices)}, "
            f"modeled_idx={len(modeled_idx)}"
        )

    ss_modeled = np.full(len(dssp), -1, dtype=np.int8)

    for i, state in enumerate(dssp):
        state = str(state)

        if state in SS_TO_INT:
            ss_modeled[i] = SS_TO_INT[state]
        elif state == "NA":
            # Leave as -1 and mask during training.
            continue
        else:
            raise ValueError(
                f"{pkl_path}: unexpected DSSP state '{state}'"
            )

    # Only assign valid DSSP labels.
    valid_dssp = ss_modeled >= 0

    ss_full[written_indices[valid_dssp]] = ss_modeled[valid_dssp]

    ss_mask = (ss_full >= 0).astype(np.int8)

    h = int(np.sum(ss_modeled == 0))
    e = int(np.sum(ss_modeled == 1))
    c = int(np.sum(ss_modeled == 2))
    na = int(np.sum(ss_modeled == -1))
    n_valid = h + e + c

    np.savez_compressed(
        output_path,
        ss=ss_full,
        ss_mask=ss_mask,
        modeled_idx=modeled_idx,
        dssp_indices=written_indices,
        seq_len=np.asarray(seq_len),
    )

    if verbose:
        if n_valid > 0:
            print(
                f"[OK] {pkl_path.name}: "
                f"seq={seq_len}, "
                f"modeled={len(modeled_idx)}, "
                f"dssp={len(ss_modeled)}, "
                f"valid={n_valid}, "
                f"NA={na}, "
                f"H={h} ({h/n_valid:.3f}), "
                f"E={e} ({e/n_valid:.3f}), "
                f"C={c} ({c/n_valid:.3f})"
            )
        else:
            print(
                f"[OK] {pkl_path.name}: "
                f"seq={seq_len}, modeled={len(modeled_idx)}, "
                f"dssp={len(ss_modeled)}, valid=0, NA={na}"
            )

    return "ok"


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default="processed_pdb",
        help="FrameFlow processed_pdb directory",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="dssp_labels",
        help="Output directory",
    )

    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help="Process only one pickle",
    )

    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)

    if args.single is not None:

        annotate_one(
            Path(args.single),
            input_root,
            output_root,
        )

        return

    files = sorted(input_root.rglob("*.pkl"))

    print(f"Found {len(files)} pickle files.")

    n_ok = 0
    n_skip = 0
    failures = []

    for i, path in enumerate(files, start=1):

        try:

            status = annotate_one(
                path,
                input_root,
                output_root,
                verbose=False,
            )

            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1

        except Exception as e:

            failures.append(
                (str(path), str(e))
            )

            print(
                f"[FAIL] {path}: {e}"
            )

        if i % 100 == 0 or i == len(files):

            print(
                f"[{i}/{len(files)}] "
                f"OK={n_ok} "
                f"SKIP={n_skip} "
                f"FAIL={len(failures)}"
            )

    failure_file = output_root / "failures.txt"

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(failure_file, "w") as f:

        for path, error in failures:
            f.write(
                f"{path}\t{error}\n"
            )

    print()
    print("Finished.")
    print(f"OK:   {n_ok}")
    print(f"SKIP: {n_skip}")
    print(f"FAIL: {len(failures)}")
    print(f"Failure log: {failure_file}")


if __name__ == "__main__":
    main()