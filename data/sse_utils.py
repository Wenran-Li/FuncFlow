from dataclasses import dataclass
from typing import List

import numpy as np
import torch


@dataclass
class SSESegment:
    """
    One contiguous secondary-structure segment.

    start/end use zero-based residue indices.
    end is inclusive.
    """
    sse_type: str
    start: int
    end: int
    length: int


IDX_TO_SS = {
    0: "H",
    1: "E",
    2: "C",
}


def ss_tensor_to_string(
    ss: torch.Tensor,
    ss_mask: torch.Tensor = None,
) -> str:
    """
    Convert residue-level SS labels to a DSSP-like string.

    Encoding:
        0 -> H
        1 -> E
        2 -> C
       -1 -> X
    """

    ss = ss.detach().cpu().numpy()

    if ss_mask is None:
        mask = np.ones(len(ss), dtype=bool)
    else:
        mask = (
            ss_mask.detach()
            .cpu()
            .numpy()
            .astype(bool)
        )

    chars = []

    for value, valid in zip(ss, mask):
        if not valid:
            chars.append("X")
        else:
            chars.append(
                IDX_TO_SS.get(int(value), "X")
            )

    return "".join(chars)


def extract_raw_segments(
    ss_string: str,
) -> List[SSESegment]:
    """
    Run-length encode an H/E/C/X secondary-structure string.
    """

    if len(ss_string) == 0:
        return []

    segments = []

    start = 0
    current = ss_string[0]

    for i in range(1, len(ss_string)):

        if ss_string[i] != current:

            segments.append(
                SSESegment(
                    sse_type=current,
                    start=start,
                    end=i - 1,
                    length=i - start,
                )
            )

            start = i
            current = ss_string[i]

    # Final segment
    segments.append(
        SSESegment(
            sse_type=current,
            start=start,
            end=len(ss_string) - 1,
            length=len(ss_string) - start,
        )
    )

    return segments

@dataclass
class SSEElement:
    """
    Structural H/E element extracted from residue-level DSSP.

    Coil runs are not represented as SSE nodes.
    """
    sse_type: str
    start: int
    end: int
    length: int

    # Number of coil residues between this SSE and the
    # previous SSE in sequence.
    linker_before: int


def extract_sse_elements(
    ss_string: str,
) -> List[SSEElement]:
    """
    Convert residue-level H/E/C annotation into structural
    H/E elements.

    H and E runs become SSE nodes.
    C runs are represented implicitly as linker lengths.

    No H/E labels are removed or relabeled.
    """

    raw_segments = extract_raw_segments(ss_string)

    elements = []

    previous_sse_end = None

    for seg in raw_segments:

        if seg.sse_type not in ("H", "E"):
            continue

        if previous_sse_end is None:
            linker_before = seg.start
        else:
            linker_before = (
                seg.start - previous_sse_end - 1
            )

        elements.append(
            SSEElement(
                sse_type=seg.sse_type,
                start=seg.start,
                end=seg.end,
                length=seg.length,
                linker_before=linker_before,
            )
        )

        previous_sse_end = seg.end

    return elements

def build_residue_to_sse_map(
    ss_string: str,
    elements: List[SSEElement] = None,
) -> np.ndarray:
    """
    Map every residue to its H/E SSE index.

    Returns
    -------
    np.ndarray, shape [L]

    Values:
        -1      : residue belongs to coil/linker
        0..K-1  : index of H/E SSE element
    """

    if elements is None:
        elements = extract_sse_elements(ss_string)

    mapping = np.full(
        len(ss_string),
        -1,
        dtype=np.int64,
    )

    for sse_idx, elem in enumerate(elements):

        mapping[
            elem.start : elem.end + 1
        ] = sse_idx

    return mapping

def sse_elements_to_features(
    ss_string: str,
):
    """
    Convert H/E SSE elements into tensor-ready features.

    Returns
    -------
    dict containing:
        sse_type       [K]   H=0, E=1
        sse_length     [K]
        sse_start      [K]
        sse_end        [K]
        linker_before  [K]
        residue_to_sse [L]
    """

    elements = extract_sse_elements(ss_string)

    type_map = {
        "H": 0,
        "E": 1,
    }

    sse_type = np.asarray(
        [type_map[e.sse_type] for e in elements],
        dtype=np.int64,
    )

    sse_length = np.asarray(
        [e.length for e in elements],
        dtype=np.int64,
    )

    sse_start = np.asarray(
        [e.start for e in elements],
        dtype=np.int64,
    )

    sse_end = np.asarray(
        [e.end for e in elements],
        dtype=np.int64,
    )

    linker_before = np.asarray(
        [e.linker_before for e in elements],
        dtype=np.int64,
    )

    residue_to_sse = build_residue_to_sse_map(
        ss_string,
        elements,
    )

    return {
        "sse_type": sse_type,
        "sse_length": sse_length,
        "sse_start": sse_start,
        "sse_end": sse_end,
        "linker_before": linker_before,
        "residue_to_sse": residue_to_sse,
    }

def extract_sse_geometry(
    ca_pos: np.ndarray,
    elements: List[SSEElement],
):
    """
    Extract per-SSE geometry from residue coordinates.

    Parameters
    ----------
    ca_pos : np.ndarray
        Residue coordinates, shape [L, 3].
        For FrameFlow we use trans_1.

    elements : list[SSEElement]
        H/E SSE elements.

    Returns
    -------
    dict
        center      : [K, 3]
        axis        : [K, 3]
        axis_valid  : [K]
    """

    ca_pos = np.asarray(ca_pos, dtype=np.float32)

    if ca_pos.ndim != 2 or ca_pos.shape[-1] != 3:
        raise ValueError(
            f"Expected coordinates [L,3], got {ca_pos.shape}"
        )

    centers = []
    axes = []
    axis_valid = []

    for elem in elements:

        coords = ca_pos[
            elem.start:elem.end + 1
        ]

        center = coords.mean(axis=0)

        # Need at least two residues to define an axis.
        if len(coords) < 2:
            axis = np.zeros(3, dtype=np.float32)
            valid = False

        else:
            centered = coords - center

            _, _, vh = np.linalg.svd(
                centered,
                full_matrices=False,
            )

            axis = vh[0].astype(np.float32)

            # Resolve +/- PCA ambiguity using N -> C direction.
            nc_direction = coords[-1] - coords[0]

            if np.dot(axis, nc_direction) < 0:
                axis = -axis

            norm = np.linalg.norm(axis)

            if norm > 1e-8:
                axis = axis / norm
                valid = True
            else:
                axis = np.zeros(3, dtype=np.float32)
                valid = False

        centers.append(center)
        axes.append(axis)
        axis_valid.append(valid)

    return {
        "center": np.asarray(
            centers,
            dtype=np.float32,
        ).reshape(-1, 3),

        "axis": np.asarray(
            axes,
            dtype=np.float32,
        ).reshape(-1, 3),

        "axis_valid": np.asarray(
            axis_valid,
            dtype=bool,
        ),
    }


def build_sse_pair_geometry(
    centers: np.ndarray,
    axes: np.ndarray,
    axis_valid: np.ndarray,
):
    """
    Build pairwise geometric relations between SSEs.
    """

    centers = np.asarray(
        centers,
        dtype=np.float32,
    )

    axes = np.asarray(
        axes,
        dtype=np.float32,
    )

    axis_valid = np.asarray(
        axis_valid,
        dtype=bool,
    )

    K = centers.shape[0]

    delta = (
        centers[:, None, :]
        - centers[None, :, :]
    )

    distance = np.linalg.norm(
        delta,
        axis=-1,
    )

    axis_cosine = axes @ axes.T

    orientation_valid = (
        axis_valid[:, None]
        & axis_valid[None, :]
    )

    indices = np.arange(K)

    sequence_separation = np.abs(
        indices[:, None]
        - indices[None, :]
    )

    sequence_adjacent = (
        sequence_separation == 1
    )

    return {
        "distance": distance.astype(np.float32),
        "axis_cosine": axis_cosine.astype(np.float32),
        "orientation_valid": orientation_valid,
        "sequence_separation": sequence_separation.astype(
            np.int64
        ),
        "sequence_adjacent": sequence_adjacent,
    }

def build_sse_pair_geometry(
    centers: np.ndarray,
    axes: np.ndarray,
    axis_valid: np.ndarray,
):
    """
    Construct pairwise SSE geometric features.

    Returns
    -------
    distance : [K, K]
        Center-to-center Euclidean distance.

    axis_cosine : [K, K]
        Signed cosine between directed SSE axes.

    orientation_valid : [K, K]

    sequence_separation : [K, K]
        |i-j|

    sequence_adjacent : [K, K]
        True when |i-j| == 1.
    """

    centers = np.asarray(centers)
    axes = np.asarray(axes)
    axis_valid = np.asarray(axis_valid)

    K = len(centers)

    delta = (
        centers[:, None, :]
        - centers[None, :, :]
    )

    distance = np.linalg.norm(
        delta,
        axis=-1,
    )

    axis_cosine = axes @ axes.T

    orientation_valid = (
        axis_valid[:, None]
        & axis_valid[None, :]
    )

    idx = np.arange(K)

    sequence_separation = np.abs(
        idx[:, None] - idx[None, :]
    )

    sequence_adjacent = (
        sequence_separation == 1
    )

    return {
        "distance": distance.astype(np.float32),
        "axis_cosine": axis_cosine.astype(np.float32),
        "orientation_valid": orientation_valid,
        "sequence_separation": sequence_separation.astype(
            np.int64
        ),
        "sequence_adjacent": sequence_adjacent,
    }

def build_sse_graph(
    ss_string,
    ca_pos,
    spatial_cutoff=8.0,
    orientation_cutoff=0.7,
):
    """
    Build a hierarchical SSE graph from residue-level secondary
    structure labels and residue coordinates.

    SSE nodes
    ---------
    H/E runs are represented as nodes.

    Node features
    -------------
    sse_type:
        0 = helix
        1 = strand

    sse_length:
        number of residues in the SSE

    linker_before:
        number of coil residues preceding this SSE

    Graph edges
    -----------
    Two edge types are used:

    0 = sequence edge
        Consecutive SSE nodes in sequence.

    1 = spatial edge
        Non-neighbouring SSEs whose minimum CA distance
        is <= spatial_cutoff.

    Orientation classes
    -------------------
    0 = other
    1 = parallel
    2 = antiparallel
    3 = unknown / invalid axis

    Returns
    -------
    dict
        Graph features as numpy arrays.
    """

    ca_pos = np.asarray(
        ca_pos,
        dtype=np.float32,
    )

    if ca_pos.ndim != 2 or ca_pos.shape[1] != 3:
        raise ValueError(
            f"Expected ca_pos [L,3], got {ca_pos.shape}"
        )

    if len(ss_string) != len(ca_pos):
        raise ValueError(
            "SS/coordinate length mismatch: "
            f"{len(ss_string)} vs {len(ca_pos)}"
        )

    # ------------------------------------------------------------
    # SSE nodes
    # ------------------------------------------------------------

    elements = extract_sse_elements(
        ss_string
    )

    K = len(elements)

    sse_type = np.asarray(
        [
            0 if e.sse_type == "H" else 1
            for e in elements
        ],
        dtype=np.int64,
    )

    sse_length = np.asarray(
        [
            e.length
            for e in elements
        ],
        dtype=np.int64,
    )

    linker_before = np.asarray(
        [
            e.linker_before
            for e in elements
        ],
        dtype=np.int64,
    )

    sse_start = np.asarray(
        [
            e.start
            for e in elements
        ],
        dtype=np.int64,
    )

    sse_end = np.asarray(
        [
            e.end
            for e in elements
        ],
        dtype=np.int64,
    )

    # ------------------------------------------------------------
    # Residue -> SSE mapping
    # ------------------------------------------------------------

    residue_to_sse = build_residue_to_sse_map(
        ss_string,
        elements,
    )

    # ------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------

    geom = extract_sse_geometry(
        ca_pos,
        elements,
    )

    centers = geom["center"]
    axes = geom["axis"]
    axis_valid = geom["axis_valid"]

    # ------------------------------------------------------------
    # Edge containers
    #
    # We create directed edges because message passing usually
    # expects i->j and j->i separately.
    # ------------------------------------------------------------

    edge_src = []
    edge_dst = []

    edge_type = []

    edge_center_distance = []
    edge_min_distance = []

    edge_orientation = []
    edge_orientation_valid = []

    edge_seq_sep = []

    # ------------------------------------------------------------
    # Helper for adding directed edges
    # ------------------------------------------------------------

    def add_edge(
        i,
        j,
        e_type,
        d_center,
        d_min,
        orientation,
        orientation_valid,
        seq_sep,
    ):

        edge_src.append(i)
        edge_dst.append(j)

        edge_type.append(e_type)

        edge_center_distance.append(
            d_center
        )

        edge_min_distance.append(
            d_min
        )

        edge_orientation.append(
            orientation
        )

        edge_orientation_valid.append(
            orientation_valid
        )

        edge_seq_sep.append(
            seq_sep
        )

    # ------------------------------------------------------------
    # Build graph
    # ------------------------------------------------------------

    for i in range(K):

        for j in range(
            i + 1,
            K
        ):

            seq_sep = j - i

            Xi = ca_pos[
                elements[i].start:
                elements[i].end + 1
            ]

            Xj = ca_pos[
                elements[j].start:
                elements[j].end + 1
            ]

            # --------------------------------------------
            # Center distance
            # --------------------------------------------

            d_center = float(
                np.linalg.norm(
                    centers[i]
                    - centers[j]
                )
            )

            # --------------------------------------------
            # Minimum CA distance
            # --------------------------------------------

            diff = (
                Xi[:, None, :]
                - Xj[None, :, :]
            )

            d_min = float(
                np.linalg.norm(
                    diff,
                    axis=-1,
                ).min()
            )

            # --------------------------------------------
            # Orientation
            # --------------------------------------------

            orientation_valid = bool(
                axis_valid[i]
                and axis_valid[j]
            )

            if orientation_valid:

                cos_theta = float(
                    np.dot(
                        axes[i],
                        axes[j],
                    )
                )

                if (
                    cos_theta
                    >= orientation_cutoff
                ):

                    orientation = 1

                elif (
                    cos_theta
                    <= -orientation_cutoff
                ):

                    orientation = 2

                else:

                    orientation = 0

            else:

                orientation = 3

            # --------------------------------------------
            # Sequence edge
            # --------------------------------------------

            if seq_sep == 1:

                # forward
                add_edge(
                    i,
                    j,
                    0,
                    d_center,
                    d_min,
                    orientation,
                    orientation_valid,
                    seq_sep,
                )

                # reverse
                add_edge(
                    j,
                    i,
                    0,
                    d_center,
                    d_min,
                    orientation,
                    orientation_valid,
                    seq_sep,
                )

            # --------------------------------------------
            # Spatial edge
            #
            # Only NONLOCAL pairs.
            # --------------------------------------------

            elif (
                d_min
                <= spatial_cutoff
            ):

                # forward
                add_edge(
                    i,
                    j,
                    1,
                    d_center,
                    d_min,
                    orientation,
                    orientation_valid,
                    seq_sep,
                )

                # reverse
                add_edge(
                    j,
                    i,
                    1,
                    d_center,
                    d_min,
                    orientation,
                    orientation_valid,
                    seq_sep,
                )

    # ------------------------------------------------------------
    # Convert edge lists
    # ------------------------------------------------------------

    if len(edge_src) > 0:

        edge_index = np.stack(
            [
                np.asarray(
                    edge_src,
                    dtype=np.int64,
                ),
                np.asarray(
                    edge_dst,
                    dtype=np.int64,
                ),
            ],
            axis=0,
        )

    else:

        edge_index = np.zeros(
            (2, 0),
            dtype=np.int64,
        )

    return {

        # Basic graph size
        "num_sse": np.int64(K),

        # Node features
        "sse_type": sse_type,
        "sse_length": sse_length,
        "linker_before": linker_before,
        "sse_start": sse_start,
        "sse_end": sse_end,

        # Mapping
        "residue_to_sse": residue_to_sse.astype(
            np.int64
        ),

        # Geometry retained for analysis / later CFA experiments
        "sse_center": centers.astype(
            np.float32
        ),
        "sse_axis": axes.astype(
            np.float32
        ),
        "sse_axis_valid": axis_valid.astype(
            bool
        ),

        # Graph edges
        "sse_edge_index": edge_index,

        "sse_edge_type": np.asarray(
            edge_type,
            dtype=np.int64,
        ),

        "sse_edge_center_distance": np.asarray(
            edge_center_distance,
            dtype=np.float32,
        ),

        "sse_edge_min_distance": np.asarray(
            edge_min_distance,
            dtype=np.float32,
        ),

        "sse_edge_orientation": np.asarray(
            edge_orientation,
            dtype=np.int64,
        ),

        "sse_edge_orientation_valid": np.asarray(
            edge_orientation_valid,
            dtype=bool,
        ),

        "sse_edge_seq_sep": np.asarray(
            edge_seq_sep,
            dtype=np.int64,
        ),
    }

def build_padded_sse_graph(
    ss_string,
    ca_pos,
    max_sse=48,
    spatial_cutoff=8.0,
    orientation_cutoff=0.7,
):
    """
    Build fixed-size SSE graph tensors.

    Node tensors
    ------------
    sse_type          [Kmax]
    sse_length        [Kmax]
    linker_before     [Kmax]
    sse_mask          [Kmax]

    Pair tensors
    ------------
    sse_edge_type         [Kmax, Kmax]
    sse_spatial_contact   [Kmax, Kmax]
    sse_orientation       [Kmax, Kmax]
    sse_orientation_valid [Kmax, Kmax]
    sse_pair_mask         [Kmax, Kmax]

    Mapping
    -------
    residue_to_sse    [L]

    Encoding
    --------
    sse_type:
        0 = H
        1 = E
        2 = PAD

    edge_type:
        0 = no edge
        1 = sequence
        2 = spatial

    orientation:
        0 = other
        1 = parallel
        2 = antiparallel
        3 = unknown
        4 = PAD / no edge
    """

    graph = build_sse_graph(
        ss_string,
        ca_pos,
        spatial_cutoff=spatial_cutoff,
        orientation_cutoff=orientation_cutoff,
    )

    K = int(graph["num_sse"])

    if K > max_sse:
        raise ValueError(
            f"Protein has {K} SSEs, "
            f"but max_sse={max_sse}"
        )

    # ============================================================
    # Node tensors
    # ============================================================

    sse_type = np.full(
        max_sse,
        2,
        dtype=np.int64,
    )

    sse_length = np.zeros(
        max_sse,
        dtype=np.int64,
    )

    linker_before = np.zeros(
        max_sse,
        dtype=np.int64,
    )

    sse_mask = np.zeros(
        max_sse,
        dtype=np.int64,
    )

    if K > 0:

        sse_type[:K] = graph[
            "sse_type"
        ]

        sse_length[:K] = graph[
            "sse_length"
        ]

        linker_before[:K] = graph[
            "linker_before"
        ]

        sse_mask[:K] = 1

    # ============================================================
    # Pair tensors
    # ============================================================

    sse_edge_type = np.zeros(
        (max_sse, max_sse),
        dtype=np.int64,
    )

    sse_spatial_contact = np.zeros(
        (max_sse, max_sse),
        dtype=np.int64,
    )

    # 4 = PAD / no edge
    sse_orientation = np.full(
        (max_sse, max_sse),
        4,
        dtype=np.int64,
    )

    sse_orientation_valid = np.zeros(
        (max_sse, max_sse),
        dtype=np.int64,
    )

    sse_pair_mask = np.zeros(
        (max_sse, max_sse),
        dtype=np.int64,
    )

    # Valid SSE pairs, excluding diagonal.
    if K > 0:

        sse_pair_mask[:K, :K] = 1

        np.fill_diagonal(
            sse_pair_mask,
            0,
        )

    # ============================================================
    # Populate edges
    # ============================================================

    edge_index = graph[
        "sse_edge_index"
    ]

    edge_type_raw = graph[
        "sse_edge_type"
    ]

    orientation_raw = graph[
        "sse_edge_orientation"
    ]

    orientation_valid_raw = graph[
        "sse_edge_orientation_valid"
    ]

    E = edge_index.shape[1]

    for e in range(E):

        i = int(
            edge_index[0, e]
        )

        j = int(
            edge_index[1, e]
        )

        # Raw:
        # 0 = sequence
        # 1 = spatial
        #
        # Padded:
        # 1 = sequence
        # 2 = spatial

        if edge_type_raw[e] == 0:
            sse_edge_type[i, j] = 1

        elif edge_type_raw[e] == 1:
            sse_edge_type[i, j] = 2
            sse_spatial_contact[i, j] = 1

        else:
            raise ValueError(
                "Unknown edge type: "
                f"{edge_type_raw[e]}"
            )

        sse_orientation[i, j] = (
            orientation_raw[e]
        )

        sse_orientation_valid[i, j] = int(
            orientation_valid_raw[e]
        )

    # ============================================================
    # residue -> SSE
    #
    # Keep -1 for coil.
    # ============================================================

    residue_to_sse = graph[
        "residue_to_sse"
    ].astype(np.int64)

    return {
        "num_sse": np.int64(K),

        "sse_type": sse_type,
        "sse_length": sse_length,
        "linker_before": linker_before,
        "sse_mask": sse_mask,

        "sse_edge_type": sse_edge_type,
        "sse_spatial_contact": sse_spatial_contact,
        "sse_orientation": sse_orientation,
        "sse_orientation_valid":
            sse_orientation_valid,
        "sse_pair_mask": sse_pair_mask,

        "residue_to_sse": residue_to_sse,
    }