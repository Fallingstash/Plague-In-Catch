import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondStereo, BondType, HybridizationType
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parent
TRAIN_PATHS = (ROOT / "trainclean.csv", ROOT / "train_clean.csv")
TEST_PATHS = (ROOT / "testclean.csv", ROOT / "test_clean.csv")
FOLDS_PATH = ROOT / "experiments" / "results" / "folds.json"
WARNINGS_PATH = ROOT / "warnings.log"

MAX_ATOMIC_NUMBER = 35
ATOMIC_NUMBER_CATEGORIES = tuple(range(1, MAX_ATOMIC_NUMBER + 1)) + ("other",)
HYBRIDIZATION_CATEGORIES = (
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
    HybridizationType.UNSPECIFIED,
    "other",
)
BOND_DEGREE_CATEGORIES = (0, 1, 2, 3, 4, 5)
FORMAL_CHARGE_CATEGORIES = (-3, -2, -1, 0, 1, 2, 3)
BOND_TYPE_CATEGORIES = (
    BondType.SINGLE,
    BondType.DOUBLE,
    BondType.TRIPLE,
    BondType.AROMATIC,
    "other",
)
BOND_STEREO_CATEGORIES = (
    BondStereo.STEREONONE,
    BondStereo.STEREOANY,
    BondStereo.STEREOZ,
    BondStereo.STEREOE,
    BondStereo.STEREOCIS,
    BondStereo.STEREOTRANS,
    "other",
)


def existing_path(candidates):
    for path in candidates:
        if path.is_file():
            return path
    expected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find any of: {expected}")


def one_hot(value, categories):
    result = np.zeros(len(categories), dtype=np.float32)
    try:
        index = categories.index(value)
    except ValueError:
        index = len(categories) - 1
    result[index] = 1.0
    return result


def atom_features(atom):
    atomic_number = atom.GetAtomicNum()
    atomic_category = (
        atomic_number if atomic_number in ATOMIC_NUMBER_CATEGORIES[:-1] else "other"
    )
    hybridization = atom.GetHybridization()
    if hybridization not in HYBRIDIZATION_CATEGORIES[:-1]:
        hybridization = "other"
    degree = min(atom.GetTotalDegree(), BOND_DEGREE_CATEGORIES[-1])
    formal_charge = int(np.clip(atom.GetFormalCharge(), -3, 3))
    return np.concatenate(
        [
            one_hot(atomic_category, ATOMIC_NUMBER_CATEGORIES),
            one_hot(hybridization, HYBRIDIZATION_CATEGORIES),
            one_hot(degree, BOND_DEGREE_CATEGORIES),
            one_hot(formal_charge, FORMAL_CHARGE_CATEGORIES),
            np.asarray([float(atom.GetIsAromatic())], dtype=np.float32),
        ]
    )


def bond_features(bond):
    bond_type = bond.GetBondType()
    if bond_type not in BOND_TYPE_CATEGORIES[:-1]:
        bond_type = "other"
    stereo = bond.GetStereo()
    if stereo not in BOND_STEREO_CATEGORIES[:-1]:
        stereo = "other"
    return np.concatenate(
        [
            one_hot(bond_type, BOND_TYPE_CATEGORIES),
            one_hot(stereo, BOND_STEREO_CATEGORIES),
        ]
    )


def make_graph(row, graph_index, fold_id=None, target_columns=None, warnings=None):
    smiles = row["SMILES"]
    try:
        molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    except Exception as error:
        molecule = None
        if warnings is not None:
            warnings.write(
                f"graph={graph_index}\tsmiles={smiles!r}\tparse_error={error}\n"
            )
    if molecule is None:
        if warnings is not None:
            warnings.write(f"graph={graph_index}\tsmiles={smiles!r}\terror=invalid SMILES\n")
        raise ValueError(f"Invalid SMILES in row {graph_index}: {smiles!r}")

    x = np.asarray([atom_features(atom) for atom in molecule.GetAtoms()], dtype=np.float32)
    edge_pairs = []
    edge_attributes = []
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        features = bond_features(bond)
        edge_pairs.extend(([begin, end], [end, begin]))
        edge_attributes.extend((features, features))

    if edge_pairs:
        edge_index = np.asarray(edge_pairs, dtype=np.int64).T
        edge_attr = np.asarray(edge_attributes, dtype=np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, len(BOND_TYPE_CATEGORIES) + len(BOND_STEREO_CATEGORIES)), dtype=np.float32)

    graph = Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
        num_nodes=int(molecule.GetNumAtoms()),
        graph_index=int(graph_index),
        smiles=smiles,
    )
    if fold_id is not None:
        graph.fold_id = int(fold_id)
    if target_columns:
        for column, attribute in target_columns.items():
            value = row.get(column)
            if value is not None and pd.notna(value):
                setattr(graph, attribute, torch.tensor([float(value)], dtype=torch.float32))
    return graph


def fold_assignments(folds, row_count):
    assignment = np.full(row_count, -1, dtype=np.int64)
    expected = set(range(row_count))
    seen_validation = []
    for fold in folds:
        train_indices = set(fold["train_indices"])
        valid_indices = set(fold["validation_indices"])
        if train_indices & valid_indices or train_indices | valid_indices != expected:
            raise ValueError(f"Fold {fold.get('fold')} does not partition train rows")
        valid = np.asarray(fold["validation_indices"], dtype=np.int64)
        if (assignment[valid] != -1).any():
            raise ValueError("A training row appears in multiple validation folds")
        assignment[valid] = int(fold["fold"])
        seen_validation.extend(fold["validation_indices"])
    if sorted(seen_validation) != list(range(row_count)) or (assignment < 0).any():
        raise ValueError("Saved folds must assign every training row exactly once")
    return assignment


def main():
    train_path = existing_path(TRAIN_PATHS)
    test_path = existing_path(TEST_PATHS)
    for path in (FOLDS_PATH,):
        if not path.is_file():
            raise FileNotFoundError(f"Required fold file is missing: {path}")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    fold_data = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))
    if len(train) != 318 or len(test) != 597:
        raise ValueError(f"Expected train/test sizes 318/597, got {len(train)}/{len(test)}")
    if "SMILES" not in train or "SMILES" not in test or "index" not in test:
        raise ValueError("Train/test must contain SMILES and test must contain index")
    assignments = fold_assignments(fold_data["folds"], len(train))
    target_columns = {}
    if "SI VACV" in train.columns:
        target_columns["SI VACV"] = "y_si"
    if "is_active" in train.columns:
        target_columns["is_active"] = "y_active"
    if "log10_SI" in train.columns:
        target_columns["log10_SI"] = "y_log10_si"

    WARNINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with WARNINGS_PATH.open("a", encoding="utf-8") as warnings:
        train_graphs = [
            make_graph(
                row,
                graph_index,
                fold_id=int(assignments[graph_index]),
                target_columns=target_columns,
                warnings=warnings,
            )
            for graph_index, (_, row) in enumerate(train.iterrows())
        ]
        test_graphs = [
            make_graph(row, graph_index, warnings=warnings)
            for graph_index, (_, row) in enumerate(test.iterrows())
        ]

    torch.save(train_graphs, ROOT / "train_graphs.pt")
    torch.save(test_graphs, ROOT / "test_graphs.pt")
    pd.DataFrame({"index": test["index"].to_numpy()}).to_csv(
        ROOT / "testindices.csv", index=False
    )
    fold_manifest = {
        "source": "experiments/results/folds.json",
        "n_splits": int(fold_data["n_splits"]),
        "random_state": int(fold_data["random_state"]),
        "fold_ids": assignments.tolist(),
        "folds": fold_data["folds"],
    }
    (ROOT / "train_graphs_folds.json").write_text(
        json.dumps(fold_manifest, indent=2), encoding="utf-8"
    )

    node_counts = np.asarray([graph.num_nodes for graph in train_graphs + test_graphs])
    edge_counts = np.asarray([graph.edge_index.shape[1] for graph in train_graphs + test_graphs])
    print(f"Train graphs: {len(train_graphs)}")
    print(f"Test graphs: {len(test_graphs)}")
    print(f"Mean nodes per graph: {node_counts.mean():.2f}")
    print(
        f"Mean directed edges per graph: {edge_counts.mean():.2f} "
        "(each chemical bond is stored in both directions)"
    )
    print(f"Node feature width: {train_graphs[0].x.shape[1]}")
    print(f"Edge feature width: {train_graphs[0].edge_attr.shape[1]}")
    print(f"Fold graph counts: {np.bincount(assignments, minlength=5).tolist()}")
    print(f"First graph:\n{train_graphs[0]}")


if __name__ == "__main__":
    main()
