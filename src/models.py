import random

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import StandardScaler
from torch import nn


torch.set_num_threads(4)
MORGAN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
    includeChirality=True,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonical(smiles):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"Invalid or empty SMILES: {smiles}")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def graph(smiles):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"Invalid or empty SMILES: {smiles}")
    atom_features = []
    for atom in molecule.GetAtoms():
        atom_features.append(
            [
                atom.GetAtomicNum() / 100.0,
                atom.GetTotalDegree() / 6.0,
                atom.GetFormalCharge() / 4.0,
                atom.GetTotalNumHs() / 4.0,
                int(atom.GetIsAromatic()),
                atom.GetMass() / 200.0,
                float(atom.IsInRing()),
                int(atom.GetHybridization()) / 10.0,
                int(atom.GetChiralTag()) / 5.0,
                atom.GetTotalValence() / 8.0,
            ]
        )
    size = len(atom_features)
    adjacency = np.eye(size, dtype=np.float32)
    for bond in molecule.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0
    degree = adjacency.sum(axis=1)
    adjacency = adjacency / np.sqrt(degree[:, None] * degree[None, :])
    return np.asarray(atom_features, dtype=np.float32), adjacency


def featurize_smiles(smiles):
    fingerprints, maccs, descriptors, canonical_smiles, scaffolds = [], [], [], [], []
    for index, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None or molecule.GetNumAtoms() == 0:
            raise ValueError(f"Invalid SMILES at row {index}: {value!r}")
        canonical_value = Chem.MolToSmiles(molecule, isomericSmiles=True)
        canonical_smiles.append(canonical_value)
        scaffolds.append(
            MurckoScaffold.MurckoScaffoldSmiles(mol=molecule) or canonical_value
        )
        fingerprints.append(MORGAN.GetFingerprintAsNumPy(molecule))
        maccs.append(np.asarray(MACCSkeys.GenMACCSKeys(molecule)))
        descriptors.append(
            [
                Descriptors.MolWt(molecule),
                Crippen.MolLogP(molecule),
                rdMolDescriptors.CalcTPSA(molecule),
                rdMolDescriptors.CalcNumHBA(molecule),
                rdMolDescriptors.CalcNumHBD(molecule),
                rdMolDescriptors.CalcNumRotatableBonds(molecule),
                rdMolDescriptors.CalcNumAromaticRings(molecule),
                rdMolDescriptors.CalcNumAliphaticRings(molecule),
                rdMolDescriptors.CalcNumRings(molecule),
                rdMolDescriptors.CalcFractionCSP3(molecule),
                rdMolDescriptors.CalcNumHeavyAtoms(molecule),
                rdMolDescriptors.CalcNumHeteroatoms(molecule),
                Descriptors.NumValenceElectrons(molecule),
                Descriptors.MolMR(molecule),
                rdMolDescriptors.CalcNumAmideBonds(molecule),
                rdMolDescriptors.CalcNumAtomStereoCenters(molecule),
            ]
        )
    return {
        "fp2": np.asarray(fingerprints, dtype=np.float32),
        "maccs": np.asarray(maccs, dtype=np.float32),
        "desc": np.nan_to_num(
            np.asarray(descriptors, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        "canonical": np.asarray(canonical_smiles),
        "scaffold": np.asarray(scaffolds),
    }


def transform_features(raw, fit_indices):
    scaler = StandardScaler().fit(raw["desc"][fit_indices])
    descriptors = np.clip(scaler.transform(raw["desc"]), -8.0, 8.0).astype(
        np.float32
    )
    features = np.concatenate(
        [raw["fp2"], raw["maccs"], descriptors],
        axis=1,
    ).astype(np.float32)
    return features, scaler


def batch(graphs, features, indices):
    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0:
        raise ValueError("Cannot create an empty molecular batch")
    max_atoms = max(len(graphs[index][0]) for index in indices)
    atoms = np.zeros((len(indices), max_atoms, 10), dtype=np.float32)
    adjacency = np.zeros(
        (len(indices), max_atoms, max_atoms),
        dtype=np.float32,
    )
    mask = np.zeros((len(indices), max_atoms), dtype=np.float32)
    for batch_index, graph_index in enumerate(indices):
        atom_features, graph_adjacency = graphs[graph_index]
        atom_count = len(atom_features)
        atoms[batch_index, :atom_count] = atom_features
        adjacency[batch_index, :atom_count, :atom_count] = graph_adjacency
        mask[batch_index, :atom_count] = 1.0
    return (
        torch.from_numpy(atoms),
        torch.from_numpy(adjacency),
        torch.from_numpy(mask),
        torch.from_numpy(features[indices]),
    )


class GraphNet(nn.Module):
    def __init__(self, extra_dim):
        super().__init__()
        self.first = nn.Linear(10, 64)
        self.conv = nn.Linear(64, 64)
        self.head = nn.Sequential(
            nn.Linear(64 + extra_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, atoms, adjacency, mask, extra):
        nodes = torch.relu(self.first(atoms))
        nodes = torch.relu(self.conv(torch.bmm(adjacency, nodes)))
        pooled = (nodes * mask[:, :, None]).sum(dim=1)
        pooled = pooled / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.head(torch.cat([pooled, extra], dim=1)).squeeze(1)


def fit_model(
    graphs,
    features,
    target,
    train_indices,
    seed,
    task,
    epochs=80,
):
    if task not in {"classifier", "regressor"}:
        raise ValueError(f"Unsupported task: {task}")
    train_indices = np.asarray(train_indices, dtype=int)
    if len(train_indices) == 0:
        raise ValueError("Training indices must not be empty")
    set_seed(seed)
    model = GraphNet(features.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01,
    )
    target_tensor = torch.as_tensor(target, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    steps = max(1, int(np.ceil(len(train_indices) / 32)))
    for _ in range(epochs):
        model.train()
        shuffled = rng.permutation(train_indices)
        for indices in np.array_split(shuffled, steps):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(*batch(graphs, features, indices))
            if task == "classifier":
                loss = F.binary_cross_entropy_with_logits(
                    prediction,
                    target_tensor[indices],
                )
            else:
                loss = F.mse_loss(prediction, target_tensor[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {task} loss")
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=3.0,
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"Nonfinite {task} gradient")
            optimizer.step()
    return model


def predict(model, graphs, features, indices):
    model.eval()
    predictions = []
    with torch.no_grad():
        for ids in np.array_split(
            np.asarray(indices, dtype=int),
            max(1, int(np.ceil(len(indices) / 32))),
        ):
            predictions.extend(model(*batch(graphs, features, ids)).cpu().numpy())
    return np.asarray(predictions, dtype=np.float64)
