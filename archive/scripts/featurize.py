from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train_clean.csv"
TEST_PATH = ROOT / "test_clean.csv"
N_BITS = 2048
FINGERPRINT_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=N_BITS,
)


def featurize_smiles(smiles_values, descriptor_names, warning_file, log_mode):
    descriptor_rows = []
    fingerprint_rows = []
    parsed_count = 0

    with warning_file.open(log_mode, encoding="utf-8") as log:
        for row_index, smiles in enumerate(smiles_values):
            molecule = None
            if isinstance(smiles, str) and smiles:
                try:
                    molecule = Chem.MolFromSmiles(smiles)
                except Exception as error:
                    log.write(f"row={row_index}\tsmiles={smiles!r}\terror={error}\n")

            if molecule is None:
                if not isinstance(smiles, str) or not smiles:
                    log.write(f"row={row_index}\tsmiles={smiles!r}\terror=empty SMILES\n")
                descriptor_rows.append([np.nan] * len(descriptor_names))
                fingerprint_rows.append(np.zeros(N_BITS, dtype=np.uint8))
                continue

            parsed_count += 1
            try:
                descriptor_row = [
                    Descriptors.descList[i][1](molecule)
                    for i in range(len(descriptor_names))
                ]
            except Exception as error:
                log.write(f"row={row_index}\tsmiles={smiles!r}\terror={error}\n")
                descriptor_row = [np.nan] * len(descriptor_names)
            try:
                fingerprint = FINGERPRINT_GENERATOR.GetFingerprintAsNumPy(
                    molecule
                ).astype(np.uint8)
            except Exception as error:
                log.write(f"row={row_index}\tsmiles={smiles!r}\tfingerprint_error={error}\n")
                fingerprint = np.zeros(N_BITS, dtype=np.uint8)
            descriptor_rows.append(descriptor_row)
            fingerprint_rows.append(fingerprint)

    descriptors = pd.DataFrame(descriptor_rows, columns=descriptor_names)
    fingerprints = np.asarray(fingerprint_rows, dtype=np.uint8)
    return descriptors, fingerprints, parsed_count


def main():
    if not TRAIN_PATH.is_file() or not TEST_PATH.is_file():
        raise FileNotFoundError(
            f"Expected {TRAIN_PATH.name} and {TEST_PATH.name} beside featurize.py"
        )

    RDLogger.DisableLog("rdApp.error")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    descriptor_names = [name for name, _ in Descriptors.descList]
    warning_file = ROOT / "warnings.log"

    train_desc, train_fp, train_parsed = featurize_smiles(
        train["SMILES"], descriptor_names, warning_file, "w"
    )
    test_desc, test_fp, test_parsed = featurize_smiles(
        test["SMILES"], descriptor_names, warning_file, "a"
    )

    train_desc.to_csv(ROOT / "featurestraindesc.csv", index=False)
    np.savez_compressed(ROOT / "featurestrainfp.npz", fingerprints=train_fp)
    test_desc.to_csv(ROOT / "featurestestdesc.csv", index=False)
    np.savez_compressed(ROOT / "featurestestfp.npz", fingerprints=test_fp)

    train_index = train["index"] if "index" in train.columns else pd.Series(
        np.arange(len(train)), index=train.index
    )
    train_meta = pd.DataFrame(
        {
            "index": train_index.to_numpy(),
            "SMILES": train["SMILES"].to_numpy(),
            "isactive": train["is_active"].to_numpy(),
            "log10SI": train["log10_SI"].to_numpy(),
        }
    )
    test_index = test["index"] if "index" in test.columns else pd.Series(
        np.arange(len(test)), index=test.index
    )
    test_meta = pd.DataFrame(
        {
            "index": test_index.to_numpy(),
            "SMILES": test["SMILES"].to_numpy(),
        }
    )
    train_meta.to_csv(ROOT / "featurestrainmeta.csv", index=False)
    test_meta.to_csv(ROOT / "featurestestmeta.csv", index=False)

    print(f"Train descriptors: {train_desc.shape}; fingerprints: {train_fp.shape}")
    print(f"Test descriptors:  {test_desc.shape}; fingerprints: {test_fp.shape}")
    print(f"Successfully parsed SMILES: train {train_parsed}/{len(train)}, test {test_parsed}/{len(test)}")
    if warning_file.stat().st_size:
        print(f"SMILES/featurization warnings logged to {warning_file.name}")


if __name__ == "__main__":
    main()
