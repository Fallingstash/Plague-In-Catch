# Scientific records, isolated from machine learning

This module reuses the real ChEMBL organism-wide collector in `data_collection/chembl.py`. It queries VACV assays rather than competition compounds, retains raw responses, DOI and URL provenance, and exports an independently reviewed scientific dataset. All rows have `training_allowed=False`. The competition loader accepts only a SHA256-pinned official train, so these files cannot be substituted for competition labels.

Run `python -m research_data_collection collect` for a new download or `python -m research_data_collection export` for the existing cached measurements. Exported pairs are candidates until the original paper confirms assay compatibility. No fabricated or automatically approved records are added. `validated_measurements.csv` is empty if no primary-source review has been completed.

Sources: https://www.ebi.ac.uk/chembl/api/data/ and https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest .
