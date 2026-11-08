"""Organism-wide public collection; never queries competition compound IDs."""
from data_collection.chembl import collect

def download(output,max_pages=30):
    return collect(output,max_pages)
