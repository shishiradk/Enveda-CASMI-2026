import os, json, tempfile
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

datasets_info = [
    {
        'slug': 'casmi-fp-models-v2',
        'title': 'casmi-fp-models-v2',
        'license': 'CC0-1.0',
        'description': 'Analog-propagation FP ranker neural network weights for CASMI 2026. Released under Creative Commons Zero 1.0 Universal (CC0 1.0).'
    },
    {
        'slug': 'casmi-sim-rows',
        'title': 'casmi-sim-rows',
        'license': 'CC0-1.0',
        'description': 'Simulated ranker rows for CASMI 2026 analog-propagation ranker training. Released under CC0 1.0 Universal.'
    },
    {
        'slug': 'casmi-coco-candidates',
        'title': 'casmi-coco-candidates',
        'license': 'CC-BY-4.0',
        'description': 'COCONUT (Collection of Open Natural Products) candidate pool fingerprints and masses for CASMI 2026. Data source: COCONUT database, licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).'
    },
    {
        'slug': 'casmi-bio-clean',
        'title': 'casmi-bio-clean',
        'license': 'CC-BY-4.0',
        'description': 'Natural product structures and fingerprints from ChEBI and LIPID MAPS for CASMI 2026. Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).'
    },
    {
        'slug': 'casmi-fm-runner',
        'title': 'casmi-fm-runner',
        'license': 'mit',
        'description': 'MassSpecGym and ms-pred forward modeling candidate re-ranker runner for CASMI 2026. Code under MIT License, model weights under MassSpecGym open terms.'
    },
    {
        'slug': 'casmi-v2-pubchem',
        'title': 'casmi-v2-pubchem',
        'license': 'CC0-1.0',
        'description': 'PubChem candidate indexes and structure representations for CASMI 2026. Source: NCBI PubChem (Public Domain / CC0 1.0).'
    },
    {
        'slug': 'casmi-e6-pcnets',
        'title': 'casmi-e6-pcnets',
        'license': 'mit',
        'description': 'PubChem candidate neural network models for CASMI 2026. MIT License.'
    },
    {
        'slug': 'casmi-e7-c3assets',
        'title': 'casmi-e7-c3assets',
        'license': 'CC-BY-4.0',
        'description': 'Class-3 biotransformation and matched molecular pair (MMP) rules and candidates for CASMI 2026. Derived from COCONUT (CC BY 4.0) and competition data; code under MIT License.'
    },
    {
        'slug': 'casmi-rdkit2025-cp313',
        'title': 'casmi-rdkit2025-cp313',
        'license': 'other',
        'description': 'RDKit 2025.3.6 wheel for Python 3.13. BSD 3-Clause License.'
    }
]

for info in datasets_info:
    slug = info['slug']
    ref = f"shishiradhikari11/{slug}"
    lic = info['license']
    tmpdir = tempfile.mkdtemp()
    meta_path = os.path.join(tmpdir, 'dataset-metadata.json')
    meta = {
        'title': info['title'],
        'id': ref,
        'licenses': [{'name': lic}],
        'description': info['description'],
        'isPrivate': False
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    try:
        api.dataset_metadata_update(ref, tmpdir)
        print(f"Successfully updated {ref} to PUBLIC with license {lic}")
    except Exception as e:
        print(f"Error updating {ref}: {e}")
