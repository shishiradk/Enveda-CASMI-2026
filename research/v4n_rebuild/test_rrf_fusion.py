"""Test reciprocal-rank fusion between v4n and E7 submissions."""

import pandas as pd
import numpy as np
from pathlib import Path

# Load E7 v9 submission
e7_path = Path("research/kaggle_e1/e7/v9_output/submission.csv")
e7_df = pd.read_csv(e7_path)

print(f"E7 submission shape: {e7_df.shape}")
print(f"E7 first row: {e7_df.iloc[0]}")
print()

# Parse E7 data: molecule_id and space-separated SMILES
e7_data = {}
for idx, row in e7_df.iterrows():
    mol_id = row['molecule_id']
    smiles_list = row['smiles'].split(';')  # Separated by semicolon, not space
    e7_data[mol_id] = smiles_list
    if idx < 2:
        print(f"E7 mol {mol_id}: {len(smiles_list)} candidates")

# For testing, create a mock v4n submission with different ranking
# In real scenario, this would be the actual v4n engine output
mock_v4n_data = {}
for mol_id, e7_smiles in e7_data.items():
    # Create mock v4n ranking: reverse order (different from E7)
    mock_v4n_data[mol_id] = list(reversed(e7_smiles))
    if mol_id == list(e7_data.keys())[0]:
        print(f"Mock v4n mol {mol_id}: reversed ranking")

print()

# RRF fusion: 1/(3 + r_v4n) + 0.6/(3 + r_e7)
# Lower rank index = higher in the list = higher priority
rrf_results = {}

for mol_id in e7_data.keys():
    e7_smiles_list = e7_data[mol_id]
    v4n_smiles_list = mock_v4n_data[mol_id]

    # Build ranking dictionaries
    e7_ranks = {smiles: rank for rank, smiles in enumerate(e7_smiles_list)}
    v4n_ranks = {smiles: rank for rank, smiles in enumerate(v4n_smiles_list)}

    # Compute RRF scores for all unique candidates
    all_candidates = set(e7_smiles_list) | set(v4n_smiles_list)
    rrf_scores = {}

    for smiles in all_candidates:
        score = 0.0
        if smiles in v4n_ranks:
            score += 1.0 / (3 + v4n_ranks[smiles])
        if smiles in e7_ranks:
            score += 0.6 / (3 + e7_ranks[smiles])
        rrf_scores[smiles] = score

    # Sort by RRF score (descending) and keep top 60
    sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    top_60 = [smiles for smiles, score in sorted_candidates[:60]]
    rrf_results[mol_id] = top_60

print(f"RRF results for first molecule:")
mol_id = list(rrf_results.keys())[0]
print(f"  Molecule: {mol_id}")
print(f"  Top 5 candidates: {rrf_results[mol_id][:5]}")
print(f"  Total kept: {len(rrf_results[mol_id])}")
print()

# Verify all molecules have 60 or fewer candidates
valid_count = sum(1 for smiles_list in rrf_results.values() if len(smiles_list) <= 60)
print(f"Valid molecules (≤60 candidates): {valid_count}/{len(rrf_results)}")

# Create submission format
submission_df = pd.DataFrame({
    'molecule_id': list(rrf_results.keys()),
    'smiles': [';'.join(rrf_results[mol_id]) for mol_id in rrf_results.keys()]
})

print(f"\nFinal submission shape: {submission_df.shape}")
print(f"First row:\n{submission_df.iloc[0]}")

# Verify row count is 400
assert len(submission_df) == 400, f"Expected 400 rows, got {len(submission_df)}"
print("\n✓ RRF fusion test passed: 400 molecules with proper format")
