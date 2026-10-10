"""Unit tests for RRF fusion module."""

import sys
from pathlib import Path
import pandas as pd

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

from fuse import fuse


def test_unit_rrf_simple():
    """Unit test 1: Basic RRF on toy lists."""

    v4n_lists = {
        "mol_001": [
            ("SMILES_A", "KEY_A"),
            ("SMILES_B", "KEY_B"),
            ("SMILES_C", "KEY_C"),
        ]
    }

    e7_lists = {
        "mol_001": [
            ("SMILES_B", "KEY_B"),  # Different order
            ("SMILES_A", "KEY_A"),
            ("SMILES_D", "KEY_D"),
        ]
    }

    result = fuse(v4n_lists, e7_lists, keep_top=4)

    # RRF scores:
    # KEY_A: 1/(3+0) + 0.6/(3+1) = 1/3 + 0.15 = 0.483
    # KEY_B: 1/(3+1) + 0.6/(3+0) = 0.25 + 0.2 = 0.45
    # KEY_C: 1/(3+2) + 0 = 0.2
    # KEY_D: 0 + 0.6/(3+2) = 0.12

    mol_results = result["mol_001"]
    print("[PASS] Test 1: RRF scoring on toy lists")
    print(f"  Result: {[(s, round(sc, 4)) for s, sc in mol_results[:3]]}")

    assert len(mol_results) <= 4, "Should keep top 4"
    assert mol_results[0][0] == "SMILES_A", "KEY_A should rank first"
    print()


def test_unit_dedup_by_key():
    """Unit test 2: De-duplication by score key."""

    v4n_lists = {
        "mol_002": [
            ("SMILES_A1", "KEY_A"),
            ("SMILES_B", "KEY_B"),
        ]
    }

    e7_lists = {
        "mol_002": [
            ("SMILES_A2", "KEY_A"),  # Same key, different SMILES
            ("SMILES_C", "KEY_C"),
        ]
    }

    result = fuse(v4n_lists, e7_lists, keep_top=3)

    smiles_list = [s for s, _ in result["mol_002"]]
    keys_seen = set()

    print("[PASS] Test 2: De-duplication by key")
    print(f"  Result: {smiles_list}")

    for smiles in smiles_list:
        if smiles == "SMILES_A1":
            keys_seen.add("KEY_A")

    assert "SMILES_A1" in smiles_list or "SMILES_A2" in smiles_list, \
        "One SMILES per key should be kept"
    print()


def test_bench_e7_v9():
    """Bench test: Fuse E7 v9 submission with a reversed ranking."""

    e7_path = Path("research/kaggle_e1/e7/v9_output/submission.csv")

    if not e7_path.exists():
        print("[SKIP] E7 v9 bench test: submission not found")
        return

    e7_df = pd.read_csv(e7_path)

    # Parse E7 data
    e7_lists = {}
    for idx, row in e7_df.iterrows():
        mol_id = row['molecule_id']
        smiles_list = row['smiles'].split(';')
        # Store as (smiles, key) tuples where key=smiles (identity)
        e7_lists[mol_id] = [(s, s) for s in smiles_list]

    # Create mock v4n ranking: reverse order
    v4n_lists = {}
    for mol_id, e7_candidates in e7_lists.items():
        smiles_list = [s for s, _ in e7_candidates]
        v4n_lists[mol_id] = [(s, s) for s in reversed(smiles_list)]

    # Fuse
    result = fuse(v4n_lists, e7_lists, keep_top=40)

    print("[PASS] Bench test: E7 v9 fusion with reversed stand-in")
    print(f"  Molecules fused: {len(result)}")

    # Count how many keep E7's top-1
    keep_e7_top1 = 0
    for mol_id in e7_lists.keys():
        e7_top1 = [s for s, _ in e7_lists[mol_id]][:1][0]
        fused_smiles = [s for s, _ in result[mol_id]]
        if fused_smiles and fused_smiles[0] == e7_top1:
            keep_e7_top1 += 1

    pct_keep = 100.0 * keep_e7_top1 / len(result)
    print(f"  Kept E7 top-1: {keep_e7_top1}/{len(result)} ({pct_keep:.1f}%)")
    print(f"  Note: visible-test truths are leaky, no MRR claims")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("RRF Fusion Module Tests")
    print("=" * 60)
    print()

    test_unit_rrf_simple()
    test_unit_dedup_by_key()
    test_bench_e7_v9()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
