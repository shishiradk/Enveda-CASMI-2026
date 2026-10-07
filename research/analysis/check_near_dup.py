import json
data = json.load(open(r"D:\Enveda-CASMI-2026\results\exp004_novelty_audit.json"))
# Get all max_sim_direct values
vals = [(d["rid"], d.get("max_sim_direct", 0)) for d in data if d.get("max_sim_direct") is not None]
vals.sort(key=lambda x: x[1], reverse=True)
print("Top 10 max_sim_direct:")
for rid, val in vals[:10]:
    print(f"rid {rid}: {val}")
print("\nAll values >= 0.90:")
for rid, val in vals:
    if val >= 0.90:
        print(f"rid {rid}: {val}")
print(f"Count >= 0.90: {sum(1 for _, v in vals if v >= 0.90)}")
print(f"Count > 0.90: {sum(1 for _, v in vals if v > 0.90)}")
print(f"Count == 0.90: {sum(1 for _, v in vals if v == 0.90)}")