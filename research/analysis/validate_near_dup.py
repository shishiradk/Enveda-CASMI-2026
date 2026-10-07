import json
data = json.load(open(r"D:\Enveda-CASMI-2026\results\exp004_novelty_audit.json"))
count = 0
for d in data:
    if d.get("max_sim_direct", 0) >= 0.90:
        count += 1
        print(f"rid {d['rid']}: max_sim_direct={d['max_sim_direct']}")
print(f"Total near duplicate exclusions (max_sim_direct >= 0.90): {count}")