import json
from collections import defaultdict

# 準備一個二層 dict：dist[category][label] = count
dist = defaultdict(lambda: defaultdict(int))

with open("logs.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        cat   = rec.get("category", "Unknown")
        label = rec.get("label",    "Unknown")
        dist[cat][label] += 1

# 如果你想要把結果寫成 distribution.json
import os
if os.path.exists("distribution.json"):
    os.remove("distribution.json")
with open("distribution.json", "w", encoding="utf-8") as fout:
    json.dump(dist, fout, ensure_ascii=False, indent=2)

# 或者直接印出來看看
for cat, counter in dist.items():
    print(f"{cat}:")
    for label, cnt in counter.items():
        print(f"  {label}: {cnt}")