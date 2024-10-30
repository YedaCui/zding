import os

import pandas as pd

times = {}
scores = {}
for filename in os.listdir("log"):
    if not filename.endswith(".txt"):
        continue
    name = filename[:-4]
    with open(f"log/{filename}", "r") as fin:
        for line in fin:
            if not line.startswith("case"):
                continue
            items = line.strip().split()
            times[name, int(items[1])] = float(items[2])
            scores[name, int(items[1])] = float(items[3])
scores = pd.Series(scores).unstack().rank(pct=True, ascending=False)
df = pd.DataFrame({
    "time": pd.Series(times).unstack().sum(axis=1),
    "finish": pd.Series(times).unstack().count(axis=1),
    "#best": (scores == scores.max(axis=0)).sum(axis=1),
    "score": scores.sum(axis=1),
})
df = round(df, 2)
print(df)
