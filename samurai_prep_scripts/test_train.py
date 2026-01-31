#!/usr/bin/env python3
from pathlib import Path

root = Path(".")  # run inside LaSOT folder
seqs = sorted([p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")])

(root / "training_set.txt").write_text("\n".join(seqs) + "\n")
(root / "testing_set.txt").write_text("\n".join(seqs) + "\n")

print(f"Wrote {len(seqs)} entries to training_set.txt and testing_set.txt")
