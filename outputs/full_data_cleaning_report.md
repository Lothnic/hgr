# Full Data Cleaning Report

- Input file: `src/hgr/data/parallel.csv`
- Output file: `src/hgr/data/parallel.filtered.csv`
- Raw CSV lines (excluding header): **26785**
- Parseable rows: **25541**
- Malformed rows dropped at parse: **1244**

## Filtering summary
- Dropped empty pairs: **0**
- Dropped too-short pairs: **0**
- Dropped exact matches: **73**
- Dropped duplicates: **1613**
- Dropped length-ratio outliers: **2087**
- Dropped artifact-noise rows: **649**
- Dropped heavy-latin rows: **5**

## Final kept rows: **21114**