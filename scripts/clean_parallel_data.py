import argparse
import re
from pathlib import Path

import pandas as pd


def norm_text(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = x.replace("\u200b", " ")
    x = re.sub(r"\s+", " ", x).strip().strip('"').strip("'")
    return x


def has_heavy_latin(s: str) -> bool:
    if not s:
        return False
    latin = len(re.findall(r"[A-Za-z]", s))
    return (latin / max(len(s), 1)) > 0.25


def has_artifact_noise(s: str) -> bool:
    # Conservative: catch obvious OCR/scrape artifacts, keep numerals generally.
    return bool(re.search(r"[{}]", s)) or bool(re.search(r"\.\.\.\.+", s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="src/hgr/data/parallel.csv")
    ap.add_argument("--output", default="src/hgr/data/parallel.filtered.csv")
    ap.add_argument("--report", default="outputs/full_data_cleaning_report.md")
    ap.add_argument("--min_chars", type=int, default=2)
    ap.add_argument("--max_len_ratio", type=float, default=2.5)
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    report = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    raw_lines = sum(1 for _ in inp.open("r", encoding="utf-8")) - 1

    df = pd.read_csv(inp, encoding="utf-8", engine="python", on_bad_lines="skip")
    initial_rows = len(df)

    assert "src" in df.columns and "tgt" in df.columns, f"Missing src/tgt columns: {df.columns.tolist()}"

    df["src"] = df["src"].map(norm_text)
    df["tgt"] = df["tgt"].map(norm_text)

    stats = {}

    before = len(df)
    df = df[(df["src"] != "") & (df["tgt"] != "")]
    stats["drop_empty"] = before - len(df)

    before = len(df)
    df = df[df["src"].str.len() >= args.min_chars]
    df = df[df["tgt"].str.len() >= args.min_chars]
    stats["drop_too_short"] = before - len(df)

    before = len(df)
    df = df[df["src"] != df["tgt"]]
    stats["drop_exact_match"] = before - len(df)

    before = len(df)
    df = df.drop_duplicates(subset=["src", "tgt"]).reset_index(drop=True)
    stats["drop_duplicates"] = before - len(df)

    before = len(df)
    ratio = (df["src"].str.len() + 1) / (df["tgt"].str.len() + 1)
    df = df[(ratio <= args.max_len_ratio) & (ratio >= 1 / args.max_len_ratio)]
    stats["drop_len_ratio_outliers"] = before - len(df)

    before = len(df)
    noise_mask = df["src"].map(has_artifact_noise) | df["tgt"].map(has_artifact_noise)
    df = df[~noise_mask]
    stats["drop_artifact_noise"] = before - len(df)

    before = len(df)
    latin_mask = df["src"].map(has_heavy_latin) | df["tgt"].map(has_heavy_latin)
    df = df[~latin_mask]
    stats["drop_heavy_latin"] = before - len(df)

    final_rows = len(df)
    df.to_csv(out, index=False, encoding="utf-8")

    malformed_dropped = raw_lines - initial_rows

    report.write_text(
        "\n".join(
            [
                "# Full Data Cleaning Report",
                "",
                f"- Input file: `{inp}`",
                f"- Output file: `{out}`",
                f"- Raw CSV lines (excluding header): **{raw_lines}**",
                f"- Parseable rows: **{initial_rows}**",
                f"- Malformed rows dropped at parse: **{malformed_dropped}**",
                "",
                "## Filtering summary",
                f"- Dropped empty pairs: **{stats['drop_empty']}**",
                f"- Dropped too-short pairs: **{stats['drop_too_short']}**",
                f"- Dropped exact matches: **{stats['drop_exact_match']}**",
                f"- Dropped duplicates: **{stats['drop_duplicates']}**",
                f"- Dropped length-ratio outliers: **{stats['drop_len_ratio_outliers']}**",
                f"- Dropped artifact-noise rows: **{stats['drop_artifact_noise']}**",
                f"- Dropped heavy-latin rows: **{stats['drop_heavy_latin']}**",
                "",
                f"## Final kept rows: **{final_rows}**",
            ]
        ),
        encoding="utf-8",
    )

    print(f"WROTE {out} rows={final_rows}")
    print(f"WROTE {report}")


if __name__ == "__main__":
    main()
