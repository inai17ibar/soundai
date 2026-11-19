"""Fetch a deterministic 20-category x 10-clip ESC-50 subset."""
import argparse
import csv
import io
import json
from pathlib import Path
import tempfile
import urllib.request
import zipfile

URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
LICENSE = "ESC-50: CC BY-NC 3.0, non-commercial personal evaluation only. Attribution: K. J. Piczak. https://github.com/karolpiczak/ESC-50\n"


def fetch(source, output, queries):
    targets = {q["category"] for q in json.loads(queries.read_text())}
    if len(targets) > 20:
        raise ValueError("At most 20 categories supported")
    archive = None
    if source.is_dir():
        meta = next(source.rglob("esc50.csv"))
        root = meta.parent.parent
        text = meta.read_text()
        read = lambda filename: (root / "audio" / filename).read_bytes()
    else:
        archive = zipfile.ZipFile(source)
        meta = next(n for n in archive.namelist() if n.endswith("/meta/esc50.csv"))
        root = meta.removesuffix("meta/esc50.csv")
        text = archive.read(meta).decode()
        read = lambda filename: archive.read(root + "audio/" + filename)
    try:
        reader = csv.DictReader(io.StringIO(text))
        fields = reader.fieldnames
        selected, counts = [], {}
        for row in sorted(reader, key=lambda r: r["filename"]):
            category = row["category"]
            if category in targets and counts.get(category, 0) < 10:
                if Path(row["filename"]).name != row["filename"] or Path(category).name != category:
                    raise ValueError("Invalid metadata path")
                dest = output / category / row["filename"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(read(row["filename"]))
                selected.append(row)
                counts[category] = counts.get(category, 0) + 1
        if any(counts.get(c, 0) != 10 for c in targets):
            raise ValueError("Source lacks 10 clips for every query category")
        with (output / "meta.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
        (output / "LICENSE.txt").write_text(LICENSE)
        print(f"Wrote {len(selected)} clips to {output}")
    finally:
        if archive:
            archive.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip-cache", type=Path, help="Existing ZIP or extracted ESC-50 directory")
    p.add_argument("--out", type=Path, default=Path("data/samples/esc50"))
    p.add_argument("--queries", type=Path, default=Path("data/hil_queries.json"))
    a = p.parse_args()
    if a.zip_cache:
        fetch(a.zip_cache, a.out, a.queries)
    else:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "esc50.zip"
            urllib.request.urlretrieve(URL, path)
            fetch(path, a.out, a.queries)


if __name__ == "__main__":
    main()
