import csv
import json
import zipfile
from scripts.fetch_esc50_subset import fetch


def test_cached_archive_and_directory(tmp_path):
    root = tmp_path / 'ESC-50-master'
    (root / 'meta').mkdir(parents=True)
    (root / 'audio').mkdir()
    with (root / 'meta/esc50.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'category'])
        for i in range(12):
            name = f'{i:02}.wav'
            writer.writerow([name, 'dog'])
            (root / 'audio' / name).write_bytes(b'RIFFtest')
    queries = tmp_path / 'queries.json'
    queries.write_text(json.dumps([{'query': 'bark', 'category': 'dog'}]))
    archive = tmp_path / 'esc.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for path in root.rglob('*'):
            if path.is_file():
                z.write(path, path.relative_to(tmp_path))
    for i, source in enumerate([root, archive]):
        output = tmp_path / f'out{i}'
        fetch(source, output, queries)
        assert len(list(output.rglob('*.wav'))) == 10
        assert len(list(csv.DictReader((output / 'meta.csv').open()))) == 10
        assert 'Piczak' in (output / 'LICENSE.txt').read_text()
