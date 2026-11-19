"""Package the standalone local prototype and verify each included file."""
import argparse
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = [
    'README.md', 'LICENSE', 'Avvia_demo.cmd', 'confeziona.py', 'demo.py',
    'macchina.py', 'trasporto.py', 'trame.py', 'verifica.py',
    'test_macchina.py', 'test_trasporto.py', 'test_trame.py',
    'docs/laboratorio-wow.md', 'docs/modello-ai.md',
    'Avvia_demo_AI.cmd', 'demo_ai.py', 'modello_ai.py', 'addestra_ai.py',
    'test_modello_ai.py', 'test_demo_ai.py', 'models/cifre.json',
    'models/cifre-addestramento.json', 'models/cifre-esempi.json',
    'radio.py', 'radio_phy.py', 'radio_medie.py', 'test_radio.py',
    'requirements-radio.txt', 'docs/radio.md',
    'confronta_radio.py', 'radio_compatto.py', 'test_radio_compatto.py',
]


def run(archive_path=None):
    files = [ROOT / name for name in SOURCE_FILES]
    if any(not path.is_file() or not path.resolve().is_relative_to(ROOT) for path in files):
        raise ValueError('Missing or external source file in release list')
    hashes = {ROOT.name+'/'+path.relative_to(ROOT).as_posix():
              hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    manifest = ROOT/'MANIFEST_SHA256.txt'
    manifest.write_text(''.join(f'{digest}  {name}\n' for name,digest in hashes.items()),encoding='utf-8')
    archive_path = archive_path or ROOT.parent/'Seme_Interstellare.zip'
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in files+[manifest]:
            archive.write(path,ROOT.name+'/'+path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        for name,digest in hashes.items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==digest
    print(f'Pacchetto verificato: {len(files)+1} file, {archive_path.stat().st_size} byte')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    run(parser.parse_args().output)
