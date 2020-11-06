"""Concatenates *.y4m files into single one per folder"""
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

p = ArgumentParser('Reduce encoding overhead via concatenation of *.y4m files')
p.add_argument('root', type=Path, help='Location of folders with *.y4m files.')

ROOT = p.parse_args().root
TXT = ROOT / 'list.txt'

FFMPEG = (
    next(Path(__file__).parent.rglob('ffmpeg.exe'))
    if sys.platform == 'win32' else 'ffmpeg')

for folder in ROOT.iterdir():
    if not folder.is_dir():
        continue

    TXT.write_text('\n'.join(
        f"file '{p.as_posix()}'" for p in folder.glob('*.y4m')))

    subprocess.run(
        (f'{FFMPEG} -y -r 30 -f concat -safe 0 -i {TXT.as_posix()}'
         f' {folder.as_posix()}.y4m'),
        shell=True,
        check=True,
    )
    TXT.unlink()
