import subprocess
from pathlib import Path

ROOT = Path('D:/Encode')

FFMPEG = ROOT / 'bin/ffmpeg-n4.3.1-20-g8a2acdc6da-win64-gpl-4.3/ffmpeg.exe'
DATA = ROOT / 'xiph.org'
TXT = ROOT / 'xiph.org/list.txt'

for dir_ in DATA.iterdir():
    if not dir_.is_dir():
        continue

    TXT.write_text('\n'.join(
        f"file '{p.as_posix()}'" for p in dir_.glob('*.y4m')))

    subprocess.run(
        f(f'{FFMPEG} -y -r 30 -f concat -safe 0 -i {TXT.as_posix()}'
          f' {dir_.as_posix()}.y4m'),
        shell=True,
        check=True,
    )
    TXT.unlink()
