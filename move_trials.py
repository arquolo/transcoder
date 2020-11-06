from argparse import ArgumentParser
from pathlib import Path

import glow  # noqa
import optuna
# import numpy as np
from tqdm import tqdm

p = ArgumentParser()
p.add_argument('src', type=Path)
p.add_argument('dst', type=Path)

ARGS = p.parse_args()

src = optuna.load_study(
    storage='sqlite:///optuna.db', study_name=ARGS.src.as_posix())
dst = optuna.create_study(
    storage='sqlite:///optuna.new.db', study_name=ARGS.dst.as_posix(),
    direction=src.direction.name.lower())

for t in tqdm(src.trials):
    if t.state.is_finished():
        # if 'fps' in t.user_attrs:
        #     fps = np.round(t.user_attrs['fps'], 1)
        #     ssim = np.round(t.user_attrs.get('ssim', 0), 3)
        #     t.value = 0.1 * np.log(fps) / np.log(10) + min(ssim, 30)

        # t.user_attrs['cli'] = ' '.join(sorted(
        #     f'--{px}' for p in t.user_attrs['cli'].split('--')
        #     if (px := p.strip())
        # ))

        # break
        dst.add_trial(t)
