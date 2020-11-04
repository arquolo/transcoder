#!/usr/bin/python3
"""
Example usage:
    python show.py -n 20 --mask "aq-mode=3:tskip" foldername
    python show.py --mask "aq-mode=3" foldername
    python show.py foldername
"""

import textwrap
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker
import optuna


COLUMNS = ('cli', 'crf', 'fps', 'ssim')

parser = ArgumentParser()
parser.add_argument('-n', '--count', default=5, type=int)
parser.add_argument('--mask', default='')
parser.add_argument('folder', type=Path)
ARGS = parser.parse_args()

study = optuna.load_study(ARGS.folder.as_posix(), 'sqlite:///optuna.db')

df = study.trials_dataframe(('value', 'user_attrs', 'state'))
df = df.rename(columns={f'user_attrs_{k}': k for k in COLUMNS})

print(f'Total trials: {len(df)}')
print(df.state.value_counts().to_string())

df = df[df.state == 'COMPLETE'].drop(columns='state').dropna()
print(f'Completed trials: {len(df)}, unique: {len(df.cli.unique())}')

if ARGS.mask:
    print(f'Use mask: {ARGS.mask}')
    for m in ARGS.mask.replace('=', ' ').split(':'):
        if m.startswith('^'):
            df = df[df.cli.apply(lambda s: m[1:] not in s)]
        else:
            df = df[df.cli.apply(lambda s: m in s)]

print(f'Selected trials: {len(df)}')
print()
print(df.describe())

for key in ('fps', 'value', 'ssim'):
    print()
    print(f'Top {ARGS.count} best-{key} trials:')
    for r in df.nlargest(ARGS.count, key).sort_values('ssim').itertuples():
        print(f' Score: {r.value:.3f}, SSIM: {r.ssim:.3f}, FPS: {r.fps:.2f}')
        params = ' '.join('='.join(s2.split()) for s in r.cli.split('--')
                          if (s2 := s.strip()))
        print(*(f'  {line}' for line in textwrap.wrap(params, width=140)),
              sep='\n')

df['complexity'] = df.cli.apply(lambda s: s.count('--'))
min_complexity = df['complexity'].min()

# key = 'value'
# for len_, g in df.groupby('complexity'):
    # if len_ > min_complexity + 5:
        # continue    
    # for r in g.nlargest(ARGS.count, key).sort_values('ssim').itertuples():
        # print(f' Score: {r.value:.3f}, SSIM: {r.ssim:.3f}, FPS: {r.fps:.2f}')
        # params = ' '.join('='.join(s2.split()) for s in r.cli.split('--')
                          # if (s2 := s.strip()))
        # print(*(f'  {line}' for line in textwrap.wrap(params, width=140)),
              # sep='\n')
    # print()


fig = plt.figure('SSIM / FPS diagram')
ax = fig.add_subplot(111, xlabel='FPS', ylabel='SSIM (dB)')
ax.set_xscale('log', base=2)
ax.xaxis.set_major_formatter(
    matplotlib.ticker.ScalarFormatter())
ax.xaxis.set_major_locator(
    matplotlib.ticker.LogLocator(base=2, subs=(0.5, 0.707, 1.0)))

# Remove outliers
df = df[(df.ssim >= df.ssim.quantile(0.01)) &
        (df.fps <= df.fps.quantile(0.99))]

fps, ssim, score = df[['fps', 'ssim', 'value']].values.T
df.plot.scatter('fps', 'ssim', c='complexity', ax=ax, cmap='viridis')
plt.clabel(ax.tricontour(fps, ssim, score, 20, colors='gray'),
           inline=1, fontsize=10)
ax.grid()

plt.tight_layout()
plt.show()
