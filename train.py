import csv
import os
import shutil
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import glow  # noqa
import numpy as np
import optuna
from scipy.interpolate import interp1d
from scipy.stats.mstats import hmean
from tqdm import tqdm

from train_utils import DuplicatesPruner, optimize_v2

NUM_CPUS = os.cpu_count() or 1
HERE = Path(__file__).parent / 'x265'
BINARIES = [p.name for p in HERE.glob('*.exe')]

COLUMNS = {
    'bitrate': 'Bitrate',
    'fps': 'FPS',
    'ssim': 'SSIM (dB)',
}
LOG_HEADER = ' | bitrate |  CRF  |  FPS  |  SSIM  |'
LOG_PATTERN = ' | {bitrate:7.1f} | {crf:5.2f} | {fps:5.2f} | {ssim:6.3f} |'

FPS_MIN = 4.
RATE_RELATION = 1.1  # Stop when `bitrate_max / bitrate_min <= RATE_RELATION`

CRF_MIN = 10.0
CRF_MAX = 50.0
CRF_EPS = 0.02  # Stop when `crf_max - crf_min <= CRF_EPS`


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('-r', '--bitrate', default=1500, type=int)
    parser.add_argument('-e', '--evals', default=1000, type=int)

    parser.add_argument('file_or_folder', type=Path)
    return parser.parse_args()


def get_params(trial: optuna.Trial) -> Iterator[str]:

    def get_flag(name: str) -> str:
        return name if trial.suggest_categorical(name, [False, True]) else ''

    def get_int(name: str, low: int, high: int,
                step: int = 1, default: bool = 0) -> str:
        if default != (value :=
                       trial.suggest_int(name, low, high, step)):
            return f'{name} {value}'
        return ''

    # Performance options

    if pmode := get_flag('pmode'):
        yield pmode
    yield get_flag('pme')

    # Mode decision / Analysis

    if 1 != (tu_intra_depth := trial.suggest_int('tu-intra-depth', 1, 4)):
        yield f'tu-intra-depth {tu_intra_depth}'

    yield get_int('tu-inter-depth', 1, 4, default=1)
    yield get_int('limit-tu', 0, 4)

    qg_size: int = trial.suggest_categorical('qg-size',
                                             [8, 16, 32, 64])  # type: ignore

    if ctu := get_flag('ctu 32'):
        yield ctu
        yield get_int('rdpenalty', 0, (2 if tu_intra_depth >= 2 else 1))
        qg_size = min(qg_size, 32)
    else:
        yield get_int('rdpenalty', 0, (2 if tu_intra_depth >= 3 else 1))

    if min_cu_size := get_flag('min-cu-size 16'):
        yield min_cu_size
        qg_size = max(qg_size, 16)

    if qg_size != 32:
        yield f'qg-size {qg_size}'

    if 32 != (max_tu_size :=
              trial.suggest_categorical('max-tu-size', [4, 8, 16, 32])):
        yield f'max-tu-size {max_tu_size}'

    yield get_int('limit-refs', 0, 3, default=3)
    # yield get_int('limit-refs', 0, 1 if pmode else 3, default=3)

    if rect := get_flag('rect'):
        yield rect
        yield get_flag('amp')
        yield get_flag('limit-modes')

    rd = trial.suggest_int('rd', 1, 6)
    if 3 != rd:
        yield f'rd {rd}'

    if rd >= 5:
        yield get_flag('rd-refine')
        yield get_flag('opt-cu-delta-qp')
    else:
        yield get_flag('fast-intra')

    if rd >= 3 and (tskip := get_flag('tskip')):
        yield tskip
        yield get_flag('tskip-fast')

    yield get_int('rskip', 0, 2, default=1)
    yield get_int('rdoq-level', 0, 2, default=0)

    yield get_flag('no-early-skip')
    # ! yield get_flag('no-weightp')
    yield get_flag('weightb')
    yield get_flag('splitrd-skip')

    # Temporal / motion search options

    # {0: dia, 1: hex, 2: umh, 3: star, 4: sea, 5: full}
    yield get_int('me', 0, 5, default=1)
    yield get_int('subme', 0, 7, default=2)
    yield get_int('max-merge', 1, 5, default=3)

    yield get_flag('no-temporal-mvp')
    yield get_flag('hme')

    # Spatial/intra options

    yield get_flag('no-strong-intra-smoothing')
    yield get_flag('constrained-intra')
    yield get_flag('no-b-intra')

    # Slice decision options

    bframes = trial.suggest_int('bframes', 0, 16)
    if bframes != 4:
        yield f'bframes {bframes}'

    yield get_int('b-adapt', 0, 2, default=2)

    max_refs = 8
    if bframes:
        max_refs -= 1
        if no_b_pyramid := get_flag('no-b-pyramid'):
            yield no_b_pyramid
        else:
            max_refs -= 1
    yield get_int('ref', 1, max_refs, default=3)

    if no_open_gop := get_flag('no-open-gop'):
        yield no_open_gop
        yield get_int('radl', 0, bframes)

    rc_lookahead = trial.suggest_int('rc-lookahead', max(bframes + 1, 20), 250,
                                     step=10)
    if rc_lookahead != 20:
        yield f'rc-lookahead {rc_lookahead}'

    yield get_int('gop-lookahead', 0, rc_lookahead, step=5)
    yield get_int('lookahead-slices', 0, 16, step=2, default=8)
    yield get_int('lookahead-threads', 0, NUM_CPUS // 2, step=2)
    # ! yield get_flag('fades')

    # Quality, rate control and rate distortion options

    if hevc_aq := get_flag('hevc-aq'):
        yield hevc_aq
    else:
        aq_mode = trial.suggest_int('aq-mode', 0, 4)
        if aq_mode != 2:
            yield f'aq-mode {aq_mode}'
        if aq_mode:
            aq_strength = trial.suggest_discrete_uniform(
                'aq-strength', 0.6, 2.0, q=0.1)
            aq_strength = np.round(aq_strength, 1)
            if aq_strength != 1.0:
                yield f'aq-strength {aq_strength:.1f}'

        yield get_flag('aq-motion')

    # ! yield get_flag('no-cutree')
    if 0.6 != (qcomp :=
               trial.suggest_discrete_uniform('qcomp', 0.6, 1.0, q=0.1)):
        yield f'qcomp {qcomp:.1f}'

    # Loop filter

    # ! yield get_flag('no-deblock')
    if no_sao := get_flag('no-sao'):
        yield no_sao
    else:
        yield get_flag('sao-non-deblock')
        yield get_flag('limit-sao')
        yield get_int('selective-sao', 0, 4)


def get_score(fps: float, ssim: float,
              ssim_max: float = 30.,
              rate: float = 10.) -> float:
    """
    Combines FPS and SSIM to single value.
    Prioritizes FPS when SSIM > `ssim_max`.
    Allows SSIM to drop 0.1dB if FPS is will increase `rate`x times.
    """
    fps = np.round(fps, 1)
    ssim = np.round(ssim, 3)
    return 0.1 * np.log(fps) / np.log(rate) + min(ssim, ssim_max)


def _transcode_crf_one(binary: Path, params: str, crf: float, path: Path
                       ) -> Dict[str, float]:
    """Transcode one file with specified CRF"""
    root = Path(f'{path}.temp')
    if root.exists():
        shutil.rmtree(root)

    root.mkdir(parents=True)
    out = root / 'out.265'
    csv_log = root / 'log.csv'

    # ? Maybe use '--bitrate {bitrate} --pass {i}' with 'pass' in (1, 2)
    cmd = (
        f'{binary} --input {path} --fps 24000/1001'
        f' --crf {crf}'
        f' --profile main10 {params} --ssim-rd --ssim'
        f' --output {out} --csv {csv_log}'
    )
    res = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        shell=True, text=True)
    if res.returncode:
        for f in (res.stdout, res.stderr):
            if f:
                print(f)
        raise RuntimeError(res.args)

    with csv_log.open() as fp:
        *_, stats = csv.DictReader(fp, skipinitialspace=True)

    return {k: float(stats[k_raw]) for k, k_raw in COLUMNS.items()}


def _transcode_crf(binary: Path, params: str, crf: float) -> Dict[str, float]:
    """Transcode folder with specified CRF"""
    with tqdm(FILES, leave=False, desc=f'Try CRF={crf}') as paths:
        scores = [_transcode_crf_one(binary, params, crf, p) for p in paths]
    return {k: hmean([score[k] for score in scores]) for k in COLUMNS}


def transcode(binary: Path, params: str, target_bitrate: float,
              crf_min: float = CRF_MIN,
              crf_max: float = CRF_MAX) -> Tuple[float, ...]:
    scores: List[List[float]] = []

    bitrate_min = bitrate_max = None
    crf: float
    crf_prev = 0.
    bitrate = 0.
    print(LOG_HEADER)

    while ((bitrate_max or 1e10) / (bitrate_min or 1) > RATE_RELATION) \
            and (crf_max - crf_min > 2 * CRF_EPS) \
            and len(scores) < 10:
        if bitrate_min and bitrate_max:
            crf = interp1d(
                [np.log2(bitrate_min), np.log2(bitrate_max)],
                [crf_max, crf_min])(np.log2(target_bitrate))  # type: ignore
            if abs(crf - crf_prev) < CRF_EPS:
                if bitrate < target_bitrate:
                    crf -= CRF_EPS
                elif target_bitrate < bitrate:
                    crf += CRF_EPS
        elif bitrate_max:
            crf = crf_min + 6 * np.log2(bitrate_max / target_bitrate)
        elif bitrate_min:
            crf = crf_max - 6 * np.log2(target_bitrate / bitrate_min)
        else:
            crf = np.mean([crf_max, crf_min])

        crf = np.clip(crf, crf_min + CRF_EPS, crf_max - CRF_EPS)
        crf = np.round(crf, 2)

        record = _transcode_crf(binary, params, crf)
        crf_prev = crf

        bitrate, fps, ssim = (record[k] for k in ('bitrate', 'fps', 'ssim'))
        if fps < FPS_MIN:
            raise TimeoutError(fps)

        print(LOG_PATTERN.format(bitrate=bitrate, crf=crf, fps=fps, ssim=ssim))
        scores.append([bitrate, crf, fps, ssim])

        if target_bitrate < bitrate:
            bitrate_max = min(bitrate, bitrate_max or 1e10)
            crf_min = max(crf, crf_min)

        elif bitrate < target_bitrate:
            bitrate_min = max(bitrate, bitrate_min or 1)
            crf_max = min(crf, crf_max)

    bitrates, *etc = zip(*scores)
    crf, fps, ssim = interp1d(bitrates, etc)(target_bitrate)  # type: ignore

    print(
        LOG_PATTERN.format(
            bitrate=target_bitrate, crf=crf, fps=fps, ssim=ssim),
        'final'
    )
    return crf, fps, ssim


def objective(trial: optuna.Trial) -> float:
    binary = HERE / str(trial.suggest_categorical('binary', BINARIES))

    params_raw = [f'--{p}' for p in get_params(trial) if p]
    params = ' '.join(params_raw)
    trial.set_user_attr('cli', ' '.join(sorted(params_raw)))

    if trial.should_prune():
        raise optuna.TrialPruned('Duplicate parameters')

    print(f'Test: {params}')
    try:
        crf, fps, ssim = transcode(binary, params,
                                   target_bitrate=ARGS.bitrate)
    except TimeoutError as exc:
        fps, = exc.args
        trial.set_user_attr('fps', fps)
        return get_score(fps, 0)

    trial.set_user_attr('crf', np.round(crf, 3))
    trial.set_user_attr('fps', fps)
    trial.set_user_attr('ssim', ssim)

    return get_score(fps, ssim)


ARGS = parse_args()
if ARGS.file_or_folder.is_file():
    assert ARGS.file_or_folder.suffix == '.y4m'
    FILES = [ARGS.file_or_folder]
else:
    FILES = [*ARGS.file_or_folder.glob('*.y4m')]

study = optuna.create_study(
    'sqlite:///optuna.db',
    pruner=DuplicatesPruner(lambda t: t.user_attrs['cli']),
    study_name=ARGS.file_or_folder.as_posix(),
    direction='maximize',
    load_if_exists=True)

optimize_v2(study, objective, n_trials=ARGS.evals,
            catch=(RuntimeError,))
