__all__ = ['DuplicatesPruner', 'optimize_v2']

import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Type

import optuna
from optuna import Study
from optuna.pruners import BasePruner
from optuna.trial import FrozenTrial, Trial, TrialState

_Objective = Callable[[Trial], float]
_Callback = Callable[[Study, FrozenTrial], None]


class DuplicatesPruner(BasePruner):
    def __init__(self, key_fn=lambda t: t.params):
        super().__init__()
        self.__key_fn = key_fn

    def prune(self, study: Study, trial: FrozenTrial) -> bool:
        key = self.__key_fn(trial)
        return any(key == self.__key_fn(t)
                   for t in study.get_trials(deepcopy=False)
                   if t.state == TrialState.COMPLETE)


def _dump_logs(study: Study):
    print('Update dashboard...')
    optuna.dashboard._write(study, 'optuna_dashboard.html')

    for name, fn in {
        'history': optuna.visualization.plot_optimization_history,
        'importance': optuna.visualization.plot_param_importances,
        'slice': optuna.visualization.plot_slice,
    }.items():
        print(f'Call {fn.__name__}...')
        html = fn(study).to_html()  # type: ignore
        Path(f'optuna_{name}.html').write_text(html)

    print('Done update logs')


class _ReportSota:
    completed = 0

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        completed = sum(t.state == TrialState.COMPLETE
                        for t in study.get_trials(deepcopy=False))
        if not completed:
            return

        title = (f'{Path(sys.argv[0]).name}: '
                 f'{study.best_value:.3f} SoTA '
                 f'(run {study.best_trial.number}/{trial.number})')
        print(f'\33]0;{title}\a', end='', flush=True)

        if self.completed != completed and (completed + 1) % 10 == 0:
            self.completed = completed
            _dump_logs(study)


def optimize_v2(study: Study,
                objective: _Objective,
                n_trials: Optional[int] = None,
                catch: Tuple[Type[Exception], ...] = (),
                callbacks: Optional[List[_Callback]] = None) -> None:
    try:
        study.optimize(
            objective,
            n_trials,
            catch=(*catch, RuntimeError),
            callbacks=[*(callbacks or []), _ReportSota()])
    finally:
        if any(t.state == TrialState.COMPLETE
               for t in study.get_trials(deepcopy=False)):
            _dump_logs(study)
            print(study.best_params)
            print(study.best_value)
