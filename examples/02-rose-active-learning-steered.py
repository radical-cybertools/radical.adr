"""02-rose-ensemble-steered.py

Two ROSE SequentialActiveLearner instances steered by one ADR operator.

Each ADR cycle spawns both learners as @act tasks.  The operator observes
their MSE, then steers: the better-performing learner gets more active
learning budget (n_select) next cycle.  Both learners accumulate labeled
data across cycles via their own pickle files.

ADR concepts demonstrated
  - @act wrapping a multi-step ROSE run (simulation → train → active-learn)
  - @observe extracting per-learner metrics
  - EnsemblePolicy(RulePolicy) allocating n_select budget based on comparative MSE
  - ScientificGoal stopping when either learner crosses the MSE threshold

Dependencies:
    pip install numpy scikit-learn
    pip install -e /path/to/ROSE
    pip install -e /path/to/radical.asyncflow

Usage:
    python 02-rose-ensemble-steered.py
"""

from __future__ import annotations

import asyncio
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from radical.asyncflow import LocalExecutionBackend, WorkflowEngine
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error

from rose.al import SequentialActiveLearner
from rose.metrics import MEAN_SQUARED_ERROR_MSE

from radical.adr import Decision, Operator, Policy, decide, goals, observe, act
from radical.adr.goals import Goal

# ── Shared state (module-level because @act impls take no self) ──────────────

_engine: WorkflowEngine | None = None

DATA = {
    "A": Path("/tmp/rose_ensemble_A.pkl"),
    "B": Path("/tmp/rose_ensemble_B.pkl"),
}


def _target(X: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * X) + 0.1 * np.random.randn(*X.shape)


def _save(path: Path, X_lab, y_lab, X_pool, y_pool, model=None) -> None:
    with open(path, "wb") as f:
        pickle.dump({"X_labeled": X_lab, "y_labeled": y_lab,
                     "X_pool": X_pool, "y_pool": y_pool, "model": model}, f)


def _load(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _make_learner(learner_id: str, n_select: int) -> SequentialActiveLearner:
    """Create a ROSE learner whose tasks read/write its own data file."""
    path    = DATA[learner_id]
    learner = SequentialActiveLearner(_engine)

    async def simulation(*_, n_initial: int = 10, n_pool: int = 100) -> dict:
        np.random.seed({"A": 0, "B": 1}[learner_id])
        X_lab  = np.random.uniform(0, 1, (n_initial, 1))
        y_lab  = _target(X_lab)
        X_pool = np.random.uniform(0, 1, (n_pool, 1))
        y_pool = _target(X_pool)
        _save(path, X_lab, y_lab, X_pool, y_pool)
        return {}

    async def training(*_) -> dict:
        d      = _load(path)
        model  = GaussianProcessRegressor(
            kernel=RBF() + WhiteKernel(noise_level=0.1), n_restarts_optimizer=2)
        model.fit(d["X_labeled"], d["y_labeled"].ravel())
        _save(path, d["X_labeled"], d["y_labeled"], d["X_pool"], d["y_pool"], model)
        return {}

    async def active_learn(*_) -> dict:
        d      = _load(path)
        _, std = d["model"].predict(d["X_pool"], return_std=True)
        idx    = np.argsort(std)[-n_select:]
        _save(path,
              np.vstack([d["X_labeled"], d["X_pool"][idx]]),
              np.vstack([d["y_labeled"], d["y_pool"][idx].reshape(-1, 1)]),
              np.delete(d["X_pool"], idx, axis=0),
              np.delete(d["y_pool"], idx, axis=0),
              d["model"])
        return {}

    async def check_mse(*_) -> float:
        d      = _load(path)
        X_val  = np.linspace(0, 1, 50).reshape(-1, 1)
        return mean_squared_error(_target(X_val), d["model"].predict(X_val))

    learner.simulation_task(as_executable=False)(simulation)
    learner.training_task(as_executable=False)(training)
    learner.active_learn_task(as_executable=False)(active_learn)
    learner.as_stop_criterion(metric_name=MEAN_SQUARED_ERROR_MSE, threshold=0.01,
                              as_executable=False)(check_mse)
    return learner


# ── Operator ──────────────────────────────────────────────────────────────────

class EnsembleOperator(Operator):

    best_mse: float = float("inf")

    @goals
    def convergence_criteria(self):
        return Goal(name="converged", metric="best_mse",
                              threshold=0.005, direction="minimize")

    @act
    async def run_learner(self, learner_id: str, n_select: int = 5) -> dict:
        """Run one ROSE learner for 5 iterations and return its final MSE."""
        learner = _make_learner(learner_id, n_select)
        state   = None
        async for state in learner.start(max_iter=5):
            pass
        return {"learner_id": learner_id,
                "mse":         state.metric_value,
                "n_select":    n_select}

    @observe
    def extract(self, snapshot) -> dict:
        results = [v for k, v in snapshot.observations.items()
                   if k.startswith("result.") and isinstance(v, dict) and "mse" in v]
        by_id   = {r["learner_id"]: r for r in results}
        mselist = [r["mse"] for r in results if r.get("mse") is not None]
        if mselist:
            self.best_mse = min(mselist)
        return {"by_id": by_id, "best_mse": self.best_mse, "cycle": snapshot.cycle}



class EnsemblePolicy(Policy):

    def __init__(self, op: EnsembleOperator) -> None:
        super().__init__()
        self._act = op.get_actions()

    @decide
    async def run(self, obs: dict) -> Decision:
        if obs["cycle"] >= 6:
            return Decision(stop=True)

        if not obs["by_id"]:
            return Decision(actions=[
                self._act.run_learner(learner_id="A", n_select=5),
                self._act.run_learner(learner_id="B", n_select=5),
            ])

        a_mse  = obs["by_id"].get("A", {}).get("mse", float("inf"))
        b_mse  = obs["by_id"].get("B", {}).get("mse", float("inf"))
        a_sel, b_sel = (8, 3) if a_mse <= b_mse else (3, 8)
        return Decision(actions=[
            self._act.run_learner(learner_id="A", n_select=a_sel),
            self._act.run_learner(learner_id="B", n_select=b_sel),
        ])


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    global _engine
    backend = await LocalExecutionBackend(ProcessPoolExecutor(max_workers=4))
    _engine = await WorkflowEngine.create(backend)

    op        = EnsembleOperator(_engine)
    op.policy = EnsemblePolicy(op)

    print("Ensemble ROSE  |  goal: best_mse < 0.005\n")
    print(f"  {'cycle':>5}  {'A mse':>8}  {'B mse':>8}  {'A sel':>6}  {'B sel':>6}")
    print("  " + "-" * 40)

    async for snapshot in op.run():
        by_id = {r["learner_id"]: r
                 for k, r in snapshot.observations.items()
                 if k.startswith("result.") and isinstance(r, dict) and "mse" in r}
        a = by_id.get("A", {})
        b = by_id.get("B", {})
        print(f"  {snapshot.cycle:>5}  "
              f"{a.get('mse', float('nan')):>8.5f}  "
              f"{b.get('mse', float('nan')):>8.5f}  "
              f"{a.get('n_select', '-'):>6}  "
              f"{b.get('n_select', '-'):>6}")

    print(f"\nBest MSE: {op.best_mse:.5f}  cycles: {op.state.cycle}")

    await _engine.shutdown()
    for p in DATA.values():
        p.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
