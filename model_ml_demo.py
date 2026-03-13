import logging
import os
import sys
from datetime import datetime
from multiprocessing import get_context
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from jade_ml.subrun import subrun

try:
    from .abstract.factor import Factor
except ImportError:
    from abstract.factor import Factor

logger = logging.getLogger(__name__)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TinyBinaryNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def _train_submodel(task: Dict[str, object]) -> Dict[str, object]:
    model_name = str(task["model_name"])
    seed = int(task["seed"])
    feature_dim = int(task["feature_dim"])
    lookback = int(task["lookback"])
    train_steps = int(task["train_steps"])
    symbols = list(task["symbols"])
    train_time = str(task["train_time"])

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    sample_count = len(symbols) * lookback
    features = rng.normal(0, 1, size=(sample_count, feature_dim)).astype(np.float32)
    logits = features[:, 0] * 0.8 - features[:, 1] * 0.3 + features[:, 2] * 0.2
    labels = (logits > 0).astype(np.float32).reshape(-1, 1)

    x_train = torch.tensor(features)
    y_train = torch.tensor(labels)
    model = TinyBinaryNet(input_dim=feature_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    with subrun(
        run_name=f"submodel:{model_name}",
        extra_tags={"jade.submodel": model_name},
        extra_params={"model": model_name, "seed": seed},
    ) as ctx:
        ctx.log_params(
            {
                "model": model_name,
                "seed": seed,
                "symbols": len(symbols),
                "lookback": lookback,
                "feature_dim": feature_dim,
                "train_steps": train_steps,
                "model_type": "torch",
            }
        )

        model.train()
        for step in range(train_steps):
            optimizer.zero_grad()
            sub_logits = model(x_train)
            loss = criterion(sub_logits, y_train)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = (torch.sigmoid(sub_logits) > 0.5).float()
                acc = (pred.eq(y_train)).float().mean().item()

            ctx.log_metrics(
                {
                    "train_loss": float(loss.item()),
                    "train_acc": float(acc),
                },
                step=step + 1,
            )

        model.eval()
        infer_features = rng.normal(0, 1, size=(len(symbols), feature_dim)).astype(np.float32)
        with torch.no_grad():
            scores = torch.sigmoid(model(torch.tensor(infer_features))).cpu().numpy().reshape(-1)

        ctx.log_model(model.cpu(), model_name, model_type="torch")

    return {
        "model_name": model_name,
        "train_time": train_time,
        "seed": seed,
        "symbols": symbols,
        "scores": scores.tolist(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }


class model_ml_demo(Factor):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.symbols = [f"DEMO{i:03d}" for i in range(32)]
        self.lookback = 12
        self.feature_dim = 6
        self.train_steps = 5
        self.submodel_names = ["model_a", "model_b", "model_c"]

    def compute(self, input: Dict[str, pd.DataFrame], time: datetime, tracker=None) -> Dict[str, pd.DataFrame]:
        _ = input
        ts = pd.Timestamp(time)
        base_seed = int(ts.timestamp()) % (86400 * 365)
        train_time = ts.strftime("%Y-%m-%d %H:%M:%S")

        if tracker:
            tracker.log_params(
                {
                    "seed": base_seed,
                    "symbols": len(self.symbols),
                    "lookback": self.lookback,
                    "feature_dim": self.feature_dim,
                    "train_steps": self.train_steps,
                    "submodels": len(self.submodel_names),
                    "model_type": "torch",
                }
            )

        sub_tasks = self._build_sub_tasks(base_seed=base_seed, train_time=train_time)
        sub_results = self._run_submodels(sub_tasks)
        score_df, final_model = self._merge_results(sub_results=sub_results, time=time)

        if tracker:
            tracker.log_metrics(
                {
                    "ensemble_mean_score": float(score_df["value"].mean()),
                    "ensemble_score_std": float(score_df["value"].std(ddof=0)),
                },
                step=self.train_steps + 1,
            )
            tracker.log_model(final_model.cpu(), "model_torch_model", model_type="torch")

        logger.info(
            "model ml multiprocess demo compute complete: time=%s base_seed=%s submodels=%s",
            train_time,
            base_seed,
            len(sub_results),
        )
        return {
            "demo__model_torch_score__1d": score_df,
            "model_torch_model": {
                train_time: final_model.cpu(),
            },
        }

    def compute_history(
        self,
        input: Dict[str, pd.DataFrame],
        start_time: datetime,
        end_time: datetime,
        run_times: list,
    ) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError("model_ml_demo is schedule-only; compute_history is intentionally omitted")

    def _build_sub_tasks(self, base_seed: int, train_time: str) -> List[Dict[str, object]]:
        tasks: List[Dict[str, object]] = []
        for idx, model_name in enumerate(self.submodel_names):
            tasks.append(
                {
                    "model_name": model_name,
                    "seed": base_seed + idx * 97,
                    "feature_dim": self.feature_dim,
                    "lookback": self.lookback,
                    "train_steps": self.train_steps,
                    "symbols": self.symbols,
                    "train_time": train_time,
                }
            )
        return tasks

    def _run_submodels(self, sub_tasks: List[Dict[str, object]]) -> List[Dict[str, object]]:
        # 显式将项目目录加入 sys.path，确保 spawn 子进程可重新导入当前模块。
        ctx = get_context("spawn")
        with ctx.Pool(processes=len(sub_tasks)) as pool:
            return pool.map(_train_submodel, sub_tasks)

    def _merge_results(self, sub_results: List[Dict[str, object]], time: datetime) -> Tuple[pd.DataFrame, nn.Module]:
        score_matrix = np.array([result["scores"] for result in sub_results], dtype=np.float32)
        mean_scores = score_matrix.mean(axis=0)

        model = TinyBinaryNet(input_dim=self.feature_dim)
        avg_state = {}
        for key in sub_results[0]["state_dict"]:
            stacked = torch.stack([result["state_dict"][key] for result in sub_results], dim=0)
            avg_state[key] = stacked.mean(dim=0)
        model.load_state_dict(avg_state)
        model.eval()

        ts = pd.Timestamp(time).strftime("%Y-%m-%d %H:%M:%S")
        score_df = pd.DataFrame(
            {
                "time": [ts] * len(self.symbols),
                "symbol": self.symbols,
                "value": mean_scores,
            }
        )
        return score_df, model
