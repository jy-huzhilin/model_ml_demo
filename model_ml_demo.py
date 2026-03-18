import logging
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from jade_ml import JadeTracker

from .abstract.factor import Factor

logger = logging.getLogger(__name__)


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


class model_ml_demo(Factor):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.symbols = [f"DEMO{i:03d}" for i in range(32)]
        self.feature_dim = 6

    def compute(self, input: Dict[str, object], time: datetime) -> Dict[str, pd.DataFrame]:
        model_dict = input.get("model_torch_model")
        if not isinstance(model_dict, dict) or not model_dict:
            raise RuntimeError("model_torch_model input is empty; cannot run inference")

        latest_train_time = max(model_dict.keys())
        model = model_dict[latest_train_time]
        if not hasattr(model, "eval"):
            raise TypeError("loaded model_torch_model is not a valid torch model")

        tracker = JadeTracker(tags={"jade.task_kind": "model_ml_demo_infer"})
        with tracker.start_run(run_name=f"model_ml_demo_infer:{pd.Timestamp(time)}") as ctx:
            ctx.log_params(
                {
                    "symbols": len(self.symbols),
                    "feature_dim": self.feature_dim,
                    "loaded_model_name": "model_torch_model",
                    "loaded_train_time": str(latest_train_time),
                    "mode": "inference_only",
                }
            )

            rng = np.random.default_rng(int(pd.Timestamp(time).timestamp()) % (86400 * 365))
            infer_features = rng.normal(0, 1, size=(len(self.symbols), self.feature_dim)).astype(np.float32)

            model.eval()
            with torch.no_grad():
                logits = model(torch.tensor(infer_features))
                scores = torch.sigmoid(logits).cpu().numpy().reshape(-1)

            score_df = pd.DataFrame(
                {
                    "time": [pd.Timestamp(time).strftime("%Y-%m-%d %H:%M:%S")] * len(self.symbols),
                    "symbol": self.symbols,
                    "value": scores,
                    "loaded_train_time": [str(latest_train_time)] * len(self.symbols),
                }
            )

            ctx.log_metrics(
                {
                    "infer_mean_score": float(score_df["value"].mean()),
                    "infer_score_std": float(score_df["value"].std(ddof=0)),
                },
                step=1,
            )

        logger.info(
            "model ml demo inference complete: time=%s loaded_train_time=%s symbols=%s",
            pd.Timestamp(time).strftime("%Y-%m-%d %H:%M:%S"),
            latest_train_time,
            len(self.symbols),
        )
        return {
            "demo__model_torch_infer__1d": score_df,
        }

    def compute_history(
        self,
        input: Dict[str, pd.DataFrame],
        start_time: datetime,
        end_time: datetime,
        run_times: list,
    ) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError("model_ml_demo is schedule-only; compute_history is intentionally omitted")
