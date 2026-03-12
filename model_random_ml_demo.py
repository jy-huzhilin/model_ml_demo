import logging
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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


class model_random_ml_demo(Factor):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.symbols = [f"DEMO{i:03d}" for i in range(32)]
        self.lookback = 12
        self.feature_dim = 6
        self.train_steps = 5

    def compute(self, input: Dict[str, pd.DataFrame], time: datetime, tracker=None) -> Dict[str, pd.DataFrame]:
        _ = input
        seed = int(pd.Timestamp(time).timestamp()) % (86400 * 365)
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        X_train, y_train = self._build_train_dataset(rng)
        model = TinyBinaryNet(input_dim=self.feature_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()

        if tracker:
            tracker.log_params(
                {
                    "seed": seed,
                    "symbols": len(self.symbols),
                    "lookback": self.lookback,
                    "feature_dim": self.feature_dim,
                    "train_steps": self.train_steps,
                    "model_type": "torch",
                }
            )

        model.train()
        for step in range(self.train_steps):
            optimizer.zero_grad()
            logits = model(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = (torch.sigmoid(logits) > 0.5).float()
                acc = (pred.eq(y_train)).float().mean().item()

            if tracker:
                tracker.log_metrics(
                    {
                        "train_loss": float(loss.item()),
                        "train_acc": float(acc),
                    },
                    step=step + 1,
                )

        score_df = self._build_output_scores(model=model, time=time, rng=rng)

        if tracker:
            tracker.log_model(model.cpu(), "random_torch_model", model_type="torch")

        train_time = pd.Timestamp(time).strftime("%Y-%m-%d %H:%M:%S")
        logger.info("random ml demo compute complete: time=%s seed=%s", train_time, seed)
        return {
            "demo__random_torch_score__1d": score_df,
            "random_torch_model": {
                train_time: model.cpu(),
            },
        }

    def compute_history(
        self,
        input: Dict[str, pd.DataFrame],
        start_time: datetime,
        end_time: datetime,
        run_times: list,
    ) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError("model_random_ml_demo is schedule-only; compute_history is intentionally omitted")

    def _build_train_dataset(self, rng: np.random.Generator):
        sample_count = len(self.symbols) * self.lookback
        features = rng.normal(0, 1, size=(sample_count, self.feature_dim)).astype(np.float32)
        logits = features[:, 0] * 0.8 - features[:, 1] * 0.3 + features[:, 2] * 0.2
        labels = (logits > 0).astype(np.float32).reshape(-1, 1)
        return torch.tensor(features), torch.tensor(labels)

    def _build_output_scores(self, model: nn.Module, time: datetime, rng: np.random.Generator) -> pd.DataFrame:
        model.eval()
        features = rng.normal(0, 1, size=(len(self.symbols), self.feature_dim)).astype(np.float32)
        with torch.no_grad():
            scores = torch.sigmoid(model(torch.tensor(features))).cpu().numpy().reshape(-1)

        ts = pd.Timestamp(time).strftime("%Y-%m-%d %H:%M:%S")
        return pd.DataFrame(
            {
                "time": [ts] * len(self.symbols),
                "symbol": self.symbols,
                "value": scores,
            }
        )
