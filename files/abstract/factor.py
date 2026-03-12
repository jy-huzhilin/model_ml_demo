from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict

import pandas as pd


class Factor(ABC):
    @abstractmethod
    def compute(self, input: Dict[str, pd.DataFrame], time: datetime) -> Dict[str, pd.DataFrame]:
        pass

    @abstractmethod
    def compute_history(
        self,
        input: Dict[str, pd.DataFrame],
        start_time: datetime,
        end_time: datetime,
        run_times: list,
    ) -> Dict[str, pd.DataFrame]:
        pass
