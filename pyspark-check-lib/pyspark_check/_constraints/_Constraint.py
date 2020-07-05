from typing import List, Tuple
from abc import ABC, abstractmethod
import random
import string

from pyspark.sql import DataFrame


def _generate_constraint_column_name(constraint_type, column_name):
    random_suffix = ''.join(random.choice(string.ascii_lowercase) for i in range(12))
    return f"__pyspark_check__{column_name}_{constraint_type}_{random_suffix}"


class _Constraint(ABC):
    def __init__(self, column_name: str):
        self.column_name = column_name
        self.constraint_column_name = _generate_constraint_column_name(self.constraint_name(), column_name)

    @abstractmethod
    def constraint_name(self):
        pass

    @abstractmethod
    def prepare_df_for_check(self, data_frame: DataFrame) -> DataFrame:
        return data_frame

    @abstractmethod
    def filter_success(self, data_frame: DataFrame) -> DataFrame:
        return data_frame

    @abstractmethod
    def filter_failure(self, data_frame: DataFrame) -> DataFrame:
        return data_frame

    def validate_self(self, data_frame: DataFrame, df_columns: List[str]) -> Tuple[bool, str]:
        return self.column_name in df_columns, f"There is no '{self.column_name}' column"
