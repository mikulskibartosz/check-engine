from abc import ABC
from typing import List, Tuple

from pandas import DataFrame

from pyspark_check._constraints._Constraint import _Constraint


class _Number(_Constraint, ABC):
    def __init__(self, column_name: str, value: int):
        self.value = value
        super().__init__(column_name)

    def prepare_df_for_check(self, data_frame: DataFrame) -> DataFrame:
        return data_frame

    def validate_self(self, data_frame: DataFrame, df_columns: List[str]) -> Tuple[bool, str]:
        parent_validation_result = super().validate_self(data_frame, df_columns)
        if not parent_validation_result[0]:
            return parent_validation_result
        else:
            column_type = [dtype for name, dtype in data_frame.dtypes if name == self.column_name][0]
            return column_type in ["tinyint", "smallint", "int", "bigint"], f"Column {self.column_name} is not a number"


class _Min(_Number):
    def __init__(self, column_name: str, value: int):
        super().__init__(column_name, value)

    def filter_success(self, data_frame: DataFrame) -> DataFrame:
        return data_frame.filter(f"{self.column_name} >= {self.value}")

    def filter_failure(self, data_frame: DataFrame) -> DataFrame:
        return data_frame.filter(f"{self.column_name} < {self.value}")

    def constraint_name(self):
        return "min"


class _Max(_Number):
    def __init__(self, column_name: str, value: int):
        super().__init__(column_name, value)

    def filter_success(self, data_frame: DataFrame) -> DataFrame:
        return data_frame.filter(f"{self.column_name} <= {self.value}")

    def filter_failure(self, data_frame: DataFrame) -> DataFrame:
        return data_frame.filter(f"{self.column_name} > {self.value}")

    def constraint_name(self):
        return "max"
