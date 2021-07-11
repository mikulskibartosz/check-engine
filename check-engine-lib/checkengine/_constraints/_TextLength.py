from typing import List, Tuple

from pyspark.sql import DataFrame

from checkengine._constraints._Constraint import _Constraint


class _TextLength(_Constraint):
    def __init__(self, column_name: str, lower_bound: int, upper_bound: int):
        super().__init__(column_name)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def prepare_df_for_check(self, data_frame: DataFrame) -> DataFrame:
        return data_frame

    def validate_self(self, data_frame: DataFrame, df_columns: List[str]) -> Tuple[bool, str]:
        parent_validation_result = super().validate_self(data_frame, df_columns)
        if not parent_validation_result[0]:
            return parent_validation_result
        else:
            column_type = [dtype for name, dtype in data_frame.dtypes if name == self.column_name][0]
            if column_type != 'string':
                return False, f"Column {self.column_name} is not a string."
            else:
                return self.lower_bound <= self.upper_bound, f"Upper bound ({self.upper_bound}) cannot be lower than lower bound ({self.lower_bound})."

    def filter_success(self, data_frame: DataFrame) -> DataFrame:
        return data_frame.filter(f"LENGTH({self.column_name}) >= {self.lower_bound} AND LENGTH({self.column_name}) <= {self.upper_bound}")

    def filter_failure(self, data_frame: DataFrame) -> DataFrame:
        return data_frame.filter(f"LENGTH({self.column_name}) < {self.lower_bound} OR LENGTH({self.column_name}) > {self.upper_bound}")

    def constraint_name(self):
        return "text_length"
