from pyspark.sql import DataFrame

from checkengine._constraints._Constraint import _Constraint


class _OneOf(_Constraint):
    def __init__(self, column_name: str, allowed_values: list):
        super().__init__(column_name)
        self.allowed_values = allowed_values

    def prepare_df_for_check(self, data_frame: DataFrame) -> DataFrame:
        return data_frame

    def filter_success(self, data_frame: DataFrame) -> DataFrame:
        return data_frame.filter(data_frame[self.column_name].isin(*self.allowed_values))

    def filter_failure(self, data_frame: DataFrame) -> DataFrame:
        return data_frame.filter(~data_frame[self.column_name].isin(*self.allowed_values))

    def constraint_name(self):
        return "one_of"
