from pandas import DataFrame

from pyspark_check._constraints._Constraint import _Constraint


class _NotNull(_Constraint):
    def __init__(self, column_name: str):
        super().__init__(column_name)

    def prepare_df_for_check(self, data_frame: DataFrame) -> DataFrame:
        return data_frame

    def filter_success(self, data_frame: DataFrame) -> DataFrame:
        return data_frame.filter(f"{self.column_name} IS NOT NULL")

    def filter_failure(self, data_frame: DataFrame) -> DataFrame:
        return data_frame.filter(f"{self.column_name} IS NULL")

    def constraint_name(self):
        return "not_null"
