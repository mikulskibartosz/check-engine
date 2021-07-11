from pyspark.sql import DataFrame

from checkengine._constraints._Constraint import _Constraint


class _Unique(_Constraint):
    def __init__(self, column_name: str):
        super().__init__(column_name)

    def prepare_df_for_check(self, data_frame: DataFrame) -> DataFrame:
        count_repetitions: DataFrame = data_frame \
            .groupby(self.column_name) \
            .count() \
            .withColumnRenamed("count", self.constraint_column_name)

        return data_frame.join(count_repetitions, self.column_name, "left")

    def filter_success(self, data_frame: DataFrame) -> DataFrame:
        return data_frame.filter(f"{self.constraint_column_name} == 1")

    def filter_failure(self, data_frame: DataFrame) -> DataFrame:
        return data_frame.filter(f"{self.constraint_column_name} > 1")

    def constraint_name(self):
        return "unique"
