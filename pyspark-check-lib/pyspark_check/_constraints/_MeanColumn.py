from pyspark.sql import DataFrame

from pyspark_check._constraints._Numbers import _Number


class _MeanColumn(_Number):
    def __init__(self, column_name: str, min_mean: float, max_mean: float):
        super().__init__(column_name)
        self.lower_bound = min_mean
        self.upper_bound = max_mean

    def prepare_df_for_check(self, data_frame: DataFrame) -> DataFrame:
        average: DataFrame = data_frame \
            .groupby() \
            .avg(self.column_name) \
            .withColumnRenamed(f"avg({self.column_name})", self.constraint_column_name)

        return data_frame.crossJoin(average)

    def filter_success(self, data_frame: DataFrame) -> DataFrame:
        return data_frame.filter(f"{self.constraint_column_name} >= {self.lower_bound} AND {self.constraint_column_name} <= {self.upper_bound}")

    def filter_failure(self, data_frame: DataFrame) -> DataFrame:
        return data_frame.filter(f"{self.constraint_column_name} < {self.lower_bound} OR {self.constraint_column_name} > {self.upper_bound}")

    def constraint_name(self):
        return "mean_between"
