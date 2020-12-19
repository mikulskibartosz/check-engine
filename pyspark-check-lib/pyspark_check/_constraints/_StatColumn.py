"""
Checks that require applying a statistical function to all values in a single column.
"""
from abc import ABC

from pyspark.sql import DataFrame
import pyspark.sql.functions as F

from pyspark_check._constraints._Numbers import _Number


class _StatColumn(_Number, ABC):
    def __init__(self, column_name: str, lower_bound: float, upper_bound: float):
        super().__init__(column_name)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def filter_success(self, data_frame: DataFrame) -> DataFrame:
        return data_frame.filter(f"{self.constraint_column_name} >= {self.lower_bound} AND {self.constraint_column_name} <= {self.upper_bound}")

    def filter_failure(self, data_frame: DataFrame) -> DataFrame:
        return data_frame.filter(f"{self.constraint_column_name} < {self.lower_bound} OR {self.constraint_column_name} > {self.upper_bound}")


class _MeanColumn(_StatColumn):
    def prepare_df_for_check(self, data_frame: DataFrame) -> DataFrame:
        average: DataFrame = data_frame \
            .groupby() \
            .avg(self.column_name) \
            .withColumnRenamed(f"avg({self.column_name})", self.constraint_column_name)

        return data_frame.crossJoin(average)

    def constraint_name(self):
        return "mean_between"


class _MedianColumn(_StatColumn):
    def prepare_df_for_check(self, data_frame: DataFrame) -> DataFrame:
        median = F.expr(f"percentile_approx({self.column_name}, 0.5)")

        average: DataFrame = data_frame \
            .groupby() \
            .agg(median.alias(self.constraint_column_name))

        return data_frame.crossJoin(average)

    def constraint_name(self):
        return "median_between"
