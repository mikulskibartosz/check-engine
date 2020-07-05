from typing import Optional, List, Union

from pandas import DataFrame as Pandas_df
from pyspark.sql import DataFrame as Spark_df
from numpy.testing import assert_array_equal


class AssertDf:
    def __init__(self, df: Spark_df, order_by_column: Optional[Union[str, List[str]]] = None):
        self.df: Pandas_df = df.toPandas()
        self.order_by_column = order_by_column

    def is_empty(self):
        assert self.df.empty
        return self

    def contains_exactly(self, other: Pandas_df):
        if self.order_by_column:
            sorted_df = self.df.sort_values(self.order_by_column)
            other_sorted = other.sort_values(self.order_by_column)
            assert_array_equal(sorted_df.values, other_sorted.values, verbose=True)
        else:
            assert self.df.equals(other)
        return self

    def has_columns(self, columns: list):
        existing_columns = sorted(list(self.df.columns))
        expected_columns = sorted(columns)
        assert existing_columns == expected_columns, f"{existing_columns} != {expected_columns}"
        return self

    def has_n_rows(self, n):
        assert self.df.shape[0] == n
        return self
