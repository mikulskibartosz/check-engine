from pandas import DataFrame


class AssertDf:
    def __init__(self, df):
        self.df: DataFrame = df.toPandas()

    def is_empty(self):
        assert self.df.empty
        return self

    def contains_exactly(self, other: DataFrame):
        assert self.df.equals(other)
        return self

    def has_columns(self, columns: list):
        assert sorted(list(self.df.columns)) == sorted(columns)
        return self

    def has_n_rows(self, n):
        assert self.df.shape[0] == n
        return self
