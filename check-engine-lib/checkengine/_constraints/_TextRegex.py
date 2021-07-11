from typing import List, Tuple

from pyspark.sql import DataFrame

from checkengine._constraints._Constraint import _Constraint


class _TextRegex(_Constraint):
    def __init__(self, column_name: str, regex: str):
        super().__init__(column_name)
        self.regex = regex

    def prepare_df_for_check(self, data_frame: DataFrame) -> DataFrame:
        return data_frame.withColumn(self.constraint_column_name, data_frame[self.column_name].rlike(self.regex))

    def validate_self(self, data_frame: DataFrame, df_columns: List[str]) -> Tuple[bool, str]:
        parent_validation_result = super().validate_self(data_frame, df_columns)
        if not parent_validation_result[0]:
            return parent_validation_result
        else:
            column_type = [dtype for name, dtype in data_frame.dtypes if name == self.column_name][0]
            return column_type == 'string', f"Column {self.column_name} is not a string."

    def filter_success(self, data_frame: DataFrame) -> DataFrame:
        return data_frame.filter(f"{self.constraint_column_name} = TRUE")

    def filter_failure(self, data_frame: DataFrame) -> DataFrame:
        return data_frame.filter(f"{self.constraint_column_name} = FALSE")

    def constraint_name(self):
        return "regex_match"
