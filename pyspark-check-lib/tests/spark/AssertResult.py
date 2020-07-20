from pyspark.sql import DataFrame

from pyspark_check.validate_df import ValidationResult, ValidationError
from tests.spark.assert_df import AssertDf


class AssertValidationResult:
    def __init__(self, *, column_name: str, constraint_name: str):
        self.column_name = column_name
        self.constraint_name = constraint_name

    def check(self, *, actual: ValidationResult, expected_correct: DataFrame, expected_erroneous: DataFrame):
        if expected_correct.count() == 0:
            AssertDf(actual.correct_data) \
                .is_empty() \
                .has_columns(expected_correct.columns)
        else:
            AssertDf(actual.correct_data, order_by_column=self.column_name) \
                .contains_exactly(expected_correct.toPandas()) \
                .has_columns(expected_correct.columns)

        if expected_erroneous.count() == 0:
            AssertDf(actual.erroneous_data) \
                .is_empty() \
                .has_columns(expected_erroneous.columns)
        else:
            AssertDf(actual.erroneous_data, order_by_column=self.column_name) \
                .contains_exactly(expected_erroneous.toPandas()) \
                .has_columns(expected_erroneous.columns)

        if expected_erroneous.count() == 0:
            assert actual.errors == []
        else:
            assert actual.errors == [ValidationError(self.column_name, self.constraint_name, expected_erroneous.count())]
