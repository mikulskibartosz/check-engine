"""
This file contains the tests that verify whether the library works correctly when multiple constraints are defined at the same time.
"""

import pytest

from checkengine.validate_df import ValidateSparkDataFrame, ValidationError
from tests.spark import two_integer_columns_schema
from tests.spark.assert_df import AssertDf

pytestmark = pytest.mark.usefixtures("spark_session")


def test_should_return_rows_that_pass_all_checks_and_reject_rows_that_violate_any_test(spark_session):
    not_between = [25, 1]
    max_exceeded = [3, 30]
    correct = [3, 15]
    less_than_min = [1, 15]
    both_wrong = [7, 30]

    df = spark_session.createDataFrame([not_between, max_exceeded, correct, less_than_min, both_wrong], schema=two_integer_columns_schema)
    expected_correct = spark_session.createDataFrame([correct], schema=two_integer_columns_schema)
    expected_errors = spark_session.createDataFrame([not_between, max_exceeded, less_than_min, both_wrong], schema=two_integer_columns_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .is_between("col1", 0, 5) \
        .is_min("col1", 3) \
        .is_max("col2", 20) \
        .execute()

    AssertDf(result.correct_data, order_by_column="col1") \
        .contains_exactly(expected_correct.toPandas()) \
        .has_columns(["col1", "col2"])

    AssertDf(result.erroneous_data, order_by_column="col2") \
        .contains_exactly(expected_errors.toPandas()) \
        .has_columns(["col1", "col2"])

    assert result.errors == [ValidationError("col1", "between", 2), ValidationError("col1", "min", 1), ValidationError("col2", "max", 2)]
