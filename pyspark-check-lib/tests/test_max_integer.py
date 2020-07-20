"""
Verifies the constraints that check whether the column contains value equal or smaller than a given integer.
"""

import pytest

from tests.spark import empty_integer_df, single_integer_column_schema, two_integer_columns_schema, empty_string_df
from tests.spark.AssertResult import AssertValidationResult
from tests.spark.assert_df import AssertDf
from pyspark_check.validate_df import ValidateSparkDataFrame, ValidationError

pytestmark = pytest.mark.usefixtures("spark_session")


def test_should_return_df_without_changes_if_empty_df_with_is_max_constraint(spark_session):
    df = empty_integer_df(spark_session)

    result = ValidateSparkDataFrame(spark_session, df) \
        .is_max("col1", 5) \
        .execute()

    AssertValidationResult(column_name="col1", constraint_name="max") \
        .check(
        actual=result,
        expected_correct=df,
        expected_erroneous=df
    )


def test_should_return_df_without_changes_if_all_rows_smaller_than_max(spark_session):
    df = spark_session.createDataFrame([[5], [10], [15]], schema=single_integer_column_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .is_max("col1", 20) \
        .execute()

    AssertValidationResult(column_name="col1", constraint_name="max") \
        .check(
        actual=result,
        expected_correct=df,
        expected_erroneous=empty_integer_df(spark_session)
    )


def test_should_reject_all_rows_if_larger_than_max(spark_session):
    df = spark_session.createDataFrame([[5], [10], [15]], schema=single_integer_column_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .is_max("col1", 1) \
        .execute()

    AssertValidationResult(column_name="col1", constraint_name="max") \
        .check(
        actual=result,
        expected_correct=empty_integer_df(spark_session),
        expected_erroneous=df
    )


def test_should_return_both_correct_and_incorrect_rows(spark_session):
    df = spark_session.createDataFrame([[5], [10], [15]], schema=single_integer_column_schema)
    expected_correct = spark_session.createDataFrame([[5], [10]], schema=single_integer_column_schema)
    expected_errors = spark_session.createDataFrame([[15]], schema=single_integer_column_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .is_max("col1", 10) \
        .execute()

    AssertValidationResult(column_name="col1", constraint_name="max") \
        .check(
        actual=result,
        expected_correct=expected_correct,
        expected_erroneous=expected_errors
    )


def test_max_value_of_other_columns_is_ignored(spark_session):
    df = spark_session.createDataFrame([[5, 1], [10, 20], [15, 1]], schema=two_integer_columns_schema)
    expected_correct = spark_session.createDataFrame([[5, 1], [10, 20]], schema=two_integer_columns_schema)
    expected_errors = spark_session.createDataFrame([[15, 1]], schema=two_integer_columns_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .is_max("col1", 10) \
        .execute()

    AssertValidationResult(column_name="col1", constraint_name="max") \
        .check(
        actual=result,
        expected_correct=expected_correct,
        expected_erroneous=expected_errors
    )


def test_max_should_check_all_given_columns_separately(spark_session):
    df = spark_session.createDataFrame([[25, 1], [30, 2], [35, 3]], schema=two_integer_columns_schema)

    expected_correct = spark_session.createDataFrame([], schema=two_integer_columns_schema)
    expected_errors = spark_session.createDataFrame([[25, 1], [30, 2], [35, 3]], schema=two_integer_columns_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .is_max("col1", 20) \
        .is_max("col2", 0) \
        .execute()

    AssertDf(result.correct_data, order_by_column="col1") \
        .contains_exactly(expected_correct.toPandas()) \
        .has_columns(["col1", "col2"])

    AssertDf(result.erroneous_data, order_by_column="col2") \
        .contains_exactly(expected_errors.toPandas()) \
        .has_columns(["col1", "col2"])

    assert result.errors == [ValidationError("col1", "max", 3), ValidationError("col2", "max", 3)]


def test_should_throw_error_if_constraint_is_not_a_numeric_column(spark_session):
    with pytest.raises(ValueError):
        ValidateSparkDataFrame(spark_session, empty_string_df(spark_session)) \
            .is_max("col1", 5) \
            .execute()


def test_should_throw_error_if_constraint_uses_non_existing_column(spark_session):
    with pytest.raises(ValueError):
        ValidateSparkDataFrame(spark_session, empty_integer_df(spark_session)) \
            .is_max("column_that_does_not_exist", 5) \
            .execute()


def test_should_throw_error_if_there_are_duplicate_constraints(spark_session):
    with pytest.raises(ValueError):
        ValidateSparkDataFrame(spark_session, empty_integer_df(spark_session)) \
            .is_max("col1", 5) \
            .is_max("col1", 10) \
            .execute()
