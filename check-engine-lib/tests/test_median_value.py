"""
Tests the medianColumnValue constraint.

The implementation should reject all rows in a column if the column median value is not between the expected values.
"""
import pytest

from checkengine.validate_df import ValidateSparkDataFrame
from tests.spark import empty_integer_df, single_integer_column_schema, two_integer_columns_schema, empty_string_df
from tests.spark.AssertResult import AssertValidationResult

pytestmark = pytest.mark.usefixtures("spark_session")


def test_should_return_df_without_changes_if_empty_df_with_median_constraint(spark_session):
    df = empty_integer_df(spark_session)

    result = ValidateSparkDataFrame(spark_session, df) \
        .median_column_value("col1", 0, 1) \
        .execute()

    AssertValidationResult(column_name="col1", constraint_name="median_between") \
        .check(
        actual=result,
        expected_correct=df,
        expected_erroneous=df
    )


def test_should_return_df_without_changes_if_the_median_is_between_given_values(spark_session):
    df = spark_session.createDataFrame([[5], [10], [15]], schema=single_integer_column_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .median_column_value("col1", 5, 15) \
        .execute()

    AssertValidationResult(column_name="col1", constraint_name="median_between") \
        .check(
        actual=result,
        expected_correct=df,
        expected_erroneous=empty_integer_df(spark_session)
    )


def test_should_reject_all_rows_if_median_is_smaller_than_given_values(spark_session):
    df = spark_session.createDataFrame([[5], [10], [15]], schema=single_integer_column_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .median_column_value("col1", 12, 15) \
        .execute()

    AssertValidationResult(column_name="col1", constraint_name="median_between") \
        .check(
        actual=result,
        expected_correct=empty_integer_df(spark_session),
        expected_erroneous=df
    )


def test_should_reject_all_rows_if_median_is_larger_than_given_values(spark_session):
    df = spark_session.createDataFrame([[5], [10], [15]], schema=single_integer_column_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .median_column_value("col1", 5, 8) \
        .execute()

    AssertValidationResult(column_name="col1", constraint_name="median_between") \
        .check(
        actual=result,
        expected_correct=empty_integer_df(spark_session),
        expected_erroneous=df
    )


def test_median_value_of_other_columns_is_ignored(spark_session):
    df = spark_session.createDataFrame([[5, 1], [10, 2], [15, 3]], schema=two_integer_columns_schema)
    expected_errors = spark_session.createDataFrame([], schema=two_integer_columns_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .median_column_value("col1", 10, 10) \
        .execute()

    AssertValidationResult(column_name="col1", constraint_name="median_between") \
        .check(
        actual=result,
        expected_correct=df,
        expected_erroneous=expected_errors
    )


def test_median_should_check_all_given_columns_separately(spark_session):
    df = spark_session.createDataFrame([[5, 1], [10, 2], [15, 3]], schema=two_integer_columns_schema)
    expected_errors = spark_session.createDataFrame([], schema=two_integer_columns_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .median_column_value("col1", 10, 10) \
        .median_column_value("col2", 2, 2) \
        .execute()

    AssertValidationResult(column_name="col1", constraint_name="median_between") \
        .check(
        actual=result,
        expected_correct=df,
        expected_erroneous=expected_errors
    )


def test_should_throw_error_if_constraint_is_not_a_numeric_column(spark_session):
    with pytest.raises(ValueError):
        ValidateSparkDataFrame(spark_session, empty_string_df(spark_session)) \
            .median_column_value("col1", 10, 10) \
            .execute()


def test_should_throw_error_if_constraint_uses_non_existing_column(spark_session):
    with pytest.raises(ValueError):
        ValidateSparkDataFrame(spark_session, empty_integer_df(spark_session)) \
            .median_column_value("column_that_does_not_exist", 5, 5) \
            .execute()


def test_should_throw_error_if_there_are_duplicate_constraints(spark_session):
    with pytest.raises(ValueError):
        ValidateSparkDataFrame(spark_session, empty_integer_df(spark_session)) \
            .median_column_value("col1", 10, 10) \
            .median_column_value("col1", 5, 5) \
            .execute()
