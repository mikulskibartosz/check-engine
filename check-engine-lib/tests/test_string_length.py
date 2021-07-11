"""
Tests the textLength constraint.
"""
import pytest

from tests.spark import single_string_column_schema, two_string_columns_schema, empty_string_df, empty_integer_df
from tests.spark.AssertResult import AssertValidationResult
from tests.spark.assert_df import AssertDf
from checkengine.validate_df import ValidateSparkDataFrame, ValidationError

pytestmark = pytest.mark.usefixtures("spark_session")


def test_should_return_df_without_changes_if_empty_df_with_is_text_length_constraint(spark_session):
    df = empty_string_df(spark_session)

    result = ValidateSparkDataFrame(spark_session, df) \
        .has_length_between("col1", 0, 20) \
        .execute()

    AssertValidationResult(column_name="col1", constraint_name="text_length") \
        .check(
        actual=result,
        expected_correct=df,
        expected_erroneous=df
    )


def test_should_return_df_without_changes_if_all_are_shorter_than_upper_bound(spark_session):
    df = spark_session.createDataFrame([["abc"], ["def"], ["ghi"]], schema=single_string_column_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .has_length_between("col1", 0, 20) \
        .execute()

    AssertValidationResult(column_name="col1", constraint_name="text_length") \
        .check(
        actual=result,
        expected_correct=df,
        expected_erroneous=empty_string_df(spark_session)
    )


def test_should_return_df_without_changes_if_all_are_longer_than_lower_bound(spark_session):
    df = spark_session.createDataFrame([["abcdef"], ["ghijkl"]], schema=single_string_column_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .has_length_between("col1", 5, 20) \
        .execute()

    AssertValidationResult(column_name="col1", constraint_name="text_length") \
        .check(
        actual=result,
        expected_correct=df,
        expected_erroneous=empty_string_df(spark_session)
    )


def test_should_reject_all_rows_if_all_are_too_short_or_too_long(spark_session):
    df = spark_session.createDataFrame([["abc"], ["a"], ["abcdefghi"]], schema=single_string_column_schema)
    expected_errors = spark_session.createDataFrame([["abc"], ["a"], ["abcdefghi"]], schema=single_string_column_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .has_length_between("col1", 5, 8) \
        .execute()

    AssertValidationResult(column_name="col1", constraint_name="text_length") \
        .check(
        actual=result,
        expected_correct=empty_string_df(spark_session),
        expected_erroneous=expected_errors
    )


def test_should_return_both_correct_and_incorrect_rows(spark_session):
    df = spark_session.createDataFrame([["a"], ["abc"], ["defg"], ["hijkl"]], schema=single_string_column_schema)

    expected_correct = spark_session.createDataFrame([["abc"], ["defg"]], schema=single_string_column_schema)
    expected_errors = spark_session.createDataFrame([["a"], ["hijkl"]], schema=single_string_column_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .has_length_between("col1", 3, 4) \
        .execute()

    AssertValidationResult(column_name="col1", constraint_name="text_length") \
        .check(
        actual=result,
        expected_correct=expected_correct,
        expected_erroneous=expected_errors
    )


def test_text_length_of_other_columns_is_ignored(spark_session):
    df = spark_session.createDataFrame([["a", "123"], ["bcd", "45"], ["cd", "12345"]], schema=two_string_columns_schema)

    expected_correct = spark_session.createDataFrame([["cd", "12345"]], schema=two_string_columns_schema)
    expected_errors = spark_session.createDataFrame([["a", "123"], ["bcd", "45"]], schema=two_string_columns_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .has_length_between("col1", 2, 2) \
        .execute()

    AssertValidationResult(column_name="col1", constraint_name="text_length") \
        .check(
        actual=result,
        expected_correct=expected_correct,
        expected_erroneous=expected_errors
    )


def test_should_check_all_given_columns_separately(spark_session):
    df = spark_session.createDataFrame([["a", "12"], ["abcde", "56"], ["def", "123"]], schema=two_string_columns_schema)

    expected_correct = spark_session.createDataFrame([], schema=two_string_columns_schema)
    expected_errors = spark_session.createDataFrame([["a", "12"], ["abcde", "56"], ["def", "123"]], schema=two_string_columns_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .has_length_between("col1", 2, 4) \
        .has_length_between("col2", 1, 2) \
        .execute()

    AssertDf(result.correct_data, order_by_column="col1") \
        .contains_exactly(expected_correct.toPandas()) \
        .has_columns(["col1", "col2"])

    AssertDf(result.erroneous_data, order_by_column="col2") \
        .contains_exactly(expected_errors.toPandas()) \
        .has_columns(["col1", "col2"])

    assert result.errors == [ValidationError("col1", "text_length", 2), ValidationError("col2", "text_length", 1)]


def test_should_throw_error_if_constraint_uses_non_existing_column(spark_session):
    with pytest.raises(ValueError):
        ValidateSparkDataFrame(spark_session, empty_string_df(spark_session)) \
            .has_length_between("column_that_does_not_exist", 0, 1) \
            .execute()


def test_should_throw_error_if_there_are_duplicate_constraints(spark_session):
    with pytest.raises(ValueError):
        ValidateSparkDataFrame(spark_session, empty_string_df(spark_session)) \
            .has_length_between("col1", 0, 10) \
            .has_length_between("col1", 0, 5) \
            .execute()


def test_should_throw_error_if_lower_bound_is_greater_than_upper_bound(spark_session):
    with pytest.raises(ValueError):
        ValidateSparkDataFrame(spark_session, empty_string_df(spark_session)) \
            .has_length_between("col1", 10, 5) \
            .execute()


def test_should_throw_error_if_constraint_is_not_a_text_column(spark_session):
    with pytest.raises(ValueError):
        ValidateSparkDataFrame(spark_session, empty_integer_df(spark_session)) \
            .has_length_between("col1", 5, 10) \
            .execute()
