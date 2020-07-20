"""
Tests the Regex matching constraint.
"""
import pytest

from tests.spark import empty_string_df, single_string_column_schema, two_string_columns_schema, empty_integer_df
from tests.spark.AssertResult import AssertValidationResult
from tests.spark.assert_df import AssertDf
from pyspark_check.validate_df import ValidateSparkDataFrame, ValidationError

pytestmark = pytest.mark.usefixtures("spark_session")


def test_should_return_df_without_changes_if_empty_df_with_is_text_matches_regex_constraint(spark_session):
    df = empty_string_df(spark_session)

    result = ValidateSparkDataFrame(spark_session, df) \
        .text_matches_regex("col1", ".*") \
        .execute()

    AssertValidationResult(column_name="col1", constraint_name="regex_match") \
        .check(
        actual=result,
        expected_correct=df,
        expected_erroneous=df
    )


def test_should_return_df_without_changes_if_regex_matches_the_text(spark_session):
    df = spark_session.createDataFrame([["abc"], ["def"], ["ghi"]], schema=single_string_column_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .text_matches_regex("col1", ".*") \
        .execute()

    AssertValidationResult(column_name="col1", constraint_name="regex_match") \
        .check(
        actual=result,
        expected_correct=df,
        expected_erroneous=empty_string_df(spark_session)
    )


def test_should_reject_all_rows_if_regex_match_fails(spark_session):
    df = spark_session.createDataFrame([["abc"], ["a"], ["abcdefghi"]], schema=single_string_column_schema)
    expected_errors = spark_session.createDataFrame([["abc"], ["a"], ["abcdefghi"]], schema=single_string_column_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .text_matches_regex("col1", "[0-9]+") \
        .execute()

    AssertValidationResult(column_name="col1", constraint_name="regex_match") \
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
        .text_matches_regex("col1", "^[a-z]{3,4}$") \
        .execute()

    AssertValidationResult(column_name="col1", constraint_name="regex_match") \
        .check(
        actual=result,
        expected_correct=expected_correct,
        expected_erroneous=expected_errors
    )


def test_matching_of_other_columns_is_ignored(spark_session):
    df = spark_session.createDataFrame([["a", "123"], ["bcd", "45"], ["cd", "12345"]], schema=two_string_columns_schema)

    expected_correct = spark_session.createDataFrame([["cd", "12345"]], schema=two_string_columns_schema)
    expected_errors = spark_session.createDataFrame([["a", "123"], ["bcd", "45"]], schema=two_string_columns_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .text_matches_regex("col1", "^[cd]+$") \
        .execute()

    AssertValidationResult(column_name="col1", constraint_name="regex_match") \
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
        .text_matches_regex("col1", "[0-9]+") \
        .text_matches_regex("col2", "[a-z]+") \
        .execute()

    AssertDf(result.correct_data, order_by_column="col1") \
        .contains_exactly(expected_correct.toPandas()) \
        .has_columns(["col1", "col2"])

    AssertDf(result.erroneous_data, order_by_column="col2") \
        .contains_exactly(expected_errors.toPandas()) \
        .has_columns(["col1", "col2"])

    assert result.errors == [ValidationError("col1", "regex_match", 3), ValidationError("col2", "regex_match", 3)]


def test_should_throw_error_if_constraint_uses_non_existing_column(spark_session):
    with pytest.raises(ValueError):
        ValidateSparkDataFrame(spark_session, empty_string_df(spark_session)) \
            .text_matches_regex("column_that_does_not_exist", '.*') \
            .execute()


def test_should_throw_error_if_there_are_duplicate_constraints(spark_session):
    with pytest.raises(ValueError):
        ValidateSparkDataFrame(spark_session, empty_string_df(spark_session)) \
            .text_matches_regex("column_that_does_not_exist", '.*') \
            .text_matches_regex("column_that_does_not_exist", '[a-z]*') \
            .execute()


def test_should_throw_error_if_constraint_is_not_a_text_column(spark_session):
    with pytest.raises(ValueError):
        ValidateSparkDataFrame(spark_session, empty_integer_df(spark_session)) \
            .text_matches_regex("col1", '[a-z]*') \
            .execute()
