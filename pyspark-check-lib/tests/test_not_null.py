"""
Tests the not null constraint.
"""
import pytest

from pyspark.sql.types import *

from tests.spark.assert_df import AssertDf
from pyspark_check.validate_df import ValidateSparkDataFrame, ValidationError

pytestmark = pytest.mark.usefixtures("spark_session")


def test_should_pass_empty_df_with_not_null_constraint(spark_session):
    df_schema = StructType([StructField("col1", StringType())])
    df = spark_session.createDataFrame([], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .is_not_null("col1") \
        .execute()

    AssertDf(result.correct_data)\
        .is_empty()\
        .has_columns(["col1"])

    AssertDf(result.erroneous_data)\
        .is_empty()\
        .has_columns(["col1"])

    assert result.errors == []


def test_should_return_df_without_changes_if_all_rows_are_not_null(spark_session):
    df_schema = StructType([StructField("col1", StringType())])
    df = spark_session.createDataFrame([["abc"], ["def"], ["ghi"]], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .is_not_null("col1") \
        .execute()

    AssertDf(result.correct_data, order_by_column="col1") \
        .contains_exactly(df.toPandas()) \
        .has_columns(["col1"])

    AssertDf(result.erroneous_data) \
        .is_empty() \
        .has_columns(["col1"])

    assert result.errors == []


def test_should_reject_all_rows_if_all_are_null(spark_session):
    df_schema = StructType([StructField("col1", StringType())])
    df = spark_session.createDataFrame([[None], [None], [None]], schema=df_schema)
    expected_errors = spark_session.createDataFrame([[None]], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .is_not_null("col1") \
        .execute()

    AssertDf(result.correct_data) \
        .is_empty() \
        .has_columns(["col1"])

    AssertDf(result.erroneous_data) \
        .contains_exactly(expected_errors.toPandas()) \
        .has_columns(["col1"])

    assert result.errors == [ValidationError("col1", "not_null", 3)]


def test_should_return_both_correct_and_incorrect_rows(spark_session):
    df_schema = StructType([StructField("col1", StringType())])
    df = spark_session.createDataFrame([["abc"], [None]], schema=df_schema)

    expected_correct = spark_session.createDataFrame([["abc"]], schema=df_schema)
    expected_errors = spark_session.createDataFrame([[None]], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .is_not_null("col1") \
        .execute()

    AssertDf(result.correct_data) \
        .contains_exactly(expected_correct.toPandas()) \
        .has_columns(["col1"])

    AssertDf(result.erroneous_data) \
        .contains_exactly(expected_errors.toPandas()) \
        .has_columns(["col1"])

    assert result.errors == [ValidationError("col1", "not_null", 1)]


def test_nulls_in_other_columns_are_ignored(spark_session):
    df_schema = StructType([StructField("col1", StringType()), StructField("col2", StringType())])
    df = spark_session.createDataFrame([["abc", "123"], [None, "456"], ["def", None]], schema=df_schema)

    expected_correct = spark_session.createDataFrame([["abc", "123"], ["def", None]], schema=df_schema)
    expected_errors = spark_session.createDataFrame([[None, "456"]], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .is_not_null("col1") \
        .execute()

    AssertDf(result.correct_data, order_by_column="col1") \
        .contains_exactly(expected_correct.toPandas()) \
        .has_columns(["col1", "col2"])

    AssertDf(result.erroneous_data, order_by_column="col2") \
        .contains_exactly(expected_errors.toPandas()) \
        .has_columns(["col1", "col2"])

    assert result.errors == [ValidationError("col1", "not_null", 1)]


def test_not_null_should_check_all_given_columns_separately(spark_session):
    df_schema = StructType([StructField("col1", StringType()), StructField("col2", StringType())])
    df = spark_session.createDataFrame([["abc", None], [None, "456"], [None, None]], schema=df_schema)

    expected_errors = spark_session.createDataFrame([["abc", None], [None, "456"], [None, None]], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .is_not_null("col1") \
        .is_not_null("col2") \
        .execute()

    AssertDf(result.correct_data) \
        .is_empty() \
        .has_columns(["col1", "col2"])

    AssertDf(result.erroneous_data, order_by_column=["col1", "col2"]) \
        .contains_exactly(expected_errors.toPandas()) \
        .has_columns(["col1", "col2"])

    assert result.errors == [ValidationError("col1", "not_null", 2), ValidationError("col2", "not_null", 2)]


def test_not_null_should_check_all_given_columns_separately_even_if_all_of_them_are_defined_at_once(spark_session):
    df_schema = StructType([StructField("col1", StringType()), StructField("col2", StringType())])
    df = spark_session.createDataFrame([["abc", None], [None, "456"], [None, None]], schema=df_schema)

    expected_errors = spark_session.createDataFrame([["abc", None], [None, "456"], [None, None]], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .are_not_null(["col1", "col2"]) \
        .execute()

    AssertDf(result.correct_data) \
        .is_empty() \
        .has_columns(["col1", "col2"])

    AssertDf(result.erroneous_data, order_by_column=["col1", "col2"]) \
        .contains_exactly(expected_errors.toPandas()) \
        .has_columns(["col1", "col2"])

    assert result.errors == [ValidationError("col1", "not_null", 2), ValidationError("col2", "not_null", 2)]


def test_should_throw_error_if_constraint_uses_non_existing_column(spark_session):
    df_schema = StructType([StructField("col1", StringType())])
    df = spark_session.createDataFrame([], schema=df_schema)

    with pytest.raises(ValueError):
        ValidateSparkDataFrame(spark_session, df) \
            .is_not_null("column_that_does_not_exist") \
            .execute()


def test_should_throw_error_if_there_are_duplicate_constraints(spark_session):
    df_schema = StructType([StructField("col1", StringType())])
    df = spark_session.createDataFrame([], schema=df_schema)

    with pytest.raises(ValueError):
        ValidateSparkDataFrame(spark_session, df) \
            .is_not_null("col1") \
            .is_not_null("col1") \
            .execute()
