"""
Verifies the constraints that check whether the column contains value equal or smaller than a given integer.
"""

import pytest

from pyspark.sql.types import *

from tests.spark.assert_df import AssertDf
from pyspark_check.validate_df import ValidateSparkDataFrame, ValidationError

pytestmark = pytest.mark.usefixtures("spark_session")


def test_should_return_df_without_changes_if_empty_df_with_is_max_constraint(spark_session):
    df_schema = StructType([StructField("col1", IntegerType())])
    df = spark_session.createDataFrame([], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .is_max("col1", 5) \
        .execute()

    AssertDf(result.correct_data) \
        .is_empty() \
        .has_columns(["col1"])

    AssertDf(result.erroneous_data) \
        .is_empty() \
        .has_columns(["col1"])

    assert result.errors == []


def test_should_return_df_without_changes_if_all_rows_smaller_than_max(spark_session):
    df_schema = StructType([StructField("col1", IntegerType())])
    df = spark_session.createDataFrame([[5], [10], [15]], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .is_max("col1", 20) \
        .execute()

    AssertDf(result.correct_data, order_by_column="col1") \
        .contains_exactly(df.toPandas()) \
        .has_columns(["col1"])

    AssertDf(result.erroneous_data) \
        .is_empty() \
        .has_columns(["col1"])

    assert result.errors == []


def test_should_reject_all_rows_if_larger_than_max(spark_session):
    df_schema = StructType([StructField("col1", IntegerType())])
    df = spark_session.createDataFrame([[5], [10], [15]], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .is_max("col1", 1) \
        .execute()

    AssertDf(result.correct_data) \
        .is_empty() \
        .has_columns(["col1"])

    AssertDf(result.erroneous_data, order_by_column="col1") \
        .contains_exactly(df.toPandas()) \
        .has_columns(["col1"])

    assert result.errors == [ValidationError("col1", "max", 3)]


def test_should_return_both_correct_and_incorrect_rows(spark_session):
    df_schema = StructType([StructField("col1", IntegerType())])
    df = spark_session.createDataFrame([[5], [10], [15]], schema=df_schema)

    expected_correct = spark_session.createDataFrame([[5], [10]], schema=df_schema)
    expected_errors = spark_session.createDataFrame([[15]], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .is_max("col1", 10) \
        .execute()

    AssertDf(result.correct_data, order_by_column="col1") \
        .contains_exactly(expected_correct.toPandas()) \
        .has_columns(["col1"])

    AssertDf(result.erroneous_data, order_by_column="col1") \
        .contains_exactly(expected_errors.toPandas()) \
        .has_columns(["col1"])

    assert result.errors == [ValidationError("col1", "max", 1)]


def test_max_value_of_other_columns_is_ignored(spark_session):
    df_schema = StructType([StructField("col1", IntegerType()), StructField("col2", IntegerType())])
    df = spark_session.createDataFrame([[5, 1], [10, 20], [15, 1]], schema=df_schema)

    expected_correct = spark_session.createDataFrame([[5, 1], [10, 20]], schema=df_schema)
    expected_errors = spark_session.createDataFrame([[15, 1]], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .is_max("col1", 10) \
        .execute()

    AssertDf(result.correct_data, order_by_column="col1") \
        .contains_exactly(expected_correct.toPandas()) \
        .has_columns(["col1", "col2"])

    AssertDf(result.erroneous_data, order_by_column="col2") \
        .contains_exactly(expected_errors.toPandas()) \
        .has_columns(["col1", "col2"])

    assert result.errors == [ValidationError("col1", "max", 1)]


def test_max_should_check_all_given_columns_separately(spark_session):
    df_schema = StructType([StructField("col1", IntegerType()), StructField("col2", IntegerType())])
    df = spark_session.createDataFrame([[25, 1], [30, 2], [35, 3]], schema=df_schema)

    expected_correct = spark_session.createDataFrame([], schema=df_schema)
    expected_errors = spark_session.createDataFrame([[25, 1], [30, 2], [35, 3]], schema=df_schema)

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
    df_schema = StructType([StructField("col1", StringType())])
    df = spark_session.createDataFrame([], schema=df_schema)

    with pytest.raises(ValueError):
        ValidateSparkDataFrame(spark_session, df) \
            .is_max("col1", 5) \
            .execute()


def test_should_throw_error_if_constraint_uses_non_existing_column(spark_session):
    df_schema = StructType([StructField("col1", IntegerType())])
    df = spark_session.createDataFrame([], schema=df_schema)

    with pytest.raises(ValueError):
        ValidateSparkDataFrame(spark_session, df) \
            .is_max("column_that_does_not_exist", 5) \
            .execute()


def test_should_throw_error_if_there_are_duplicate_constraints(spark_session):
    df_schema = StructType([StructField("col1", IntegerType())])
    df = spark_session.createDataFrame([], schema=df_schema)

    with pytest.raises(ValueError):
        ValidateSparkDataFrame(spark_session, df) \
            .is_max("col1", 5) \
            .is_max("col1", 10) \
            .execute()
