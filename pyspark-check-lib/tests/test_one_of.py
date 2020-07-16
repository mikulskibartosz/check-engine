"""
Tests the one_of constraint.
"""
import pytest

from pyspark.sql.types import *

from tests.spark.assert_df import AssertDf
from pyspark_check.validate_df import ValidateSparkDataFrame, ValidationError

pytestmark = pytest.mark.usefixtures("spark_session")


def test_should_return_df_without_changes_if_empty_df_with_one_of_constraint(spark_session):
    df_schema = StructType([StructField("col1", StringType())])
    df = spark_session.createDataFrame([], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .one_of("col1", []) \
        .execute()

    AssertDf(result.correct_data)\
        .is_empty()\
        .has_columns(["col1"])

    AssertDf(result.erroneous_data)\
        .is_empty()\
        .has_columns(["col1"])

    assert result.errors == []


def test_should_return_df_without_changes_if_all_are_in_list(spark_session):
    df_schema = StructType([StructField("col1", StringType())])
    df = spark_session.createDataFrame([["abc"], ["def"], ["ghi"]], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .one_of("col1", ["abc", "def", "ghi"]) \
        .execute()

    AssertDf(result.correct_data, order_by_column="col1") \
        .contains_exactly(df.toPandas()) \
        .has_columns(["col1"])

    AssertDf(result.erroneous_data) \
        .is_empty() \
        .has_columns(["col1"])

    assert result.errors == []


def test_should_reject_all_rows_if_none_of_them_is_in_the_list(spark_session):
    df_schema = StructType([StructField("col1", StringType())])
    df = spark_session.createDataFrame([["abc"], ["a"], ["abcdefghi"]], schema=df_schema)
    expected_errors = spark_session.createDataFrame([["abc"], ["a"], ["abcdefghi"]], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .one_of("col1", ["ab", "b"]) \
        .execute()

    AssertDf(result.correct_data) \
        .is_empty() \
        .has_columns(["col1"])

    AssertDf(result.erroneous_data, order_by_column="col1") \
        .contains_exactly(expected_errors.toPandas()) \
        .has_columns(["col1"])

    assert result.errors == [ValidationError("col1", "one_of", 3)]


def test_should_return_both_correct_and_incorrect_rows(spark_session):
    df_schema = StructType([StructField("col1", StringType())])
    df = spark_session.createDataFrame([["a"], ["abc"], ["defg"], ["hijkl"]], schema=df_schema)

    expected_correct = spark_session.createDataFrame([["abc"], ["defg"]], schema=df_schema)
    expected_errors = spark_session.createDataFrame([["a"], ["hijkl"]], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .one_of("col1", ["abc", "defg"]) \
        .execute()

    AssertDf(result.correct_data, order_by_column="col1") \
        .contains_exactly(expected_correct.toPandas()) \
        .has_columns(["col1"])

    AssertDf(result.erroneous_data, order_by_column="col1") \
        .contains_exactly(expected_errors.toPandas()) \
        .has_columns(["col1"])

    assert result.errors == [ValidationError("col1", "one_of", 2)]


def test_should_return_both_correct_and_incorrect_rows_numeric_values(spark_session):
    df_schema = StructType([StructField("col1", IntegerType())])
    df = spark_session.createDataFrame([[1], [2], [3], [4]], schema=df_schema)

    expected_correct = spark_session.createDataFrame([[1], [3]], schema=df_schema)
    expected_errors = spark_session.createDataFrame([[2], [4]], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .one_of("col1", [1, 3, 5]) \
        .execute()

    AssertDf(result.correct_data, order_by_column="col1") \
        .contains_exactly(expected_correct.toPandas()) \
        .has_columns(["col1"])

    AssertDf(result.erroneous_data, order_by_column="col1") \
        .contains_exactly(expected_errors.toPandas()) \
        .has_columns(["col1"])

    assert result.errors == [ValidationError("col1", "one_of", 2)]


def test_one_of_of_other_columns_is_ignored(spark_session):
    df_schema = StructType([StructField("col1", StringType()), StructField("col2", StringType())])
    df = spark_session.createDataFrame([["a", "123"], ["bcd", "45"], ["cd", "12345"]], schema=df_schema)

    expected_correct = spark_session.createDataFrame([["cd", "12345"]], schema=df_schema)
    expected_errors = spark_session.createDataFrame([["a", "123"], ["bcd", "45"]], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .one_of("col1", ["cd", "123", "45"]) \
        .execute()

    AssertDf(result.correct_data, order_by_column="col1") \
        .contains_exactly(expected_correct.toPandas()) \
        .has_columns(["col1", "col2"])

    AssertDf(result.erroneous_data, order_by_column="col2") \
        .contains_exactly(expected_errors.toPandas()) \
        .has_columns(["col1", "col2"])

    assert result.errors == [ValidationError("col1", "one_of", 2)]


def test_should_check_all_given_columns_separately(spark_session):
    df_schema = StructType([StructField("col1", StringType()), StructField("col2", StringType())])
    df = spark_session.createDataFrame([["a", "12"], ["abcde", "56"], ["def", "123"]], schema=df_schema)

    expected_correct = spark_session.createDataFrame([], schema=df_schema)
    expected_errors = spark_session.createDataFrame([["a", "12"], ["abcde", "56"], ["def", "123"]], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .one_of("col1", ["12", "56", "def"]) \
        .one_of("col2", ["12", "56", "adcde"]) \
        .execute()

    AssertDf(result.correct_data, order_by_column="col1") \
        .contains_exactly(expected_correct.toPandas()) \
        .has_columns(["col1", "col2"])

    AssertDf(result.erroneous_data, order_by_column="col2") \
        .contains_exactly(expected_errors.toPandas()) \
        .has_columns(["col1", "col2"])

    assert result.errors == [ValidationError("col1", "one_of", 2), ValidationError("col2", "one_of", 1)]


def test_should_throw_error_if_constraint_uses_non_existing_column(spark_session):
    df_schema = StructType([StructField("col1", StringType())])
    df = spark_session.createDataFrame([], schema=df_schema)

    with pytest.raises(ValueError):
        ValidateSparkDataFrame(spark_session, df) \
            .one_of("column_that_does_not_exist", []) \
            .execute()


def test_should_throw_error_if_there_are_duplicate_constraints(spark_session):
    df_schema = StructType([StructField("col1", StringType())])
    df = spark_session.createDataFrame([], schema=df_schema)

    with pytest.raises(ValueError):
        ValidateSparkDataFrame(spark_session, df) \
            .one_of("col1", ["a"]) \
            .one_of("col1", ["b"]) \
            .execute()


