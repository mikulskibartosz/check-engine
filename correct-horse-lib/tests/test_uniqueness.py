"""
Tests the isUnique constraint.

The implementation should reject a row if there is another row that contains the same value in the given column.
In that case, the rows should be reported as an error (only once).
"""
import pytest

from pyspark.sql.types import *

from tests.spark.assert_df import AssertDf
from correct_horse.validate_df import ValidateSparkDataFrame, ValidationError

pytestmark = pytest.mark.usefixtures("spark_session")


def test_should_return_df_without_changes_if_empty_df_with_is_unique_constraint(spark_session):
    df_schema = StructType([StructField("col1", StringType())])
    df = spark_session.createDataFrame([], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .is_unique("col1") \
        .execute()

    AssertDf(result.correct_data)\
        .is_empty()\
        .has_columns(["col1"])

    AssertDf(result.erroneous_data)\
        .is_empty()\
        .has_columns(["col1"])

    assert result.errors == []


def test_should_return_df_without_changes_if_all_rows_are_unique(spark_session):
    df_schema = StructType([StructField("col1", StringType())])
    df = spark_session.createDataFrame([["abc"], ["def"], ["ghi"]], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .is_unique("col1") \
        .execute()

    AssertDf(result.correct_data, order_by_column="col1") \
        .contains_exactly(df.toPandas()) \
        .has_columns(["col1"])

    AssertDf(result.erroneous_data) \
        .is_empty() \
        .has_columns(["col1"])

    assert result.errors == []


def test_should_reject_all_rows_if_all_are_the_same(spark_session):
    df_schema = StructType([StructField("col1", StringType())])
    df = spark_session.createDataFrame([["abc"], ["abc"], ["abc"]], schema=df_schema)
    expected_errors = spark_session.createDataFrame([["abc"]], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .is_unique("col1") \
        .execute()

    AssertDf(result.correct_data) \
        .is_empty() \
        .has_columns(["col1"])

    AssertDf(result.erroneous_data, order_by_column="col1") \
        .contains_exactly(expected_errors.toPandas()) \
        .has_columns(["col1"])

    assert result.errors == [ValidationError("col1", "unique", 3)]


def test_should_return_both_correct_and_incorrect_rows(spark_session):
    df_schema = StructType([StructField("col1", StringType())])
    df = spark_session.createDataFrame([["abc"], ["abc"], ["def"]], schema=df_schema)

    expected_correct = spark_session.createDataFrame([["def"]], schema=df_schema)
    expected_errors = spark_session.createDataFrame([["abc"]], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .is_unique("col1") \
        .execute()

    AssertDf(result.correct_data, order_by_column="col1") \
        .contains_exactly(expected_correct.toPandas()) \
        .has_columns(["col1"])

    AssertDf(result.erroneous_data, order_by_column="col1") \
        .contains_exactly(expected_errors.toPandas()) \
        .has_columns(["col1"])

    assert result.errors == [ValidationError("col1", "unique", 2)]


def test_uniqueness_of_other_columns_is_ignored(spark_session):
    df_schema = StructType([StructField("col1", StringType()), StructField("col2", StringType())])
    df = spark_session.createDataFrame([["abc", "123"], ["abc", "456"], ["def", "123"]], schema=df_schema)

    expected_correct = spark_session.createDataFrame([["def", "123"]], schema=df_schema)
    expected_errors = spark_session.createDataFrame([["abc", "123"], ["abc", "456"]], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .is_unique("col1") \
        .execute()

    AssertDf(result.correct_data, order_by_column="col1") \
        .contains_exactly(expected_correct.toPandas()) \
        .has_columns(["col1", "col2"])

    AssertDf(result.erroneous_data, order_by_column="col2") \
        .contains_exactly(expected_errors.toPandas()) \
        .has_columns(["col1", "col2"])

    assert result.errors == [ValidationError("col1", "unique", 2)]


def test_uniqueness_should_check_all_given_columns_separately(spark_session):
    df_schema = StructType([StructField("col1", StringType()), StructField("col2", StringType())])
    df = spark_session.createDataFrame([["abc", "123"], ["abc", "456"], ["def", "123"]], schema=df_schema)

    expected_correct = spark_session.createDataFrame([], schema=df_schema)
    expected_errors = spark_session.createDataFrame([["abc", "123"], ["abc", "456"], ["def", "123"]], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .is_unique("col1") \
        .is_unique("col2") \
        .execute()

    AssertDf(result.correct_data, order_by_column="col1") \
        .contains_exactly(expected_correct.toPandas()) \
        .has_columns(["col1", "col2"])

    AssertDf(result.erroneous_data, order_by_column="col2") \
        .contains_exactly(expected_errors.toPandas()) \
        .has_columns(["col1", "col2"])

    assert result.errors == [ValidationError("col1", "unique", 2), ValidationError("col2", "unique", 2)]


def test_uniqueness_should_check_all_given_columns_separately_when_defining_all_columns_at_once(spark_session):
    df_schema = StructType([StructField("col1", StringType()), StructField("col2", StringType())])
    df = spark_session.createDataFrame([["abc", "123"], ["abc", "456"], ["def", "123"]], schema=df_schema)

    expected_correct = spark_session.createDataFrame([], schema=df_schema)
    expected_errors = spark_session.createDataFrame([["abc", "123"], ["abc", "456"], ["def", "123"]], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df) \
        .are_unique(["col1", "col2"]) \
        .execute()

    AssertDf(result.correct_data, order_by_column="col1") \
        .contains_exactly(expected_correct.toPandas()) \
        .has_columns(["col1", "col2"])

    AssertDf(result.erroneous_data, order_by_column="col2") \
        .contains_exactly(expected_errors.toPandas()) \
        .has_columns(["col1", "col2"])

    assert result.errors == [ValidationError("col1", "unique", 2), ValidationError("col2", "unique", 2)]


def test_should_throw_error_if_constraint_uses_non_existing_column(spark_session):
    df_schema = StructType([StructField("col1", StringType())])
    df = spark_session.createDataFrame([], schema=df_schema)

    with pytest.raises(ValueError):
        ValidateSparkDataFrame(spark_session, df) \
            .is_unique("column_that_does_not_exist") \
            .execute()


def test_should_throw_error_if_there_are_duplicate_constraints(spark_session):
    df_schema = StructType([StructField("col1", StringType())])
    df = spark_session.createDataFrame([], schema=df_schema)

    with pytest.raises(ValueError):
        ValidateSparkDataFrame(spark_session, df) \
            .is_unique("col1") \
            .is_unique("col1") \
            .execute()
