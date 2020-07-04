"""
Tests the behavior of ValidateSparkDataFrame when no constraint has been defined.
In that case, the implementation should pass all of the given data as correct and don't return any errors.
"""
import pytest

from pyspark.sql.types import *

from tests.spark.assert_df import AssertDf
from correct_horse.validate_df import ValidateSparkDataFrame

pytestmark = pytest.mark.usefixtures("spark_session")


def test_should_pass_empty_df_if_there_are_no_rules(spark_session):
    df_schema = StructType([StructField("col1", StringType())])
    df = spark_session.createDataFrame([], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df).execute()

    AssertDf(result.correct_data)\
        .is_empty()\
        .has_columns(["col1"])

    AssertDf(result.erroneous_data)\
        .is_empty()\
        .has_columns(["col1"])

    assert result.errors == []


def test_should_pass_df_if_there_are_no_rules(spark_session):
    df_schema = StructType([StructField("col1", StringType())])
    df = spark_session.createDataFrame([["abc"], ["def"]], schema=df_schema)

    result = ValidateSparkDataFrame(spark_session, df).execute()

    AssertDf(result.correct_data) \
        .has_n_rows(2) \
        .has_columns(["col1"]) \
        .contains_exactly(df.toPandas())

    AssertDf(result.erroneous_data) \
        .is_empty() \
        .has_columns(["col1"])

    assert result.errors == []
