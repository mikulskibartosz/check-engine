"""
Tests the behavior of ValidateSparkDataFrame when no constraint has been defined.
In that case, the implementation should pass all of the given data as correct and don't return any errors.
"""
import pytest

from tests.spark import empty_string_df, single_string_column_schema
from tests.spark.AssertResult import AssertValidationResult
from checkengine.validate_df import ValidateSparkDataFrame

pytestmark = pytest.mark.usefixtures("spark_session")


def test_should_pass_empty_df_if_there_are_no_rules(spark_session):
    df = empty_string_df(spark_session)

    result = ValidateSparkDataFrame(spark_session, df).execute()

    AssertValidationResult(column_name="col1", constraint_name="") \
        .check(
        actual=result,
        expected_correct=df,
        expected_erroneous=df
    )


def test_should_pass_df_if_there_are_no_rules(spark_session):
    df = spark_session.createDataFrame([["abc"], ["def"]], schema=single_string_column_schema)

    result = ValidateSparkDataFrame(spark_session, df).execute()

    AssertValidationResult(column_name="col1", constraint_name="") \
        .check(
        actual=result,
        expected_correct=df,
        expected_erroneous=empty_string_df(spark_session)
    )
