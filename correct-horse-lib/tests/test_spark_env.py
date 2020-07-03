"""
Tests that verify whether the PySpark instance used for testing is configured properly.
"""
import pytest
import logging

import pandas as pd
from pyspark.sql import DataFrame

from pyspark.sql.types import *

from tests.spark.assert_df import AssertDf

pytestmark = pytest.mark.usefixtures("spark_session")

logger = logging.getLogger('test-spark-env')


def test_empty_dataframe(spark_session):
    df_schema = StructType([StructField("col1", StringType())])

    df = spark_session.createDataFrame([], schema=df_schema)
    AssertDf(df).is_empty()


def test_spark_sql_operation(spark_session):
    df_schema = StructType([StructField("col1", StringType()), StructField("col2", IntegerType())])

    test_list = [["v1", 1], ["v1", 2], ["v2", 3]]

    df: DataFrame = spark_session.createDataFrame(test_list, schema=df_schema)
    aggregated = df.groupby("col1").sum("col2").orderBy('col1')

    AssertDf(aggregated) \
        .contains_exactly(pd.DataFrame([['v1', 3], ['v2', 3]], columns=['col1', 'sum(col2)']).sort_values('col1')) \
        .has_columns(["col1", "sum(col2)"]) \
        .has_n_rows(2)
