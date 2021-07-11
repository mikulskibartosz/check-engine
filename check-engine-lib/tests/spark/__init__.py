from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

single_string_column_schema = StructType([StructField("col1", StringType())])
two_string_columns_schema = StructType([StructField("col1", StringType()), StructField("col2", StringType())])

single_integer_column_schema = StructType([StructField("col1", IntegerType())])
two_integer_columns_schema = StructType([StructField("col1", IntegerType()), StructField("col2", IntegerType())])


def empty_string_df(spark_session: SparkSession):
    return spark_session.createDataFrame([], schema=single_string_column_schema)


def empty_integer_df(spark_session: SparkSession):
    return spark_session.createDataFrame([], schema=single_integer_column_schema)
