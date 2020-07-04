from typing import NamedTuple, List

from pyspark.sql import DataFrame, SparkSession


class ValidationError(NamedTuple):
    column_name: str
    constraint_name: str
    number_of_errors: int


class ValidationResult(NamedTuple):
    correct_data: DataFrame
    erroneous_data: DataFrame
    errors: List[ValidationError]


class ValidateSparkDataFrame:
    """
    Describes the validation rules of a Spark DataFrame and performs the validation.

    // TODO update the example when there is a new validation rule
    Usage example:
        ValidateSparkDataFrame(spark_session, data_frame) \
            .execute()
    """
    def __init__(self, spark: SparkSession, data_frame: DataFrame):
        self.spark = spark
        self.df = data_frame

    def execute(self) -> ValidationResult:
        """
        Returns a named tuple containing the data that passed the validation, the data that was rejected, and a list of violated constraints.
        :return:
        """
        return ValidationResult(self.df, self.spark.createDataFrame([], self.df.schema), [])
