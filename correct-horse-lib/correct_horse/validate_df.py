import random
import string
from dataclasses import dataclass
from typing import NamedTuple, List, Callable, Optional, Tuple

from pyspark.sql import DataFrame, SparkSession


class ValidationError(NamedTuple):
    column_name: str
    constraint_name: str
    number_of_errors: int


class ValidationResult(NamedTuple):
    correct_data: DataFrame
    erroneous_data: DataFrame
    errors: List[ValidationError]


@dataclass
class _Constraint:
    constraint_name: str
    column_name: str
    constraint_column_name: str
    filter_success: Callable[[DataFrame], DataFrame]
    filter_failure: Callable[[DataFrame], DataFrame]
    validate_constraint: Callable[[DataFrame, List[str]], Tuple[bool, str]]
    prepare_constraint_check: Callable[[DataFrame, str], DataFrame]


class ValidateSparkDataFrame:
    """
    Describes the validation rules of a Spark DataFrame and performs the validation.

    // TODO update the example when there is a new validation rule
    Usage example:
        ValidateSparkDataFrame(spark_session, data_frame) \
            .is_unique("column_name") \
            .are_unique(["column_name_2", "column_name_3"]) \
            .execute()
    """

    def __init__(self, spark: SparkSession, data_frame: DataFrame):
        self.spark: SparkSession = spark
        self.df: DataFrame = data_frame
        self.input_columns: List[str] = data_frame.columns
        self.constraints: List[_Constraint] = []

    def is_unique(self, column_name: str):
        """
        Defines a constraint that checks whether the given column contains only unique values.
        :param column_name: the name of the column
        :raises ValueError: if an unique constraint for a given column already exists.
        :return: self
        """
        existing = filter(lambda c: c.constraint_name == 'unique' and c.column_name == column_name, self.constraints)
        if list(existing):
            raise ValueError(f"An unique constraint for column {column_name} already exists.")

        def check_uniqueness(data_frame: DataFrame, constraint_column: str) -> DataFrame:
            count_repetitions: DataFrame = data_frame \
                .groupby(column_name) \
                .count() \
                .withColumnRenamed("count", constraint_column)

            return data_frame.join(count_repetitions, column_name, "left")

        constraint_column_name = self._generate_constraint_column_name("unique", column_name)
        self.constraints.append(_Constraint(
            "unique",
            column_name,
            constraint_column_name,
            lambda df: df.filter(f"{constraint_column_name} == 1"),
            lambda df: df.filter(f"{constraint_column_name} > 1"),
            lambda df, columns: (column_name in columns, f"There is no '{column_name}' column"),
            check_uniqueness
        ))

        return self

    def are_unique(self, column_names: List[str]):
        """
        Defines constraints that check whether given columns contain only unique values.
        :param column_names: a list of column names
        :return: self
        """
        for column_name in column_names:
            self.is_unique(column_name)
        return self

    def execute(self) -> ValidationResult:
        """
        Returns a named tuple containing the data that passed the validation, the data that was rejected (only unique rows), and a list of violated constraints.

        :raises ValueError: if a constraint has been defined using a non-existing column.
        :return:
        """
        self._validate_constraints()

        if self.constraints:
            for constraint in self.constraints:
                self.df = constraint.prepare_constraint_check(self.df, constraint.constraint_column_name)

            correct_output = self.df
            errors = []

            for constraint in self.constraints:
                correct_output = constraint.filter_success(correct_output)
                number_of_failures = constraint.filter_failure(self.df).count()

                if number_of_failures > 0:
                    errors.append(ValidationError(constraint.column_name, constraint.constraint_name, number_of_failures))

            correct_output = correct_output.select(self.input_columns)
            incorrect_output = self.df.select(self.input_columns).subtract(correct_output)

            return ValidationResult(correct_output, incorrect_output, errors)
        else:
            return ValidationResult(self.df, self.spark.createDataFrame([], self.df.schema), [])

    def _validate_constraints(self) -> None:
        columns = self.df.columns

        errors = []
        for constraint in self.constraints:
            is_correct, error_message = constraint.validate_constraint(self.df, columns)
            if not is_correct:
                errors.append(error_message)

        if errors:
            raise ValueError(", ".join(errors))

    def _generate_constraint_column_name(self, constraint_type, column_name):
        random_suffix = ''.join(random.choice(string.ascii_lowercase) for i in range(12))
        return f"__correct_horse__{column_name}_{constraint_type}_{random_suffix}"
