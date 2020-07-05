from typing import NamedTuple, List

from pyspark.sql import DataFrame, SparkSession

from pyspark_check._constraints._Constraint import _Constraint
from pyspark_check._constraints._NotNull import _NotNull
from pyspark_check._constraints._Unique import _Unique


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
            .is_not_null("column_name") \
            .are_not_null(["column_name_2", "column_name_3"]) \
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
        self._add_constraint(_Unique(column_name))
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

    def is_not_null(self, column_name: str):
        """
        Defines a constraint that does not allow null values in a given column.
        :param column_name: the column name
        :return: self
        """
        self._add_constraint(_NotNull(column_name))
        return self

    def are_not_null(self, column_names: List[str]):
        """
        Defines constraints that don't allow null values in all of the given columns
        :param column_names: a list of column names
        :return: self
        """
        for column_name in column_names:
            self.is_not_null(column_name)
        return self

    def execute(self) -> ValidationResult:
        """
        Returns a named tuple containing the data that passed the validation, the data that was rejected (only unique rows), and a list of violated constraints.
        Note that the order of rows and constraints is not preserved.

        :raises ValueError: if a constraint has been defined using a non-existing column.
        :return:
        """
        self._validate_constraints()

        if self.constraints:
            for constraint in self.constraints:
                self.df = constraint.prepare_df_for_check(self.df)

            correct_output = self.df
            errors = []

            for constraint in self.constraints:
                correct_output = constraint.filter_success(correct_output)
                number_of_failures = constraint.filter_failure(self.df).count()

                if number_of_failures > 0:
                    errors.append(ValidationError(constraint.column_name, constraint.constraint_name(), number_of_failures))

            correct_output = correct_output.select(self.input_columns)
            incorrect_output = self.df.select(self.input_columns).subtract(correct_output)

            return ValidationResult(correct_output, incorrect_output, errors)
        else:
            return ValidationResult(self.df, self.spark.createDataFrame([], self.df.schema), [])

    def _add_constraint(self, constraint: _Constraint) -> None:
        existing = filter(lambda c: c.constraint_name() == constraint.constraint_name() and c.column_name == constraint.column_name, self.constraints)
        if list(existing):
            raise ValueError(f"An not_null constraint for column {constraint.column_name} already exists.")

        self.constraints.append(constraint)

    def _validate_constraints(self) -> None:
        columns = self.df.columns

        errors = []
        for constraint in self.constraints:
            is_correct, error_message = constraint.validate_self(self.df, columns)
            if not is_correct:
                errors.append(error_message)

        if errors:
            raise ValueError(", ".join(errors))
