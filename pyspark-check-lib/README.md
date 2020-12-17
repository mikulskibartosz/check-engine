## Summary

The goal of this project is to implement a data validation library for PySpark. The library should detect the incorrect structure of the data, unexpected values in columns, and anomalies in the data.

## How to install

THERE IS NO PACKAGE YET!!!

## How to use

```
from pyspark_check.validate_df import ValidateSparkDataFrame

result = ValidateSparkDataFrame(spark_session, spark_data_frame) \
        .is_not_null("column_name") \
        .are_not_null(["column_name_2", "column_name_3"]) \
        .is_min("numeric_column", 10) \
        .is_max("numeric_column", 20) \
        .is_unique("column_name") \
        .are_unique(["column_name_2", "column_name_3"]) \
        .is_between("numeric_column_2", 10, 15) \
        .has_length_between("text_column", 0, 10) \
        .mean_column_value("numeric_column", 10, 20) \
        .text_matches_regex("text_column", "^[a-z]{3,10}$") \
        .one_of("text_column", ["value_a", "value_b"]) \
        .one_of("numeric_column", [123, 456]) \
        .execute()

result.correct_data #rows that passed the validation
result.erroneous_data #rows rejected during the validation
results.errors a summary of validation errors (three fields: column_name, constraint_name, number_of_errors)
```

## How to build

1. Install the Poetry build tool.

2. Run the following commands:

```
cd pyspark_check-lib
poetry build
```

## How to test locally

### Run all tests

```
cd pyspark_check-lib
poetry run pytest tests/
```

### Run a single test file

```
cd pyspark_check-lib
poetry run pytest tests/test_between_integer.py
```

### Run a single test method

```
cd pyspark_check-lib
poetry run pytest tests/test_between_integer.py -k 'test_should_return_df_without_changes_if_all_are_between'
```

## How to test in Docker

```
docker build -t pyspark-check-test pyspark-check-lib/. && docker run pyspark-check-test
```
