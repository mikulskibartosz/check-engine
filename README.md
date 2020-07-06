# PySpark-Check

![Tests](https://github.com/mikulskibartosz/correct-horse/workflows/Tests/badge.svg)
![GitHub](https://img.shields.io/github/license/mikulskibartosz/correct-horse)

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
