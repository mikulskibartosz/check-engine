import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark_session(request):
    """
    Fixture for creating a Spark Session used in the tests.
    :param request: pytest.FixtureRequest
    """
    spark_session = SparkSession.builder \
        .master("local[*]") \
        .appName("correct-horse-test") \
        .getOrCreate()

    request.addfinalizer(lambda: spark_session.sparkContext.stop())

    return spark_session
