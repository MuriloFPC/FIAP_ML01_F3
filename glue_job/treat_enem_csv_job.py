import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from awsglue.utils import getResolvedOptions

args = getResolvedOptions(
    sys.argv,
    [
        'JOB_NAME',
        'INPUT_PATH',
        'OUTPUT_PATH'
    ]
)

job_name = args['JOB_NAME']
input_path = args['INPUT_PATH']
output_path = args['OUTPUT_PATH']

spark = SparkSession.builder\
        .appName(job_name)\
        .getOrCreate()

df = spark.read.options(header='true',
                        inferSchema='true',
                        sep=';').csv(input_path)
print("import dataframe")
df.show(truncate=False, vertical=True, n=1)

df = df.na.drop()

columns_to_drop = [col for col in df.columns[:31]
                   if col not in ['TP_ESCOLA',
                                  'TP_FAIXA_ETARIA',
                                  'TP_ENSINO']]
df = df.drop(*columns_to_drop)

# Mapping to convert letters to numbers
map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9,
       'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17}


def map_values(value):
    return map.get(value, None)


map_udf = udf(map_values, IntegerType())

# Apply mapping
df = df.select([map_udf(col(c)).alias(c) if c in map
                else col(c) for c in df.columns])
print("apply mapping")
df.show(truncate=False, vertical=True, n=1)


df.coalesce(1).write.mode("overwrite").option("header", "true"
                                              ).csv(output_path)
