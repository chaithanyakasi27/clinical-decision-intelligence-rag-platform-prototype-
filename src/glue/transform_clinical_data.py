import sys
import boto3
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from pyspark.sql.functions import col, current_timestamp, avg, count
from pyspark.sql.functions import max as spark_max

args        = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc          = SparkContext()
glueContext = GlueContext(sc)
spark       = glueContext.spark_session
job         = Job(glueContext)
job.init(args['JOB_NAME'], args)

RAW       = "cdip-dev-raw-clinical-docs-971422671083"
PROCESSED = "cdip-dev-processed-docs-971422671083"

print(f"RAW bucket: {RAW}")
print(f"PROCESSED bucket: {PROCESSED}")

# Step 1 - HCC mappings
try:
    print("Reading HCC mappings...")
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(f"s3://{RAW}/reference/2024_cms_hcc_mappings.csv")
    print(f"HCC rows: {df.count()}")
    df.show(3)
    df.withColumn("processed_at", current_timestamp()) \
      .write.mode("overwrite").parquet(
          f"s3://{PROCESSED}/transformed/hcc_mappings/"
      )
    print("HCC mappings written")
except Exception as e:
    print(f"ERROR HCC mappings: {e}")

# Step 2 - HCC coefficients
try:
    print("Reading HCC coefficients...")
    df2 = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(f"s3://{RAW}/reference/hcc_coefficients.csv")
    print(f"Coefficient rows: {df2.count()}")
    df2.write.mode("overwrite").parquet(
        f"s3://{PROCESSED}/transformed/hcc_coefficients/"
    )
    print("Coefficients written")
except Exception as e:
    print(f"ERROR coefficients: {e}")

# Step 3 - FHIR bundles using multiLine mode
try:
    print("Reading FHIR JSON files with multiLine...")
    df3 = spark.read \
        .option("multiLine", "true") \
        .option("mode", "PERMISSIVE") \
        .json(f"s3://{RAW}/fhir_bundles/")
    print(f"FHIR rows: {df3.count()}")
    print(f"FHIR columns: {df3.columns}")
    df3.show(2)

    # Select only columns that exist
    available = df3.columns
    select_cols = [c for c in ["id", "resourceType", "type"] if c in available]
    print(f"Selecting columns: {select_cols}")

    df3.select(*select_cols) \
       .withColumn("processed_at", current_timestamp()) \
       .write.mode("overwrite").parquet(
           f"s3://{PROCESSED}/transformed/fhir_bundles/"
       )
    print("FHIR bundles written")
except Exception as e:
    print(f"ERROR FHIR: {e}")
    import traceback
    traceback.print_exc()

# Step 4 - HCC summary using correct column names
try:
    print("Creating HCC summary...")
    df_hcc = spark.read.parquet(
        f"s3://{PROCESSED}/transformed/hcc_mappings/"
    )
    print(f"HCC columns: {df_hcc.columns}")

    # Use actual column names from the log:
    # icd10_code, hcc_category, hcc_label
    df_summary = df_hcc.groupBy("hcc_category", "hcc_label").agg(
        count("icd10_code").alias("icd_count"),
        avg("hcc_category").alias("avg_hcc"),
    )
    df_summary.write.mode("overwrite").parquet(
        f"s3://{PROCESSED}/transformed/hcc_summary/"
    )
    print(f"Summary written: {df_summary.count()} HCC categories")
    df_summary.show(10)
except Exception as e:
    print(f"ERROR summary: {e}")
    import traceback
    traceback.print_exc()

print("=== Job Complete ===")
job.commit()