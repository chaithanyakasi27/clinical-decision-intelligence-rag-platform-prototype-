# ============================================================
# AWS Glue PySpark Transformation Job
# Reads raw clinical data from S3
# Cleans, normalizes and writes to S3 Processed bucket
# ============================================================

import sys
import json
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lower, trim, regexp_replace,
    udf, lit, current_timestamp, length
)
from pyspark.sql.types import StringType

# ── Initialize Glue context ───────────────────────────────────
args        = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc          = SparkContext()
glueContext = GlueContext(sc)
spark       = glueContext.spark_session
job         = Job(glueContext)
job.init(args['JOB_NAME'], args)

# ── Configuration ─────────────────────────────────────────────
RAW_BUCKET       = "cdip-dev-raw-clinical-docs-971422671083"
PROCESSED_BUCKET = "cdip-dev-processed-docs-971422671083"

print("=" * 60)
print("  Clinical Data Transformation Job")
print(f"  Raw bucket:       {RAW_BUCKET}")
print(f"  Processed bucket: {PROCESSED_BUCKET}")
print("=" * 60)

def clean_clinical_text(text):
    """
    Clean and normalize clinical text.
    Removes special characters and normalizes whitespace.
    """
    if not text:
        return text
    # Normalize whitespace
    text = " ".join(text.split())
    # Remove non-ASCII characters
    text = text.encode("ascii", "ignore").decode()
    return text.strip()


# Register UDF
clean_text_udf = udf(clean_clinical_text, StringType())


def process_fhir_bundles():
    """
    Read FHIR JSON bundles from S3 raw bucket.
    Extract patient and condition data.
    Write cleaned data to S3 processed bucket.
    """
    print("\n[1/3] Processing FHIR bundles...")

    try:
        # Read all FHIR JSON files
        df_raw = spark.read.json(
            f"s3://{RAW_BUCKET}/fhir_bundles/"
        )

        print(f"  Raw FHIR records: {df_raw.count()}")

        # Select and clean key fields
        df_clean = df_raw.select(
            col("id").alias("bundle_id"),
            col("resourceType"),
            col("type").alias("bundle_type"),
        ).withColumn(
            "processed_at", current_timestamp()
        ).filter(
            col("bundle_id").isNotNull()
        )

        # Write to S3 processed bucket
        df_clean.write \
            .mode("overwrite") \
            .parquet(f"s3://{PROCESSED_BUCKET}/transformed/fhir_bundles/")

        print(f"  ✓ Processed {df_clean.count()} FHIR bundles")

    except Exception as e:
        print(f"  Warning: FHIR processing error: {e}")


def process_reference_data():
    """
    Read CMS reference data (ICD-10, HCC mappings).
    Normalize codes and write to processed bucket.
    """
    print("\n[2/3] Processing reference data...")

    try:
        # Read HCC mappings CSV
        df_hcc = spark.read.csv(
            f"s3://{RAW_BUCKET}/reference/2024_cms_hcc_mappings.csv",
            header=True,
            inferSchema=True,
        )

        print(f"  Raw HCC mappings: {df_hcc.count()}")

        # Clean and normalize
        df_clean = df_hcc.withColumn(
            "ICD_CODE", trim(upper(col("ICD_CODE")))
        ).withColumn(
            "HCC_LABEL", clean_text_udf(col("HCC_LABEL"))
        ).filter(
            col("ICD_CODE").isNotNull() &
            col("RAF_SCORE").isNotNull()
        )

        # Write to processed bucket
        df_clean.write \
            .mode("overwrite") \
            .parquet(f"s3://{PROCESSED_BUCKET}/transformed/hcc_mappings/")

        print(f"  ✓ Processed {df_clean.count()} HCC mappings")

        # Read HCC coefficients
        df_coef = spark.read.csv(
            f"s3://{RAW_BUCKET}/reference/hcc_coefficients.csv",
            header=True,
            inferSchema=True,
        )

        df_coef.write \
            .mode("overwrite") \
            .parquet(f"s3://{PROCESSED_BUCKET}/transformed/hcc_coefficients/")

        print(f"  ✓ Processed {df_coef.count()} HCC coefficients")

    except Exception as e:
        print(f"  Warning: Reference data error: {e}")


def create_clinical_summary():
    """
    Create a summary dataset combining all processed data.
    This is what the RAG pipeline reads from.
    """
    print("\n[3/3] Creating clinical summary...")

    try:
        # Read processed HCC mappings
        df_hcc = spark.read.parquet(
            f"s3://{PROCESSED_BUCKET}/transformed/hcc_mappings/"
        )

        # Create summary statistics
        from pyspark.sql.functions import count, avg, max as spark_max

        df_summary = df_hcc.groupBy("HCC_NUM", "HCC_LABEL") \
            .agg(
                count("ICD_CODE").alias("icd_code_count"),
                avg("RAF_SCORE").alias("avg_raf_score"),
                spark_max("RAF_SCORE").alias("max_raf_score"),
            ) \
            .orderBy("HCC_NUM")

        # Write summary
        df_summary.write \
            .mode("overwrite") \
            .parquet(f"s3://{PROCESSED_BUCKET}/transformed/hcc_summary/")

        print(f"  ✓ Created summary for {df_summary.count()} HCC categories")

        # Show top HCC categories by RAF score
        print("\n  Top HCC categories by RAF score:")
        df_summary.orderBy(
            col("avg_raf_score").desc()
        ).show(5, truncate=False)

    except Exception as e:
        print(f"  Warning: Summary creation error: {e}")

# ── Run all transformations ───────────────────────────────────
process_fhir_bundles()
process_reference_data()
create_clinical_summary()

print("\n" + "=" * 60)
print("  Transformation complete!")
print(f"  Results in: s3://{PROCESSED_BUCKET}/transformed/")
print("=" * 60)

job.commit()