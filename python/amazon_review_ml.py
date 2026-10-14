
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, concat_ws, coalesce, lit, trim
)
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    HashingTF
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)

# 1. Initialize Spark
spark = (
    SparkSession.builder
    .appName("AmazonReviewML")
    .master("local[4]")
    .config("spark.driver.memory", "12g")
    .config("spark.sql.shuffle.partitions", "32")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

try:
    # 2. Load the dataset from the local filesystem
    train_path = "file:///home/hadoop/Downloads/archive/train.csv"

    reviews = (
        spark.read
        .option("header", "false")
        .option("quote", '"')
        .option("escape", '"')
        .csv(train_path)
        .toDF("label", "title", "content")
    )

    # 3. Clean the data and combine title + content
    reviews = reviews.filter(
        col("label").isin("1", "2")
        & col("content").isNotNull()
    )

    reviews = reviews.withColumn(
        "text",
        trim(
            concat_ws(
                " ",
                coalesce(col("title"), lit("")),
                col("content")
            )
        )
    ).filter(col("text") != "")

    reviews = reviews.withColumn(
        "label",
        (col("label").cast("double")-1.0)
    )

    # 4. Verify the dataset
    print("\nDataset schema:")
    reviews.printSchema()

    print("\nFirst five reviews:")
    reviews.select("label", "text").show(5, truncate=80)

    print("\nTotal valid reviews:", reviews.count())

    print("\nLabel distribution:")
    reviews.groupBy("label").count().orderBy("label").show()

    # 5. Split into training and test sets
    train_data, test_data = reviews.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    print("\nTraining reviews:", train_data.count())
    print("Testing reviews:", test_data.count())

    # 6. Convert text into numerical features
    tokenizer = Tokenizer(
        inputCol="text",
        outputCol="words"
    )

    stopwords = StopWordsRemover(
        inputCol="words",
        outputCol="filtered_words"
    )

    hashing_tf = HashingTF(
        inputCol="filtered_words",
        outputCol="features",
        numFeatures=1 << 16
    )

    # 7. Configure Logistic Regression
    logistic_regression = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=10,
        regParam=0.01
    )

    # 8. Build and train the complete ML pipeline
    ml_pipeline = Pipeline(
        stages=[
            tokenizer,
            stopwords,
            hashing_tf,
            logistic_regression
        ]
    )

    print("\nTraining Logistic Regression...")
    model = ml_pipeline.fit(train_data)

    # 9. Generate predictions on unseen test data
    print("\nGenerating test predictions...")
    predictions = model.transform(test_data)

    print("\nSample predictions:")
    predictions.select(
        "label", "prediction", "probability"
    ).show(10, truncate=False)

    # 10. Evaluate the model
    metrics = {}

    for metric in [
        "accuracy",
        "weightedPrecision",
        "weightedRecall",
        "f1"
    ]:
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName=metric
        )

        score = evaluator.evaluate(predictions)
        metrics[metric] = score
        print(f"{metric}: {score:.4f}")

    # 11. Calculate ROC-AUC
    binary_evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    auc = binary_evaluator.evaluate(predictions)
    metrics["roc_auc"] = auc
    print(f"Area under ROC-AUC: {auc:.4f}")
    # 11. Export predictions for charts
    from pyspark.sql.functions import udf
    from pyspark.sql.types import DoubleType

    probability_positive = udf(
        lambda v: float(v[1]),
        DoubleType()
    )

    chart_predictions = predictions.select(
        "label",
        "prediction",
        probability_positive("probability").alias("positive_probability")
    )

    chart_predictions.write \
        .mode("overwrite") \
        .option("header", "true") \
        .csv("file:///home/hadoop/AmazonSparkProject/results/predictions_csv")
    print("Predictions exported for chart generation.")
    # 12. Save actual evaluation metrics
    import os

    results_dir = "/home/hadoop/AmazonSparkProject/results"
    os.makedirs(results_dir, exist_ok=True)

    metrics_path = os.path.join(results_dir, "ml_metrics.txt")

    with open(metrics_path, "w") as f:
        f.write("Amazon Review Sentiment Classification\n")
        f.write("======================================\n")
        f.write("Dataset: Amazon Polarity\n")
        f.write("Total reviews: 3600000\n")
        f.write("Training reviews: 2880586\n")
        f.write("Testing reviews: 719414\n")
        f.write("Model: Logistic Regression\n")
        f.write("Features: HashingTF (65536 dimensions)\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Weighted Precision: {metrics['weightedPrecision']:.4f}\n")
        f.write(f"Weighted Recall: {metrics['weightedRecall']:.4f}\n")
        f.write(f"F1-score: {metrics['f1']:.4f}\n")
        f.write(f"ROC-AUC: {metrics['roc_auc']:.4f}\n")

    print(f"Metrics saved to: {metrics_path}")

finally:
    spark.stop()
