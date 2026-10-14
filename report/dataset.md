# Dataset Description

## Dataset Name
Amazon Review Polarity

## Application
Binary sentiment classification of customer reviews.

## Data Source
Amazon Review Polarity dataset, distributed as CSV files.

## Files
- Training data: /home/hadoop/Downloads/archive/train.csv
- Test data: /home/hadoop/Downloads/archive/test.csv

## Attributes

| Attribute | Description |
|---|---|
| Label | Sentiment class: 1 = negative, 2 = positive |
| Title | Title of the customer review |
| Content | Full text of the customer review |

## Problem Statement
Customer reviews contain useful information about product
satisfaction. This project processes Amazon reviews using
Apache Spark and builds a machine learning model to classify
reviews as positive or negative.

## Dataset Format
The CSV files contain three quoted fields per record:
label, title, and content. The files do not have a header row.
## Dataset Validation Results

- Training records processed by Spark: 3,600,000
- RDD creation: Successful
- Validation action: `reviewsRDD.count()`
- Result: 3,600,000 records
## RDD Transformation Results

### 1. map()

Extracted the sentiment label from each review using:

`reviewsRDD.map(row => row.getString(0))`

The first five labels returned were all `2` (positive).

### 2. filter()

Selected negative reviews using:

`labelsRDD.filter(label => label == "1")`

**Actual result:** 1,800,000 negative reviews.

**Action used:** `negativeLabelsRDD.count()`
### 3. flatMap()

**Purpose:** Splits each review's text into individual words.

**Code:**
```scala
val wordsRDD = reviewsRDD.flatMap(
  row => row.getString(2).toLowerCase.split("\\W+")
)
```

**Observed output — first 10 words:**
`this`, `sound`, `track`, `was`, `beautiful`, `it`, `paints`, `the`, `senery`, `in`

**Explanation:** Unlike `map()`, which produces one output element per input row, `flatMap()` can produce multiple output elements from each review. This makes it useful for word-level text analysis.
## RDD Action Results

### Word Count Using `count()`

**Code:**
```scala
wordsRDD.count()
```

**Output:** `275,519,555`

**Interpretation:** The `wordsRDD` contains 275,519,555 word elements generated from the 3.6 million training reviews. The `count()` action evaluates the RDD computation and returns the total number of elements.

**Actions completed so far:**
1. `reviewsRDD.count()` — counted the review records (3,600,000).
2. `negativeLabelsRDD.count()` — counted negative reviews (1,800,000).
3. `wordsRDD.count()` — counted word elements (275,519,555).
## Key-Value RDD Operations

### Sentiment Distribution Using `reduceByKey()`

**Code:**
```scala
val labelCountsRDD = labelsRDD
  .map(label => (label, 1))
  .reduceByKey(_ + _)

labelCountsRDD.collect().foreach(println)
```

**Observed output:**
```text
(1,1800000)
(2,1800000)
```

**Interpretation:** Each sentiment label is converted into a key-value pair, where the value is initially `1`. The `reduceByKey()` operation adds the values associated with identical labels to calculate the total number of reviews per sentiment class.

**Finding:** The training dataset contains 1.8 million negative reviews and 1.8 million positive reviews. Therefore, the two sentiment classes are equally represented.
## RDD Partition Analysis

### Number of Partitions

**Code:**
```scala
reviewsRDD.getNumPartitions
```

**Output:** `12`

### Records per Partition

**Code:**
```scala
reviewsRDD.mapPartitions(iter => Iterator(iter.size)).collect().foreach(println)
```

**Observed record counts:**

| Partition | Records |
|---:|---:|
| 1 | 298,935 |
| 2 | 302,167 |
| 3 | 302,477 |
| 4 | 302,238 |
| 5 | 302,745 |
| 6 | 304,769 |
| 7 | 305,138 |
| 8 | 307,818 |
| 9 | 307,444 |
| 10 | 308,217 |
| 11 | 308,548 |
| 12 | 249,504 |

### Verification

**Code:**
```scala
reviewsRDD.mapPartitions(iter => Iterator(iter.size)).sum()
```

**Output:** `3600000.0`

**Interpretation:** Spark divided the training data into 12 partitions. The total number of records across these partitions equals the original 3.6 million reviews. With `local[4]`, Spark can execute tasks using up to four local threads concurrently; 12 partitions do not mean 12 threads run simultaneously.
## RDD Lineage and Fault Tolerance

### Inspecting RDD Lineage

**Code:**
```scala
wordsRDD.toDebugString
```

**Observation:** The output contains `FileScanRDD`, intermediate `MapPartitionsRDD` entries, and the final `MapPartitionsRDD[17]` created by `flatMap()`.

**Interpretation:** Spark maintains dependency information connecting the word RDD to the CSV data source and the operations used to construct it. This dependency chain is called RDD lineage.

**Fault tolerance:** If a partition is lost, Spark can recompute it using lineage rather than storing a separate backup of every intermediate RDD. Recovery depends on the availability of the source data and the relevant dependencies.
## DAG and Shuffle Analysis

**Code:**
```scala
labelCountsRDD.toDebugString
```

**Observation:** The output contains a `ShuffledRDD` created by `reduceByKey()`, along with upstream `MapPartitionsRDD` and `FileScanRDD` dependencies.

**Interpretation:** The sentiment-counting operation converts each label into a key-value pair and combines values associated with the same key. The shuffle redistributes intermediate data as required by the key-based aggregation.

**DAG significance:** Spark uses dependency information to organize computation into stages and tasks. Shuffle dependencies generally create stage boundaries, whereas narrow transformations such as `map()` can often be executed within the same stage.

**Fault tolerance:** The dependency structure also helps Spark determine how lost partitions can be recomputed.
## RDD Caching

**Code:**
```scala
labelsRDD.cache()
labelsRDD.count()
labelsRDD.getStorageLevel
```

**Observed results:**
- Label count: `3,600,000`
- Storage level: `StorageLevel(memory, deserialized, 1 replicas)`

**Interpretation:** The `cache()` method marks the RDD for caching, and the `count()` action triggers its computation. Spark reports a memory-based storage level, allowing computed partitions to be reused by later operations when they remain cached.

**Benefit:** Caching is useful when the same RDD is accessed repeatedly because it can reduce the need to recompute its transformations.
## RDD Persistence Experiment

### Using `persist()` with `MEMORY_AND_DISK`

**Code:**
```scala
import org.apache.spark.storage.StorageLevel

wordsRDD.persist(StorageLevel.MEMORY_AND_DISK)
wordsRDD.count()
wordsRDD.getStorageLevel
```

**Observed word count:** `275,519,555`

**Observed storage level:** `StorageLevel(disk, memory, deserialized, 1 replicas)`

**Observation:** During the count operation, Spark reported that some RDD partitions could not fit in memory and were persisted to disk instead. Despite these warnings, the operation completed successfully.

**Interpretation:** The experiment demonstrates that `MEMORY_AND_DISK` allows Spark to use both memory and disk for persisted RDD partitions. This is useful for large datasets that exceed the available Spark storage memory.

**Comparison:** `cache()` uses the default RDD storage level, while `persist()` allows a storage level to be selected explicitly.
## Spark Web UI: Storage Verification

The Spark Storage tab displayed RDD ID 17, corresponding to the persisted word RDD.

| Metric | Observed value |
|---|---|
| Storage level | Disk Memory |
| Cached partitions | 12 |
| Fraction cached | 100% |
| Size in memory | 0.0 B |
| Size on disk | 1642.4 MiB |

**Interpretation:** All 12 partitions of the word RDD were retained on disk because the RDD was too large to fit in the available storage memory. The result supports the earlier Spark Shell warnings and demonstrates the practical use of `MEMORY_AND_DISK` persistence.

**Evidence:** Screenshot of the Spark Web UI Storage tab.
## Reproducible Application Execution

The Scala application was compiled into `AmazonReviewRDD.jar` and executed successfully using `spark-submit` with Spark 3.5.1 in `local[4]` mode.

The captured execution output confirmed:
- 3,600,000 training reviews processed.
- 1,800,000 negative and 1,800,000 positive reviews.
- 12 input partitions.
- 275,519,555 word elements.
- RDD lineage output for word processing and sentiment aggregation.
- Successful persistence using `MEMORY_AND_DISK`, with some partitions written to disk due to memory constraints.

**Execution evidence:** `results/scala_run_20261013_180347.log`

The log contains the actual console output from the successful application run.
## PySpark Data Validation Results

The Amazon Polarity training dataset was successfully loaded using PySpark 3.5.1. The dataset contains 3,600,000 reviews with three original columns: label, title, and content.

| Validation check | Result |
|---|---:|
| Total reviews | 3,600,000 |
| Negative reviews (label 1) | 1,800,000 |
| Positive reviews (label 2) | 1,800,000 |
| Missing labels | 0 |
| Missing titles | 48 |
| Missing review content | 0 |

The dataset has an equal number of positive and negative reviews, giving it a balanced class distribution. Forty-eight reviews have missing titles, but their review content is present. The preprocessing stage combines the title and content into a single text feature, treating a missing title as an empty string.

The label values are retained as strings during initial loading. They will be converted to the appropriate format during machine-learning preprocessing.
