import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel

object AmazonReviewRDD {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("AmazonReviewRDD")
      .master("local[4]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val dataPath = "file:///home/hadoop/Downloads/archive/train.csv"

    val rawDF = spark.read
      .option("header", "false")
      .option("quote", "\"")
      .option("escape", "\"")
      .csv(dataPath)

    val reviewsRDD = rawDF.rdd
    val labelsRDD = reviewsRDD.map(row => row.getString(0))
    val wordsRDD = reviewsRDD.flatMap(
      row => row.getString(2).toLowerCase.split("\\W+")
    )

    println("RDD created successfully")
    println(s"Total reviews: ${reviewsRDD.count()}")

    println("First five sentiment labels:")
    labelsRDD.take(5).foreach(println)

    val negativeLabelsRDD =
      labelsRDD.filter(label => label == "1")

    println(s"Negative review count: ${negativeLabelsRDD.count()}")

    val reviewLabelRDD =
      reviewsRDD.map(row => (row.getString(0), row.getString(1)))

    println("First five label-title pairs:")
    reviewLabelRDD.take(5).foreach(println)

    val labelCountsRDD =
      labelsRDD.map(label => (label, 1)).reduceByKey(_ + _)

    println("Sentiment distribution:")
    labelCountsRDD.collect().foreach(println)

    println(s"Number of partitions: ${reviewsRDD.getNumPartitions}")

    println("Records per partition:")
    reviewsRDD.mapPartitions(iter => Iterator(iter.size))
      .collect().foreach(println)

    println("Word RDD lineage:")
    println(wordsRDD.toDebugString)

    println("Sentiment aggregation lineage:")
    println(labelCountsRDD.toDebugString)

    labelsRDD.cache()
    println(s"Cached label count: ${labelsRDD.count()}")
    println(s"Label RDD storage level: ${labelsRDD.getStorageLevel}")

    wordsRDD.persist(StorageLevel.MEMORY_AND_DISK)
    println(s"Total word count: ${wordsRDD.count()}")
    println(s"Word RDD storage level: ${wordsRDD.getStorageLevel}")

    spark.stop()
  }
}
