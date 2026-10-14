import org.apache.spark.storage.StorageLevel
val dataPath = "file:///home/hadoop/Downloads/archive/train.csv"

val rawDF = spark.read.option("header", "false").option("quote", "\"").option("escape", "\"").csv(dataPath)

val reviewsRDD = rawDF.rdd

println("RDD created successfully")
val labelsRDD = reviewsRDD.map(row => row.getString(0))

println("First five sentiment labels:")
labelsRDD.take(5).foreach(println)
val wordsRDD = reviewsRDD.flatMap(row => row.getString(2).toLowerCase.split("\\W+"))
val negativeLabelsRDD = labelsRDD.filter(label => label == "1")

val reviewLabelRDD = reviewsRDD.map(row => (row.getString(0), row.getString(1)))

val labelCountsRDD = labelsRDD.map(label => (label, 1)).reduceByKey(_ + _)

println("Negative review count:")
println(negativeLabelsRDD.count())

println("Sentiment distribution:")
labelCountsRDD.collect().foreach(println)
println("Number of partitions:")
println(reviewsRDD.getNumPartitions)

println("Records per partition:")
reviewsRDD.mapPartitions(iter => Iterator(iter.size))
  .collect().foreach(println)

println("Word RDD lineage:")
println(wordsRDD.toDebugString)

println("Sentiment aggregation lineage:")
println(labelCountsRDD.toDebugString)

println("Caching labelsRDD:")
labelsRDD.cache()
println("Label count after caching:")
println(labelsRDD.count())
println("Label RDD storage level:")
println(labelsRDD.getStorageLevel)

println("Persisting wordsRDD with MEMORY_AND_DISK:")
wordsRDD.persist(StorageLevel.MEMORY_AND_DISK)
println("Word count after persistence:")
println(wordsRDD.count())
println("Word RDD storage level:")
println(wordsRDD.getStorageLevel)
