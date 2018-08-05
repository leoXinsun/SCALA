import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}


object Exercise2 {
  def main(args: Array[String]): Unit = {

    val sparkSession = SparkSession.builder().master("local[16]").appName("exercise2").getOrCreate()

    // read the data
    val data = sparkSession.sparkContext.textFile("files/data.csv",100)

    // data processing
    val header = data.first
    val only_rows = data.filter(x => x != header)
    val labels = only_rows.map(s => s.split(",").takeRight(1).map(_.toInt)).map(s => s(0))
    val RDD = only_rows.map(x => x.split(",").drop(1).map(_.toDouble))
    val RDD1 = RDD.map(s => Vectors.dense(s.take(178).map(_.toDouble))).cache()

    // Cluster the data into two classes using KMeans
    val numClusters = 5
    val numIterations = 100
    val model = KMeans.train(RDD1, numClusters, numIterations)

    // print the clusters
    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)

    // calculate the size
    val size = model.predict(RDD1).map(s=>(s,1)).reduceByKey((a,b)=>a+b)
    val size_array = size.take(5)

    // sort the size
    val sortedBysize = size_array.sortBy{case(cluster,size) => size}
    val max = sortedBysize(4)
    val max_id = max._1
    val max_size = max._2
    val min = sortedBysize(0)
    val min_id = min._1
    val min_size = min._2

    println("The largest cluster is:")
    println(model.clusterCenters(max_id))
    println("the size of the largest cluster is " + max_size)
    println("The smallest cluster is:")
    println(model.clusterCenters(min_id))
    println("the size of the smallest cluster is " + min_size)

    // calculate the distance
    def euclidean(x: Vector, y: Vector) = {
      math.sqrt(x.toArray.zip(y.toArray).
        map(p => p._1 - p._2).map(d => d*d).sum)
    }
    println("the distance between the largest cluster and the smallest cluster is " + euclidean(model.clusterCenters(max_id),model.clusterCenters(min_id)))

    val prediction = model.predict(RDD1)
    val pre_label = prediction.zip(labels)
    val pre_largest = pre_label.filter({case (pred, label) => pred == max_id}).countByValue()
    val sort_largest = pre_largest.map{case((pred,label),count) => List(pred,label,count)}.toArray.sortBy(s =>s(2))
    val major_largest=sort_largest(sort_largest.size - 1)
    val label_largest = major_largest(1)
    val label_largest_count = major_largest(2)
    println("the majority label for the largest cluster is " + label_largest + ", and the number is " + label_largest_count)


    val pre_smallest = pre_label.filter({case (pred, label) => pred == min_id}).countByValue()
    val sort_smallest = pre_smallest.map{case((pred,label),count) => List(pred,label,count)}.toArray.sortBy(s =>s(2))
    val major_smallest= sort_smallest(sort_smallest.size - 1)
    val label_smallest = major_smallest(1)
    val label_smallest_count = major_smallest(2)
    println("the majority label for the smallest cluster is " + label_smallest + ", and the number is " + label_smallest_count)


    sparkSession.stop()
  }
}
