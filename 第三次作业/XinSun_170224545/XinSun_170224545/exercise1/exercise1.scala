import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.feature.{StandardScaler, StandardScalerModel}
import org.apache.spark.mllib.linalg.SingularValueDecomposition


object Exercise1 {
  def main(args: Array[String]): Unit = {

    val sparkSession = SparkSession.builder().master("local[16]").appName("Exercise 1").config("spark.local.dir","/data/act17xs/tmp").getOrCreate()
    import sparkSession.implicits._
    // read the data
    val data = sparkSession.sparkContext.textFile("files/NIPS_1987-2015.csv.csv")

    // drop the first row and the first column
    val header = data.first
    val only_rows = data.filter(x => x != header)
    // transpose the data
    val rdd = only_rows.map(x => x.split(",")).map(x => x.drop(1))
    val RDD = rdd.collect.transpose.map(x => Vectors.dense(x.map(_.toDouble)))
    val datardd = sparkSession.sparkContext.parallelize(RDD, 200)

    val dataRDD = new StandardScaler(withMean = true, withStd = true).fit(datardd).transform(datardd)
    val mat = new RowMatrix(dataRDD)

    // Compute the top 2 principal components.
    // Principal components are stored in a local dense matrix.

    val svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(2, computeU = true)
    val pc: Matrix = mat.computePrincipalComponents(2)
    val pc_array = pc.toArray

    val pc1 = pc_array.take(11463)
    val pc2 = pc_array.drop(11463)

    // calculate the eigenvalues
    val s: Vector = svd.s
    val pc1_s = s(0) * s(0)
    val pc2_s = s(1) * s(1)
    println("PC1 eigenvalues:")
    println(pc1_s)
    println("PC2 eigenvalues:")
    println(pc2_s)

    // calculate the variance
    val pc_variance = new PCA(2).fit(dataRDD)
    val variance = pc_variance.explainedVariance
    println("PC1 variance:")
    println(variance(0))
    println("PC2 variance:")
    println(variance(1))

    // show the top 10
    val PC1 = new DenseMatrix(11463, 1, pc1)
    val PC2 = new DenseMatrix(11463, 1, pc2)
    val pc1_10 = pc1.take(10)
    val pc2_10 = pc2.take(10)
    val PC1_10 = new DenseMatrix(10, 1, pc1_10)
    val PC2_10 = new DenseMatrix(10, 1, pc2_10)
    println("PC1 first 10 entries:")
    println(PC1_10)
    println("PC2 first 10 entries:")
    println(PC2_10)

    // Project the rows to the linear space spanned by the top 2 principal components.
    val projected: RowMatrix = mat.multiply(pc)
    val rdd_projected = projected.rows.map( x => x.toArray.mkString(","))
    // read the output
    rdd_projected.coalesce(1).saveAsTextFile("files/exercise1_output")


    sparkSession.stop()
  }
}


