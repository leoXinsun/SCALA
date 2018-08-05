import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.regression.GeneralizedLinearRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, RegressionEvaluator}

object Exercise2 {
  def main(args: Array[String]): Unit = {

    val sparkSession = SparkSession.builder.master("local[4]").appName("exercise 2").getOrCreate()
    
    val text = sparkSession.read.option("header", "true").csv("files/train_set.csv").na.drop().repartition(40)

    val indexers = text.columns.map(feature => new StringIndexer().setInputCol(feature).setOutputCol(s"${feature}Index"))
    val assembler  = new VectorAssembler().setInputCols(text.columns.map(feature => s"${feature}Index")).setOutputCol("features")
    val indexer =  new StringIndexer().setInputCol("Claim_Amount").setOutputCol("label")

    val pipeline_0 = new Pipeline().setStages(indexers)
    val model_0 = pipeline_0.fit(text)
    val data_0 = model_0.transform(text)

    val pipeline = new Pipeline().setStages(Array(assembler,indexer))
    val model = pipeline.fit(data_0)
    val data = model.transform(data_0)

    // Split the data into training and test sets (30% held out for testing)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // create two different Generalised Linear Models : Gaussian and Poisson

    // Gaussian Generalised Linear Model

    val time_gau_s = System.nanoTime
    val glr_gau = new GeneralizedLinearRegression().setFamily("gaussian").setLink("identity").setMaxIter(10).setRegParam(0.3)

    // Fit the model
    val model_gau = glr_gau.fit(trainingData)

    // Print the coefficients and intercept for generalized linear regression model
    println("Gaussian: " + s"Coefficients: ${model_gau.coefficients}")
    println("Gaussian: " + s"Intercept: ${model_gau.intercept}")

    // Summarize the model over the training set and print out some metrics
    println("some metrics of Gaussian Generalised Linear Models : ")
    val summary_gau = model_gau.summary
    println("Gaussian: " + s"Coefficient Standard Errors: ${summary_gau.coefficientStandardErrors.mkString(",")}")
    println("Gaussian: " + s"T Values: ${summary_gau.tValues.mkString(",")}")
    println("Gaussian: " + s"P Values: ${summary_gau.pValues.mkString(",")}")
    println("Gaussian: " + s"Dispersion: ${summary_gau.dispersion}")
    println("Gaussian: " + s"Null Deviance: ${summary_gau.nullDeviance}")
    println("Gaussian: " + s"Residual Degree Of Freedom Null: ${summary_gau.residualDegreeOfFreedomNull}")
    println("Gaussian: " + s"Deviance: ${summary_gau.deviance}")
    println("Gaussian: " + s"Residual Degree Of Freedom: ${summary_gau.residualDegreeOfFreedom}")
    println("Gaussian: " + s"AIC: ${summary_gau.aic}")
    println("Gaussian: " + "Deviance Residuals: ")
    summary_gau.residuals().show()

    // make the prediction
    var predictions_gau = model_gau.transform(testData)

    // calculate the RMSE
    val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
    var rmse_gau = evaluator.evaluate(predictions_gau)
    println("RMSE of Gaussion generalised linear = " + rmse_gau)
    val duration_gau = (System.nanoTime - time_gau_s) / 1e9d
    println("The training time for Gaussian Generalised Linear Model is " + duration_gau + "s")


    // Poisson Generalised Linear Model

    val time_poi_s = System.nanoTime
    val glr_poi = new GeneralizedLinearRegression().setFamily("poisson").setLink("identity").setMaxIter(10).setRegParam(0.3)

    // Fit the model
    val model_poi = glr_poi.fit(trainingData)

    // Print the coefficients and intercept for generalized linear regression model
    println("Poisson: " + s"Coefficients: ${model_poi.coefficients}")
    println("Poisson: " + s"Intercept: ${model_poi.intercept}")

    // Summarize the model over the training set and print out some metrics
    println("some metrics of Poisson Generalised Linear Models : ")
    val summary_poi = model_poi.summary
    println("Poisson: " + s"Coefficient Standard Errors: ${summary_poi.coefficientStandardErrors.mkString(",")}")
    println("Poisson: " + s"T Values: ${summary_poi.tValues.mkString(",")}")
    println("Poisson: " + s"P Values: ${summary_poi.pValues.mkString(",")}")
    println("Poisson: " + s"Dispersion: ${summary_poi.dispersion}")
    println("Poisson: " + s"Null Deviance: ${summary_poi.nullDeviance}")
    println("Poisson: " + s"Residual Degree Of Freedom Null: ${summary_poi.residualDegreeOfFreedomNull}")
    println("Poisson: " + s"Deviance: ${summary_poi.deviance}")
    println("Poisson: " + s"Residual Degree Of Freedom: ${summary_poi.residualDegreeOfFreedom}")
    println("Poisson: " + s"AIC: ${summary_poi.aic}")
    println("Poisson: " + "Deviance Residuals: ")
    summary_poi.residuals().show()

    // make the prediction
    var predictions_poi = model_poi.transform(testData)

    // calculate the RMSE
    var rmse_poi = evaluator.evaluate(predictions_poi)
    println("RMSE of Gaussion generalised linear = " + rmse_poi)
    val duration_poi = (System.nanoTime - time_poi_s) / 1e9d
    println("The training time for Poisson Generalised Linear Model is " + duration_poi + "s")

    sparkSession.stop()
  }
}