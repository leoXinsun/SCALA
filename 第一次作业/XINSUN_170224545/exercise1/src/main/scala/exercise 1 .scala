import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Row
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}

object Exercise1 {
  def main(args: Array[String]): Unit = {
    val time = System.nanoTime
    // "Open the bridge" 
    val sparkSession = SparkSession.builder.master("local[2]").appName("exercise 1").getOrCreate()

    // Import the data as text
    val text = sparkSession.sparkContext.textFile("files/HIGGS.csv").repartition(40)

    // Separate into array
    val data_rdd = text.map(line => line.split(',').map(_.toDouble)).map(t => (t(1), Vectors.dense(t.take(29).drop(1))))

    // Convert to a dataframe
    import sparkSession.implicits._
    val data = data_rdd.toDF("label", "features")

    // Fit on whole dataset to include all labels in index.
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

    // Automatically identify categorical features, and index them.
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

    // Split the data into training and test sets 
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 170224545)

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

    // Decision Trees for Classification
    val dt_c = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
    val pipeline_dtc = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt_c, labelConverter))
    val paramGrid_dtc = new ParamGridBuilder().addGrid(dt_c.maxBins, Array(5, 10, 20)).addGrid(dt_c.maxDepth, Array(1, 3, 5)).build()
    val metric_dtc = "accuracy"
    val evaluator_dtc = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName(metric_dtc)
    val cv_dtc = new CrossValidator().setEstimator(pipeline_dtc).setEvaluator(evaluator_dtc).setEstimatorParamMaps(paramGrid_dtc).setNumFolds(5)
    val cvModel_dtc = cv_dtc.fit(trainingData)
    val pred_dtc = cvModel_dtc.transform(testData)
    val accuracy_dtc = evaluator_dtc.evaluate(pred_dtc)
    println("The Test Error applying Decision Trees for Classification = " + (1.0 - accuracy_dtc))
    // build the tree to know the most relevant features
    val bestDTM_dtc = cvModel_dtc.bestModel.asInstanceOf[PipelineModel].stages(2).asInstanceOf[DecisionTreeClassificationModel]
    bestDTM_dtc.getImpurity
    bestDTM_dtc.getMaxBins
    bestDTM_dtc.getMaxDepth


    // Decision Trees for Regression
    val dt_r = new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("features")
    val pipeline_dtr = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt_r, labelConverter))
    val paramGrid_dtr = new ParamGridBuilder().addGrid(dt_r.maxBins, Array(5, 10, 20)).addGrid(dt_r.maxDepth, Array(1, 3, 5)).build()
    val metric_dtr = "rmse"
    val evaluator_dtr = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName(metric_dtr)
    val cv_dtr = new CrossValidator().setEstimator(pipeline_dtr).setEvaluator(evaluator_dtr).setEstimatorParamMaps(paramGrid_dtr).setNumFolds(5)
    val cvModel_dtr = cv_dtr.fit(trainingData)
    val pred_dtr = cvModel_dtr.transform(testData)
    val rmse_dtr = evaluator_dtr.evaluate(pred_dtr)
    println("The RMSE applying Decision Trees for Regression = " + rmse_dtr)

    // Logistic Regression
    val lr = new LogisticRegression().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
    lr.setMaxIter(10).setRegParam(0.3)
    val pipeline_lr = new Pipeline().setStages(Array(labelIndexer, featureIndexer, lr, labelConverter))
    val paramGrid_lr = new ParamGridBuilder().addGrid(lr.regParam, Array(0.3, 0.6, 0.9)).build()
    val metric_lr = "rmse"
    val evaluator_lr = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName(metric_lr)
    val cv_lr = new CrossValidator().setEstimator(pipeline_lr).setEvaluator(evaluator_lr).setEstimatorParamMaps(paramGrid_lr).setNumFolds(5)
    val cvModel_lr = cv_lr.fit(trainingData)
    val pred_lr = cvModel_lr.transform(testData)
    val rmse_lr = evaluator_lr.evaluate(pred_lr)
    println("The RMSE applying Logistic Regression = " + rmse_lr)

    //Provide training times 
    val duration = (System.nanoTime - time) / 1e9d
    println("The training time is " + duration + "s")

    sparkSession.stop()
  }
}