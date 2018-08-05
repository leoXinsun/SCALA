import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.util.MLUtils

import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.rdd.RDD


object Exercise3 {
  def main(args: Array[String]): Unit = {
  
    val sparkSession = SparkSession.builder.master("local[16]").appName("exercise3").getOrCreate()
    val sc = sparkSession.sparkContext

    // Load and parse the data
    val data1 = sc.textFile("files/r1.train")
    val ratings1 = data1.map(_.split("::") match { case Array(user, item, rate, timestamp) =>
    Rating(user.toInt, item.toInt, rate.toDouble)
    })

    // Build the recommendation model using ALS
    val rank1 = 10
    val numIterations1 = 10
    val model1 = ALS.train(ratings1, rank1, numIterations1, 0.01)

    val test1 = sc.textFile("files/r1.test")
    val ratingstest1 = test1.map(_.split("::") match { case Array(user, item, rate, timestamp) =>Rating(user.toInt, item.toInt, rate.toDouble)
    })
    // Evaluate the model on rating data
    val usersProducts1 = ratingstest1.map { case Rating(user, product, rate) =>
    (user, product)
    }
    val predictions1 =
    model1.predict(usersProducts1).map { case Rating(user, product, rate) =>
        ((user, product), rate)
    }
    val ratesAndPreds1 = ratingstest1.map { case Rating(user, product, rate) =>
        ((user, product), rate)
    }.join(predictions1)
    val MSE1 = ratesAndPreds1.map { case ((user, product), (r1, r2)) =>
        val err = (r1 - r2)
        err * err
    }.mean()

    println("Mean Squared Error for the first split = " + MSE1)

    // Load and parse the data
    val data2 = sc.textFile("files/r2.train")
    val ratings2 = data2.map(_.split("::") match { case Array(user, item, rate, timestamp) =>
    Rating(user.toInt, item.toInt, rate.toDouble)
    })

    // Build the recommendation model using ALS
    val rank2 = 10
    val numIterations2 = 10
    val model2 = ALS.train(ratings2, rank2, numIterations2, 0.01)

    val test2 = sc.textFile("files/r2.test")
    val ratingstest2 = test2.map(_.split("::") match { case Array(user, item, rate, timestamp) =>Rating(user.toInt, item.toInt, rate.toDouble)
    })
    // Evaluate the model on rating data
    val usersProducts2 = ratingstest2.map { case Rating(user, product, rate) =>
    (user, product)
    }
    val predictions2 =
    model2.predict(usersProducts2).map { case Rating(user, product, rate) =>
        ((user, product), rate)
    }
    val ratesAndPreds2 = ratingstest2.map { case Rating(user, product, rate) =>
        ((user, product), rate)
    }.join(predictions2)
    val MSE2 = ratesAndPreds2.map { case ((user, product), (r1, r2)) =>
        val err = (r1 - r2)
        err * err
    }.mean()

    println("Mean Squared Error for the second split = " + MSE2)    

    // Load and parse the data
    val data3 = sc.textFile("files/r3.train")
    val ratings3 = data3.map(_.split("::") match { case Array(user, item, rate, timestamp) =>
    Rating(user.toInt, item.toInt, rate.toDouble)
    })

    // Build the recommendation model using ALS
    val rank3 = 10
    val numIterations3 = 10
    val model3 = ALS.train(ratings3, rank3, numIterations3, 0.01)

    val test3 = sc.textFile("files/r3.test")
    val ratingstest3 = test3.map(_.split("::") match { case Array(user, item, rate, timestamp) =>Rating(user.toInt, item.toInt, rate.toDouble)
    })
    // Evaluate the model on rating data
    val usersProducts3 = ratingstest3.map { case Rating(user, product, rate) =>
    (user, product)
    }
    val predictions3 =
    model3.predict(usersProducts3).map { case Rating(user, product, rate) =>
        ((user, product), rate)
    }
    val ratesAndPreds3 = ratingstest3.map { case Rating(user, product, rate) =>
        ((user, product), rate)
    }.join(predictions3)
    val MSE3 = ratesAndPreds3.map { case ((user, product), (r1, r2)) =>
        val err = (r1 - r2)
        err * err
    }.mean()

    println("Mean Squared Error for the third split = " + MSE3)

    // Load and parse the data
    val data4 = sc.textFile("files/r4.train")
    val ratings4 = data4.map(_.split("::") match { case Array(user, item, rate, timestamp) =>
    Rating(user.toInt, item.toInt, rate.toDouble)
    })

    // Build the recommendation model using ALS
    val rank4 = 10
    val numIterations4 = 10
    val model4 = ALS.train(ratings4, rank4, numIterations4, 0.01)

    val test4 = sc.textFile("files/r4.test")
    val ratingstest4 = test4.map(_.split("::") match { case Array(user, item, rate, timestamp) =>Rating(user.toInt, item.toInt, rate.toDouble)
    })
    // Evaluate the model on rating data
    val usersProducts4 = ratingstest4.map { case Rating(user, product, rate) =>
    (user, product)
    }
    val predictions4 =
    model4.predict(usersProducts4).map { case Rating(user, product, rate) =>
        ((user, product), rate)
    }
    val ratesAndPreds4 = ratingstest4.map { case Rating(user, product, rate) =>
        ((user, product), rate)
    }.join(predictions4)
    val MSE4 = ratesAndPreds4.map { case ((user, product), (r1, r2)) =>
        val err = (r1 - r2)
        err * err
    }.mean()

    println("Mean Squared Error for the forth split = " + MSE4)

    // Load and parse the data
    val data5 = sc.textFile("files/r5.train")
    val ratings5 = data5.map(_.split("::") match { case Array(user, item, rate, timestamp) =>
    Rating(user.toInt, item.toInt, rate.toDouble)
    })

    // Build the recommendation model using ALS
    val rank5 = 10
    val numIterations5 = 10
    val model5 = ALS.train(ratings5, rank5, numIterations5, 0.01)

    val test5 = sc.textFile("files/r5.test")
    val ratingstest5 = test5.map(_.split("::") match { case Array(user, item, rate, timestamp) =>Rating(user.toInt, item.toInt, rate.toDouble)
    })
    // Evaluate the model on rating data
    val usersProducts5 = ratingstest5.map { case Rating(user, product, rate) =>
    (user, product)
    }
    val predictions5 =
    model5.predict(usersProducts5).map { case Rating(user, product, rate) =>
        ((user, product), rate)
    }
    val ratesAndPreds5 = ratingstest5.map { case Rating(user, product, rate) =>
        ((user, product), rate)
    }.join(predictions5)
    val MSE5 = ratesAndPreds5.map { case ((user, product), (r1, r2)) =>
        val err = (r1 - r2)
        err * err
    }.mean()

    println("Mean Squared Error for the fifth split = " + MSE5)


    val movies=model1.productFeatures.map{case (a,b)=>Vectors.dense(b)}
    val users=model1.userFeatures.map{case (a,b)=>Vectors.dense(b)}
    val mat_movies: RowMatrix = new RowMatrix(movies)
    val mat_users: RowMatrix = new RowMatrix(users)
    val pc_movies: Matrix = mat_movies.computePrincipalComponents(2)
    val pc_users: Matrix = mat_users.computePrincipalComponents(2)
    val projected_movies: RowMatrix = mat_movies.multiply(pc_products)
    val projected_users: RowMatrix = mat_users.multiply(pc_users)

    val rdd_movies = projected_movies.rows.map( x => x.toArray.mkString(","))
    rdd_movies.coalesce(1).saveAsTextFile("files/movies_output")
    val rdd_users = projected_users.rows.map( x => x.toArray.mkString(","))
    rdd_users.coalesce(1).saveAsTextFile("files/users_output")

    sparkSession.stop()
  }
}
