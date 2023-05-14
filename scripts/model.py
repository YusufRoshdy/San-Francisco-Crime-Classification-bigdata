from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
    BooleanType,
    DateType,
    TimestampType,
    DoubleType,
)
from pyspark.ml.feature import (
    ChiSqSelector,
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
)
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    HashingTF,
    IDF,
    Word2Vec,
    Tokenizer,
    MinMaxScaler,
    Word2VecModel,
    Word2Vec,
)

from pyspark.sql.functions import from_unixtime, col
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCols
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import (
    udf,
    col,
    explode,
    dayofmonth,
    month,
    year,
    radians,
    sin,
    cos,
    lit,
)
from pyspark.sql import DataFrame
import math

from pyspark.sql import SparkSession
import pandas as pd
import utils
import os

# suppress warnings
import warnings

warnings.filterwarnings("ignore")

# add python random seed
import random

random.seed(42)


class DateTimeTransformer(Transformer, HasInputCol, HasOutputCols):
    def __init__(self, inputCol=None, outputCols=None):
        super(DateTimeTransformer, self).__init__()
        self._set(inputCol=inputCol)
        self._set(outputCols=outputCols)

    def _transform(self, df):
        input_col = self.getInputCol()
        output_cols = self.getOutputCols()

        df = df.withColumn(output_cols[0], sin(2 * math.pi * col(input_col) / 12))
        df = df.withColumn(output_cols[1], cos(2 * math.pi * col(input_col) / 12))

        return df


class LoggingEvaluator(MulticlassClassificationEvaluator):
    def __init__(self, **kwargs):
        super(LoggingEvaluator, self).__init__(**kwargs)

    def _evaluate(self, dataset):
        metric = super(LoggingEvaluator, self)._evaluate(dataset)
        print("The score for current fold:", metric)
        print("====================================")
        return metric


class HepyrParameterTuning:
    def __init__(
        self,
        model_name,
        fine_tune_method="cv_grid_search",
        num_folds=3,
        seed=42,
        models_dir="./models/",
        output_dir="./output/",
    ):
        self.model_name = model_name
        self.fine_tune_method = fine_tune_method
        self.num_folds = num_folds
        self.seed = seed
        self.models_dir = models_dir
        self.output_dir = output_dir
        if model_name == "logistic_regression":
            lr = LogisticRegression(
                featuresCol="features", labelCol="label", predictionCol="prediction"
            )
            self.param_grid = (
                ParamGridBuilder()
                .addGrid(lr.regParam, [0.0, 0.1, 0.5])
                .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
                .addGrid(lr.maxIter, [5, 10])
                .addGrid(lr.tol, [1e-8, 1e-6, 1e-4])
                .build()
            )
            self.model = lr

        elif model_name == "random_forest":
            rf = RandomForestClassifier(
                featuresCol="features",
                labelCol="label",
                predictionCol="prediction",
                seed=self.seed,
            )
            self.param_grid = (
                ParamGridBuilder()
                .addGrid(rf.maxDepth, [5, 10, 15])
                .addGrid(rf.numTrees, [10, 20, 30])
                .build()
            )
            self.param_grid = (
                ParamGridBuilder()
                .addGrid(rf.maxDepth, [15])
                .addGrid(rf.numTrees, [1])
                .addGrid(rf.maxBins, [20])
                .build()
            )
            self.model = rf
        elif model_name == "decision_tree":
            dt = DecisionTreeClassifier(
                featuresCol="features",
                labelCol="label",
                predictionCol="prediction",
                seed=self.seed,
            )
            # self.param_grid = (
            #     ParamGridBuilder()
            #     .addGrid(dt.maxDepth, [5, 10, 15])
            #     .addGrid(dt.maxBins, [10, 20, 30])
            #     .build()
            # )
            self.param_grid = (
                ParamGridBuilder()
                .addGrid(dt.maxDepth, [10])
                .addGrid(dt.maxBins, [30])
                .build()
            )
            self.model = dt
        else:
            raise ValueError("Model", model_name, " is not supported")

    def tune_model(self, train_data, test_data):
        # get the shape of the data
        print(
            "Train data shape:",
            train_data.count(),
            "rows, {len(train_data.columns)} columns",
        )
        print("Fine tuning", self.model_name, "model")
        if self.fine_tune_method == "cv_grid_search":
            # create the cross validator
            logging_evaluator = LoggingEvaluator(
                labelCol="label", predictionCol="prediction", metricName="accuracy"
            )
            cross_val = CrossValidator(
                estimator=self.model,
                estimatorParamMaps=self.param_grid,
                evaluator=logging_evaluator,
                numFolds=self.num_folds,
            )

            # fit the model
            cv_model = cross_val.fit(train_data)

            # get the best model
            best_model = cv_model.bestModel

            utils.log_model_info(
                best_model, self.output_dir, "best", test_data=test_data
            )

            # save the best model
            model_name = best_model.__class__.__name__
            best_model.write().overwrite().save(
                os.path.join(self.models_dir, "best_" + model_name + ".model")
            )
        else:
            self.fine_tune_rf(train_data, test_data)

    def fine_tune_rf(self, train_data, test_data):
        max_depths = [5, 10, 15]
        num_trees = [10, 20, 30]

        # Initialize the best parameters and the best score
        best_score = 0
        best_params = None

        # Loop over the parameters
        for num_tree in num_trees:
            for max_depth in max_depths:
                # Create the model
                rf = RandomForestClassifier(
                    labelCol="label",
                    featuresCol="features",
                    numTrees=num_tree,
                    maxDepth=max_depth,
                    seed=self.seed,
                )
                # Fit the model
                model = rf.fit(train_data)
                # Make predictions
                predictions = model.transform(test_data)
                # Initialize evaluator
                evaluator = MulticlassClassificationEvaluator(
                    labelCol="label", predictionCol="prediction", metricName="accuracy"
                )
                # Compute the accuracy on the test set
                accuracy = evaluator.evaluate(predictions)
                # Print the parameters and the score
                print(
                    "NumTrees:",
                    num_tree,
                    ", MaxDepth:",
                    max_depth,
                    ", Score:",
                    accuracy,
                )
                # Check if we got a better score
                if accuracy > best_score:
                    best_score = accuracy
                    best_params = (num_tree, max_depth)

        print("Best parameters:", best_params, ", Score:", best_score)
        # save the best model
        rf = RandomForestClassifier(
            labelCol="label",
            featuresCol="features",
            numTrees=best_params[0],
            maxDepth=best_params[1],
            seed=self.seed,
        )
        model = rf.fit(train_data)

        # save the model summary to the output directory
        utils.log_model_info(model, self.output_dir, "best")

        # save the best model
        model_name = model.__class__.__name__
        model.write().overwrite().save(
            os.path.join(self.models_dir, "best_" + "model_name" + ".model")
        )


class SanFranciscoCrimeClassification:
    def __init__(self, data_path, output_dir, models_dir):
        spark = (
            SparkSession.builder.appName("BDT Project")
            .config("spark.sql.catalogImplementation", "hive")
            .config("hive.metastore.uris", "thrift://sandbox-hdp.hortonworks.com:9083")
            .config("spark.sql.avro.compression.codec", "snappy")
            .enableHiveSupport()
            .getOrCreate()
        )

        spark.conf.set(
            "spark.sql.repl.eagerEval.enabled", True
        )  # Property used to format output tables better
        spark.sparkContext.setLogLevel("ERROR")
        self.spark = spark
        self.data_path = data_path
        self.output_dir = output_dir
        self.models_dir = models_dir

    def read_data_spark(self, data_path):
        df = self.spark.read.format("avro").table("projectdb.crime_data")
        df.createOrReplaceTempView("df")

        df = df.withColumn("Dates", (col("Dates") / 1000))
        df = df.withColumn("Dates", from_unixtime(col("Dates")).cast("timestamp"))
        df = (
            df.withColumnRenamed(" id", "Id")
            .withColumnRenamed("Dates", "Dates")
            .withColumnRenamed("category", "Category")
            .withColumnRenamed("descript", "Descript")
            .withColumnRenamed("day_of_week", "DayOfWeek")
            .withColumnRenamed("pd_district", "PdDistrict")
            .withColumnRenamed("resolution", "Resolution")
            .withColumnRenamed("address", "Address")
            .withColumnRenamed("x", "X")
            .withColumnRenamed("y", "Y")
        )
        print(df)
        return df

    def feature_selection(self):
        cat_cols = ["DayOfWeek", "PdDistrict", "Resolution"]

        df_new = self.read_data_spark(self.data_path)

        indexer = StringIndexer(inputCol="Category", outputCol="Category_index")
        df_new = indexer.fit(df_new).transform(df_new)

        # convert categorical columns to numerical
        for col in cat_cols:
            indexer = StringIndexer(inputCol=col, outputCol=col + "_index")
            encoder = OneHotEncoder(
                inputCols=[col + "_index"], outputCols=[col + "_vec"]
            )
            df_new = indexer.fit(df_new).transform(df_new)
            df_new = encoder.fit(df_new).transform(df_new)

        # assemble the features
        assembler = VectorAssembler(
            inputCols=["DayOfWeek_vec", "PdDistrict_vec", "Resolution_vec"],
            outputCol="features",
        )

        df_new = assembler.transform(df_new)

        # select the features
        selector = ChiSqSelector(
            numTopFeatures=10,
            featuresCol="features",
            outputCol="selectedFeatures",
            labelCol="Category_index",
        )
        selector_model = selector.fit(df_new)
        result = selector_model.transform(df_new)

        return result

    def prepare_pipeline(self):
        print("Preparing pipeline...")

        df_new = self.read_data_spark(self.data_path)

        # feature selection
        # df_new = self.feature_selection()

        cat_features = ["DayOfWeek", "PdDistrict", "Resolution", "Category"]
        text_features = ["Descript"]

        df_new = df_new.withColumn("month", month("Dates"))
        df_new = df_new.withColumn("year", year("Dates"))
        df_new = df_new.withColumn("day", dayofmonth("Dates"))
        df_new = df_new.withColumn(
            "hour", df_new.Dates.cast("string").substr(12, 2).cast("int")
        )
        df_new = df_new.withColumn(
            "minute", df_new.Dates.cast("string").substr(15, 2).cast("int")
        )

        date_time_features = ["day", "month", "year", "hour", "minute"]

        # convert (X, Y) to float
        df_new = df_new.withColumn("X", df_new["X"].cast("float"))
        df_new = df_new.withColumn("Y", df_new["Y"].cast("float"))

        # convert the string labels to numbers
        indexers = [
            StringIndexer(inputCol=column, outputCol=column + "_index").fit(df_new)
            for column in cat_features
        ]

        # convert the numbers to one hot vectors
        encoders = [
            OneHotEncoder(inputCol=column + "_index", outputCol=column + "_vec")
            for column in cat_features
        ]

        # tokenize the text
        tokenizers = [
            Tokenizer(inputCol=column, outputCol=column + "_words")
            for column in text_features
        ]

        # convert the text to vectors using Word2Vec
        word2Vecs = [
            Word2Vec(
                vectorSize=3,
                minCount=0,
                inputCol=column + "_words",
                outputCol=column + "_vec",
            )
            for column in text_features
        ]

        date_time_encoders = [
            DateTimeTransformer(
                inputCol=column, outputCols=[column + "_sin", column + "_cos"]
            )
            for column in date_time_features
        ]

        # assemble the features
        assembler = VectorAssembler(
            inputCols=[
                "DayOfWeek_vec",
                "PdDistrict_vec",
                "Resolution_vec",
                "Descript_vec",
                "X",
                "Y",
                "day_sin",
                "day_cos",
                "month_sin",
                "month_cos",
                "year_sin",
                "year_cos",
                "hour_sin",
                "hour_cos",
                "minute_sin",
                "minute_cos",
            ],
            outputCol="features",
        )

        # create a pipeline
        pipeline = Pipeline(
            stages=indexers
            + encoders
            + tokenizers
            + word2Vecs
            + date_time_encoders
            + [assembler]
        )

        print("Fitting the pipeline...")

        # fit the pipeline
        model_pipeline = pipeline.fit(df_new)
        transformed_data = (
            model_pipeline.transform(df_new)
            .select("features", "Category_index")
            .withColumnRenamed("Category_index", "label")
        )

        return transformed_data

    def train_model(self, model_name):
        # prepare the data
        transformed_data = self.prepare_pipeline()

        # split the data
        train_data, test_data = transformed_data.randomSplit([0.7, 0.3], seed=42)

        # print the first 5 rows of the training data
        print("The first 5 rows of the training data:")
        train_data.show(5, truncate=False)

        print("Training the model...")
        if model_name == "logistic_regression":
            # define logistic regression to be reproducible
            lr = LogisticRegression(
                featuresCol="features", labelCol="label", maxIter=10
            )  # define the model
            lr_model = lr.fit(train_data)  # fit the model

            for iteration, loss in enumerate(lr_model.summary.objectiveHistory):
                print("Iteration", iteration, ": Loss =", loss)

            # save the summary in a file in the output path
            utils.log_model_info(lr_model, self.output_dir)

            # save the model
            model_name = lr.__class__.__name__
            lr_model.write().overwrite().save(
                os.path.join(self.models_dir, "baseline_" + model_name + ".model")
            )

        elif model_name == "random_forest":
            rf = RandomForestClassifier(
                labelCol="label", featuresCol="features", seed=42
            )  # define the model

            rf_model = rf.fit(train_data)  # fit the model

            # save the summary in a file in the output path
            utils.log_model_info(rf_model, self.output_dir)

            # save the model
            model_name = rf.__class__.__name__
            rf_model.write().overwrite().save(
                os.path.join(self.models_dir, "baseline_" + model_name + ".model")
            )

        # add the Decision Tree model
        elif model_name == "decision_tree":
            dt = DecisionTreeClassifier(
                labelCol="label", featuresCol="features", seed=42
            )

            dt_model = dt.fit(train_data)

            # save the summary in a file in the output path
            utils.log_model_info(dt_model, self.output_dir, test_data=test_data)

            # save the model
            model_name = dt.__class__.__name__
            dt_model.write().overwrite().save(
                os.path.join(self.models_dir, "baseline_" + model_name + ".model")
            )

        else:
            raise ValueError("Invalid model name!")


if __name__ == "__main__":
    model_name = "logistic_regression"
    # create an instance of the class
    modeling = SanFranciscoCrimeClassification(
        "./data/train.csv", "./output", "./models"
    )

    # train the model
    modeling.train_model(model_name)

    # processed_data = modeling.prepare_pipeline()
    # train_data, test_data = processed_data.randomSplit([0.7, 0.3], seed=42)

    # # fine_tuner = HepyrParameterTuning(model_name, num_folds=2)
    # # fine_tuner.tune_model(train_data, test_data)

    # utils.get_models_predictions("./models", test_data=test_data)
