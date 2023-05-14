"""
This module contains the following classes: DateTimeTransformer, LoggingEvaluator,
HepyrParameterTuning, and SanFranciscoCrimeClassification.

These classes are used for transforming data, evaluating machine learning models,
parameter tuning, and classifying crime in San Francisco, respectively.
"""

import random
import os
import math
import warnings


from pyspark.ml.feature import (
    ChiSqSelector,
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    Word2Vec,
    Tokenizer,
)

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCols
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import (
    col,
    dayofmonth,
    month,
    year,
    sin,
    cos,
    from_unixtime,
)

from pyspark.sql import SparkSession

import utils


warnings.filterwarnings("ignore")

# add python random seed
random.seed(42)


class DateTimeTransformer(Transformer, HasInputCol, HasOutputCols):
    """
    DateTimeTransformer is a class for transforming datetime features
    into two separate features for the sine and cosine transformations of the date time.

    This class inherits from Transformer, HasInputCol, and HasOutputCols.
    """

    def __init__(self, input_col=None, output_cols=None):
        super(DateTimeTransformer, self).__init__()
        self._set(inputCol=input_col)
        self._set(outputCols=output_cols)

    def _transform(self, dataframe):
        input_col = self.getInputCol()
        output_cols = self.getOutputCols()

        dataframe = dataframe.withColumn(
            output_cols[0], sin(2 * math.pi * col(input_col) / 12)
        )
        dataframe = dataframe.withColumn(
            output_cols[1], cos(2 * math.pi * col(input_col) / 12)
        )

        return dataframe


class LoggingEvaluator(MulticlassClassificationEvaluator):
    """
    LoggingEvaluator is a class for evaluating multiclass classification models
    and logging the results.

    This class inherits from MulticlassClassificationEvaluator.
    """

    def _evaluate(self, dataset):
        metric = super(LoggingEvaluator, self)._evaluate(dataset)
        print("The score for current fold:", metric)
        print("====================================")
        return metric


class HepyrParameterTuning:
    """
    HepyrParameterTuning is a class for tuning the parameters of a machine learning model.

    It allows for systematic grid search or random search over specified parameter ranges.
    """

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
            logistic_regression = LogisticRegression(
                featuresCol="features", labelCol="label", predictionCol="prediction"
            )
            self.param_grid = (
                ParamGridBuilder()
                .addGrid(logistic_regression.regParam, [0.0, 0.1, 0.5])
                .addGrid(logistic_regression.elasticNetParam, [0.0, 0.5, 1.0])
                .addGrid(logistic_regression.maxIter, [5, 10])
                .addGrid(logistic_regression.tol, [1e-8, 1e-6, 1e-4])
                .build()
            )
            self.model = logistic_regression

        elif model_name == "random_forest":
            random_forest = RandomForestClassifier(
                featuresCol="features",
                labelCol="label",
                predictionCol="prediction",
                seed=self.seed,
            )
            self.param_grid = (
                ParamGridBuilder()
                .addGrid(random_forest.maxDepth, [5, 10, 15])
                .addGrid(random_forest.numTrees, [10, 20, 30])
                .build()
            )
            self.param_grid = (
                ParamGridBuilder()
                .addGrid(random_forest.maxDepth, [15])
                .addGrid(random_forest.numTrees, [1])
                .addGrid(random_forest.maxBins, [20])
                .build()
            )
            self.model = random_forest
        elif model_name == "decision_tree":
            decision_tree = DecisionTreeClassifier(
                featuresCol="features",
                labelCol="label",
                predictionCol="prediction",
                seed=self.seed,
            )
            # self.param_grid = (
            #     ParamGridBuilder()
            #     .addGrid(decision_tree.maxDepth, [5, 10, 15])
            #     .addGrid(decision_tree.maxBins, [10, 20, 30])
            #     .build()
            # )
            self.param_grid = (
                ParamGridBuilder()
                .addGrid(decision_tree.maxDepth, [10])
                .addGrid(decision_tree.maxBins, [30])
                .build()
            )
            self.model = decision_tree
        else:
            raise ValueError("Model", model_name, " is not supported")

    def tune_model(self, train_data, test_data):
        """
        This method is used for tuning the hyperparameters of the machine learning model.

        It uses a grid search or a random search approach to find the best hyperparameters
        that optimize the performance of the model on the validation set.

        Parameters:
        train_data (pyspark.sql.DataFrame): The training data on which the model will be trained.
            It is assumed to be preprocessed and ready for training.

        test_data (pyspark.sql.DataFrame): The validation data on which the model's performance
            is evaluated during the tuning process. It is assumed to be preprocessed
            and ready for validation.

        Returns:
        model: The model with the best hyperparameters found during the tuning process.
        """
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
        """
        This method is used for fine-tuning the Random Forest classifier.

        It uses a grid search approach to find the best combination of 'max_depth' and 'num_trees'
        hyperparameters that optimize the performance of the model on the test data.

        Parameters:
        train_data (pyspark.sql.DataFrame): The training data on which
            the Random Forest model will be trained.
        test_data (pyspark.sql.DataFrame): The validation data on which
            the model's performance is evaluated during the tuning process.

        Returns:
        None: The function updates the best_params and best_score attributes of the class object.
        """
        max_depths = [5, 10, 15]
        num_trees = [10, 20, 30]

        # Initialize the best parameters and the best score
        best_score = 0
        best_params = None

        # Loop over the parameters
        for num_tree in num_trees:
            for max_depth in max_depths:
                # Create the model
                random_forest = RandomForestClassifier(
                    labelCol="label",
                    featuresCol="features",
                    numTrees=num_tree,
                    maxDepth=max_depth,
                    seed=self.seed,
                )
                # Fit the model
                model = random_forest.fit(train_data)
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
        random_forest = RandomForestClassifier(
            labelCol="label",
            featuresCol="features",
            numTrees=best_params[0],
            maxDepth=best_params[1],
            seed=self.seed,
        )
        model = random_forest.fit(train_data)

        # save the model summary to the output directory
        utils.log_model_info(model, self.output_dir, "best")

        # save the best model
        model_name = model.__class__.__name__
        model.write().overwrite().save(
            os.path.join(self.models_dir, "best_" + model_name + ".model")
        )


class SanFranciscoCrimeClassification:
    """
    SanFranciscoCrimeClassification is a class for building and training
    a model to classify crime in San Francisco.

    It involves data preprocessing, feature extraction, model training, and evaluation steps.
    """

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

    def read_data_spark(self):
        """
        This method is responsible for reading the crime data from a Spark table,
        preprocessing it and returning a DataFrame.

        The preprocessing steps include:
        1. Converting the 'Dates' column from UNIX timestamp (bigint) to a standard timestamp.
        2. Renaming the columns to more appropriate names for easier handling downstream.

        The method assumes the existence of a Spark table named 'projectdb.crime_data'
        with specific columns.

        Returns:
        df (pyspark.sql.DataFrame): The preprocessed DataFrame with appropriately named columns.

        Note: This function prints the DataFrame after preprocessing. Consider redirecting stdout
        if running in a non-interactive session.
        """
        dataframe = self.spark.read.format("avro").table("projectdb.crime_data")
        dataframe.createOrReplaceTempView("df")

        dataframe = dataframe.withColumn("Dates", (col("Dates") / 1000))
        dataframe = dataframe.withColumn(
            "Dates", from_unixtime(col("Dates")).cast("timestamp")
        )
        dataframe = (
            dataframe.withColumnRenamed(" id", "Id")
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
        print(dataframe)
        return dataframe

    def feature_selection(self):
        """
        This method applies feature selection to the dataset.

        The steps in the feature selection process are as follows:
        1. Convert 'Category' to numerical form with the help of StringIndexer.
        2. Convert categorical columns to numerical form with the help of StringIndexer
        and OneHotEncoder.
        3. Use VectorAssembler to assemble these features into a vector.

        The method modifies the DataFrame in-place, transforming the categorical columns
        'DayOfWeek', 'PdDistrict', and 'Resolution' to numerical form
        and assembling them into a vector.

        Returns:
        df_new (pyspark.sql.DataFrame): The DataFrame with the new features added.
        """
        cat_cols = ["DayOfWeek", "PdDistrict", "Resolution"]

        df_new = self.read_data_spark()

        indexer = StringIndexer(inputCol="Category", outputCol="Category_index")
        df_new = indexer.fit(df_new).transform(df_new)

        # convert categorical columns to numerical
        for category in cat_cols:
            indexer = StringIndexer(inputCol=category, outputCol=category + "_index")
            encoder = OneHotEncoder(
                inputCols=[category + "_index"], outputCols=[category + "_vec"]
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
        """
        This method prepares the data pipeline for the model.

        It reads the data using the `read_data_spark` method and performs
            several preprocessing steps.

        These steps include:
        1. Extracting date-time features from the 'Dates' column such as 'month', 'year',
            'day', 'hour', and 'minute'.
        2. Converting the time features to integer format.
        3. Converting categorical features into numerical form.

        The method constructs a new DataFrame with the newly created features and returns it.

        Returns:
        df_new (pyspark.sql.DataFrame): The DataFrame with the new features added.
        """
        print("Preparing pipeline...")

        df_new = self.read_data_spark()

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
                input_col=column, output_cols=[column + "_sin", column + "_cos"]
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
        """
        This method trains a model based on the given model name.

        It first prepares the data pipeline by calling the `prepare_pipeline` method and then
        splits the data into training and test sets. Depending on the 'model_name' parameter,
        it fits a Logistic Regression, Random Forest, or Decision Tree model to the training data.

        It also prints the first 5 rows of the training data and logs the model's information.
        The trained model is saved into a .model file in the specified models directory.

        Parameters:
        model_name (str): The name of the model to be trained. Can be 'logistic_regression',
                        'random_forest', or 'decision_tree'.

        Raises:
        ValueError: If an invalid 'model_name' is provided.

        Returns:
        None. The function saves the trained model to disk.
        """
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
            logistic_regression = LogisticRegression(
                featuresCol="features", labelCol="label", maxIter=10
            )  # define the model
            lr_model = logistic_regression.fit(train_data)  # fit the model

            for iteration, loss in enumerate(lr_model.summary.objectiveHistory):
                print("Iteration", iteration, ": Loss =", loss)

            # save the summary in a file in the output path
            utils.log_model_info(lr_model, self.output_dir)

            # save the model
            model_name = logistic_regression.__class__.__name__
            lr_model.write().overwrite().save(
                os.path.join(self.models_dir, "baseline_" + model_name + ".model")
            )

        elif model_name == "random_forest":
            random_forest = RandomForestClassifier(
                labelCol="label", featuresCol="features", seed=42
            )  # define the model

            rf_model = random_forest.fit(train_data)  # fit the model

            # save the summary in a file in the output path
            utils.log_model_info(rf_model, self.output_dir)

            # save the model
            model_name = random_forest.__class__.__name__
            rf_model.write().overwrite().save(
                os.path.join(self.models_dir, "baseline_" + model_name + ".model")
            )

        # add the Decision Tree model
        elif model_name == "decision_tree":
            decision_tree = DecisionTreeClassifier(
                labelCol="label", featuresCol="features", seed=42
            )

            dt_model = decision_tree.fit(train_data)

            # save the summary in a file in the output path
            utils.log_model_info(dt_model, self.output_dir, test_data=test_data)

            # save the model
            model_name = decision_tree.__class__.__name__
            dt_model.write().overwrite().save(
                os.path.join(self.models_dir, "baseline_" + model_name + ".model")
            )

        else:
            raise ValueError("Invalid model name!")


if __name__ == "__main__":
    model_names = [
        "logistic_regression",  # logistic regression model
        # "random_forest",  # random forest model
        # "decision_tree",  # decision tree model
    ]

    for model_name in model_names:
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
