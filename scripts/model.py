from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
    BooleanType,
    DateType,
    TimestampType,
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
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCols
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

# suppress warnings
import warnings

warnings.filterwarnings("ignore")


class DateTimeTransformer(Transformer, HasInputCol, HasOutputCols):
    def __init__(self, inputCol=None, outputCols=None):
        super(DateTimeTransformer, self).__init__()
        self._set(inputCol=inputCol)
        self._set(outputCols=outputCols)

    def _transform(self, df: DataFrame) -> DataFrame:
        input_col = self.getInputCol()
        output_cols = self.getOutputCols()

        df = df.withColumn(output_cols[0], sin(2 * math.pi * col(input_col) / 12))
        df = df.withColumn(output_cols[1], cos(2 * math.pi * col(input_col) / 12))

        return df


class SanFranciscoCrimeClassification:
    def __init__(self, spark, data_path):
        port = 4050
        spark = (
            SparkSession.builder.master("local[*]")
            .appName("Colab")
            .config("spark.ui.port", str(port))
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .getOrCreate()
        )
        spark.conf.set(
            "spark.sql.repl.eagerEval.enabled", True
        )  # Property used to format output tables better
        self.spark = spark
        self.data_path = data_path

    def read_data_spark(self, data_path):
        schema = StructType(
            [
                StructField("Dates", TimestampType(), True),
                StructField("Category", StringType(), True),
                StructField("Descript", StringType(), True),
                StructField("DayOfWeek", StringType(), True),
                StructField("PdDistrict", StringType(), True),
                StructField("Resolution", StringType(), True),
                StructField("Address", StringType(), True),
                StructField("X", StringType(), True),
                StructField("Y", StringType(), True),
            ]
        )

        df = self.spark.read.csv(data_path, header=True, schema=schema)
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

        df_new = self.read_data_spark(self.data_path)

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
            Tokenizer(inputCol=column, outputCol=f"{column}_words")
            for column in text_features
        ]

        # convert the text to vectors using Word2Vec
        word2Vecs = [
            Word2Vec(
                vectorSize=3,
                minCount=0,
                inputCol=f"{column}_words",
                outputCol=f"{column}_vec",
            )
            for column in text_features
        ]

        date_time_encoders = [
            DateTimeTransformer(
                inputCol=column, outputCols=[f"{column}_sin", f"{column}_cos"]
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

        # fit the pipeline
        model_pipeline = pipeline.fit(df_new)
        transformed_data = model_pipeline.transform(df_new)

        return transformed_data

    def train_model(self, model_name, output_path):

        # prepare the data
        transformed_data = self.prepare_pipeline()

        # split the data
        train_data, test_data = transformed_data.randomSplit([0.7, 0.3])

        if model_name == "logistic_regression":
            lr = LogisticRegression(
                featuresCol="features", labelCol="label", maxIter=10
            )  # define the model
            lr_model = lr.fit(train_data)  # fit the model

            for iteration, loss in enumerate(lr_model.summary.objectiveHistory):
                print(f"Iteration {iteration}: Loss = {loss}")

            # save the summary in a file in the output path
            with open(output_path + f"/summary_base_{model_name}_model.txt", "w") as f:
                f.write(f"Accuracy: {lr_model.summary.accuracy}\n")
                f.write(
                    f"False positive rate by label:\n{lr_model.summary.falsePositiveRateByLabel}\n"
                )
                f.write(
                    f"True positive rate by label:\n{lr_model.summary.truePositiveRateByLabel}\n"
                )
                f.write(f"F-measure by label:\n{lr_model.summary.fMeasureByLabel()}\n")
                f.write(f"Precision by label:\n{lr_model.summary.precisionByLabel}\n")
                f.write(f"Recall by label:\n{lr_model.summary.recallByLabel}\n")
                f.write(f"Area under ROC: {lr_model.summary.areaUnderROC}\n")
                f.write(f"ROC: {lr_model.summary.roc}\n")
                f.write(
                    f"Objective history: {[float(loss) for loss in lr_model.summary.objectiveHistory]}\n"
                )

        elif model_name == "random_forest":
            rf = RandomForestClassifier(
                labelCol="label", featuresCol="features", seed=24
            ) # define the model

            rf_model = rf.fit(train_data) # fit the model

            # save the summary in a file in the output path
            with open(output_path + f"/summary_base_{model_name}_model.txt", "w") as f:
                f.write(f"Accuracy: {rf_model.summary.accuracy}\n")
                f.write(
                    f"False positive rate by label:\n{rf_model.summary.falsePositiveRateByLabel}\n"
                )
                f.write(
                    f"True positive rate by label:\n{rf_model.summary.truePositiveRateByLabel}\n"
                )
                f.write(f"F-measure by label:\n{rf_model.summary.fMeasureByLabel()}\n")
                f.write(f"Precision by label:\n{rf_model.summary.precisionByLabel}\n")
                f.write(f"Recall by label:\n{rf_model.summary.recallByLabel}\n")
                f.write(f"Area under ROC: {rf_model.summary.areaUnderROC}\n")
                f.write(f"ROC: {rf_model.summary.roc}\n")

        else:
            # the current model is not supported
            print("The model is not supported")
