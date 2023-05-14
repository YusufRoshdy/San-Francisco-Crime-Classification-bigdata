"""
This module contains utility functions for logging model information, loading models,
and getting model predictions.
"""

import os

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import (
    RandomForestClassificationModel,
    DecisionTreeClassificationModel,
    LogisticRegressionModel,
)


def log_model_info(model, output_dir, model_type="baseline", test_data=None):
    """
    Writes model information to a text file.

    Parameters:
    - model: The model to be logged. Can be of type RandomForestClassificationModel,
        DecisionTreeClassificationModel, LogisticRegressionModel.
    - output_dir (str): The path of the directory where the log file will be saved.
    - model_type (str, optional): The type of the model. Defaults to "baseline".
    - test_data (DataFrame, optional): The test data to be used if model.summary is not available.
        Defaults to None.

    Returns:
    - None
    """
    model_name = model.__class__.__name__
    with open(
        os.path.join(output_dir, "summary_%s_%s.txt" % (model_type, model_name)), "w"
    ) as f:
        f.write("Model name: %s\n\n" % model_name)
        f.write("Model parameters:\n")
        java_obj = model._java_obj
        if model_name == "RandomForestClassificationModel":
            f.write("Number of trees: %s\n" % model.getNumTrees)
            f.write("Max depth: %s\n" % java_obj.getMaxDepth())

            f.write("Feature importances: %s\n" % model.featureImportances)
            f.write("Number of nodes: %s\n" % model.numNodes)
        elif model_name == "LogisticRegressionModel":
            f.write("(regParam): %s\n" % java_obj.getRegParam())
            f.write("(elasticNetParam): %s\n" % java_obj.getElasticNetParam())
            f.write("(maxIter): %s\n" % java_obj.getMaxIter())
            f.write("(tol): %s\n" % java_obj.getTol())

            f.write("Coefficients: %s\n" % model.coefficientMatrix)
            f.write("Intercept: %s\n" % model.interceptVector)
        elif model_name == "DecisionTreeClassificationModel":
            f.write("Max depth: %s\n" % java_obj.getMaxDepth())
            f.write("Max bins: %s\n" % java_obj.getMaxBins())
        else:
            print("Model not supported")

        f.write("=" * 50)
        f.write("\n")

        if hasattr(model, "summary"):
            f.write("Accuracy: %s\n\n" % model.summary.accuracy)
            f.write(
                "False positive rate by label:\n%s\n\n"
                % model.summary.falsePositiveRateByLabel
            )
            f.write(
                "True positive rate by label:\n%s\n\n"
                % model.summary.truePositiveRateByLabel
            )
            f.write("F-measure by label:\n%s\n\n" % model.summary.fMeasureByLabel())
            f.write("Precision by label:\n%s\n\n" % model.summary.precisionByLabel)
            f.write("Recall by label:\n%s\n\n" % model.summary.recallByLabel)
        else:
            predictions = model.transform(test_data)
            accuracy = predictions.filter(
                predictions.label == predictions.prediction
            ).count() / float(predictions.count())
            f.write("Accuracy: %s\n\n" % accuracy)
            metrics = ["weightedPrecision", "weightedRecall", "f1"]
            for metric in metrics:
                evaluator = MulticlassClassificationEvaluator(metricName=metric)
                result = evaluator.evaluate(predictions)
                f.write("%s: %s\n" % (metric, result))

            f.write("=" * 50)
            f.write("\n")
            f.write("Model interpretation:\n\n")
            f.write("Feature importances: %s\n" % model.featureImportances)
            f.write("Number of nodes: %s\n" % model.toDebugString)


def load_model(model_name):
    """
    Loads a saved model from a specified path.

    Parameters:
    - model_name (str): The path to the saved model.

    Returns:
    - model: The loaded model.
    """
    print("Loading model: %s" % model_name)
    model = None
    if "Random Forest" in model_name:
        model = RandomForestClassificationModel.load(model_name)
    elif "LogisticRegression" in model_name:
        model = LogisticRegressionModel.load(model_name)
    elif "DecisionTree" in model_name:
        model = DecisionTreeClassificationModel.load(model_name)
    else:
        print("Model not supported")
    return model


def get_models_predictions(models_dir, test_data):
    """
    Generates predictions for each model in a specified directory and saves them to a CSV file.

    Parameters:
    - models_dir (str): The path of the directory where the models are saved.
    - test_data (DataFrame): The data that the model will use to generate predictions.

    Returns:
    - None
    """
    for model_name in os.listdir(models_dir):
        if ".model" not in model_name:
            continue

        model = load_model(os.path.join(models_dir, model_name))
        predictions = model.transform(test_data)
        predictions.coalesce(1).select("prediction", "label").write.mode(
            "overwrite"
        ).format("csv").option("sep", ",").option("header", "true").csv(
            "./output/%s_predictions.csv" % model_name
        )
        print("Predictions for %s saved" % model_name)
