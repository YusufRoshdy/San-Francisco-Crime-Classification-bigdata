import os
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def log_model_info(model, output_dir, model_type="baseline", test_data=None):
    model_name = model.__class__.__name__
    with open(os.path.join(output_dir, "summary_%s_%s.txt" % (model_type, model_name)), "w") as f:
        f.write("Model name: %s\n\n" % model_name)
        f.write("Model parameters:\n")

        if model_name == "RandomForestClassificationModel":
            f.write("Number of trees: %s\n" % model.getNumTrees)
            f.write("Max depth: %s\n" % model._java_obj.getMaxDepth())

            f.write("Feature importances: %s\n" % model.featureImportances)
            f.write("Number of nodes: %s\n" % model.numNodes)
        elif model_name == "LogisticRegressionModel":
            f.write("(regParam): %s\n" % model._java_obj.getRegParam())
            f.write("(elasticNetParam): %s\n" % model._java_obj.getElasticNetParam())
            f.write("(maxIter): %s\n" % model._java_obj.getMaxIter())
            f.write("(tol): %s\n" % model._java_obj.getTol())

            f.write("Coefficients: %s\n" % model.coefficientMatrix)
            f.write("Intercept: %s\n" % model.interceptVector)
        elif model_name == "DecisionTreeClassificationModel":
            f.write("Max depth: %s\n" % model._java_obj.getMaxDepth())
            f.write("Max bins: %s\n" % model._java_obj.getMaxBins())
        else:
            print("Model not supported")

        f.write("=" * 50)
        f.write("\n")

        if hasattr(model, "summary"):
            f.write("Accuracy: %s\n\n" % model.summary.accuracy)
            f.write("False positive rate by label:\n%s\n\n" % model.summary.falsePositiveRateByLabel)
            f.write("True positive rate by label:\n%s\n\n" % model.summary.truePositiveRateByLabel)
            f.write("F-measure by label:\n%s\n\n" % model.summary.fMeasureByLabel())
            f.write("Precision by label:\n%s\n\n" % model.summary.precisionByLabel)
            f.write("Recall by label:\n%s\n\n" % model.summary.recallByLabel)
        else:
            predictions = model.transform(test_data)
            accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(predictions.count())
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
    print("Loading model: %s" % model_name)
    model = None
    if "Random Forest" in model_name:
        from pyspark.ml.classification import RandomForestClassificationModel
        model = RandomForestClassificationModel.load(model_name)
    elif "LogisticRegression" in model_name:
        from pyspark.ml.classification import LogisticRegressionModel
        model = LogisticRegressionModel.load(model_name)
    elif "DecisionTree" in model_name:
        from pyspark.ml.classification import DecisionTreeClassificationModel
        model = DecisionTreeClassificationModel.load(model_name)
    else:
        print("Model not supported")
    return model

def get_models_predictions(models_dir, test_data):
    for model_name in os.listdir(models_dir):
        if ".model" not in model_name:
            continue
    
        model = load_model(os.path.join(models_dir, model_name))
        predictions = model.transform(test_data)
        predictions.coalesce(1)\
            .select("prediction",'label')\
            .write\
            .mode("overwrite")\
            .format("csv")\
            .option("sep", ",")\
            .option("header","true")\
            .csv("./output/%s_predictions.csv" % model_name)
        print("Predictions for %s saved" % model_name)

