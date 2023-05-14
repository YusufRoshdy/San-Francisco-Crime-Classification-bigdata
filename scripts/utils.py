import os
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def log_model_info(model, output_dir, model_type="baseline", test_data=None):
    model_name = model.__class__.__name__
    with open(
        os.path.join(output_dir, f"summary_{model_type}_{model_name}.txt"), "w"
    ) as f:
        # if the model is random forest
        f.write(f"Model name: {model_name}\n\n")
        f.write(f"Model parameters:\n")

        if model_name == "RandomForestClassificationModel":
            f.write(f"Number of trees: {model.getNumTrees}\n")
            f.write(f"Max depth: {model._java_obj.getMaxDepth()}\n")

            # print the coefficients
            f.write(f"Feature importances: {model.featureImportances}\n")
            # print the tree structure
            f.write(f"Number of nodes: {model.numNodes}\n")
        # if the model is logistic regression
        elif model_name == "LogisticRegressionModel":
            f.write("(regParam): {}\n".format(model._java_obj.getRegParam()))
            f.write(
                "(elasticNetParam): {}\n".format(model._java_obj.getElasticNetParam())
            )
            f.write("(maxIter): {}\n".format(model._java_obj.getMaxIter()))
            f.write("(tol): {}\n".format(model._java_obj.getTol()))

            # print the coefficients
            f.write(f"Coefficients: {model.coefficients}\n")
            f.write(f"Intercept: {model.intercept}\n")

        elif model_name == "DecisionTreeClassificationModel":
            f.write(f"Max depth: {model._java_obj.getMaxDepth()}\n")
            f.write(f"Max bins: {model._java_obj.getMaxBins()}\n")
        else:
            print("Model not supported")

        f.write("=" * 50)
        f.write("\n")

        # if the model has no summary attribute, we can't get the metrics
        if hasattr(model, "summary"):
            f.write(f"Accuracy: {model.summary.accuracy}\n\n")
            f.write(
                f"False positive rate by label:\n{model.summary.falsePositiveRateByLabel}\n\n"
            )
            f.write(
                f"True positive rate by label:\n{model.summary.truePositiveRateByLabel}\n\n"
            )
            f.write(f"F-measure by label:\n{model.summary.fMeasureByLabel()}\n\n")
            f.write(f"Precision by label:\n{model.summary.precisionByLabel}\n\n")
            f.write(f"Recall by label:\n{model.summary.recallByLabel}\n\n")
        else:
            predictions = model.transform(test_data)
            f.write(f"Accuracy: {predictions.filter(predictions.label == predictions.prediction).count() / float(predictions.count())}\n\n")
            metrics = ["weightedPrecision", "weightedRecall", "f1"]
            for metric in metrics:
                evaluator = MulticlassClassificationEvaluator(metricName=metric)
                result = evaluator.evaluate(predictions)    
                f.write(f"{metric}: {result}\n")

            f.write("=" * 50)
            f.write("\n")
            # write: Model interpretation
            f.write(f"Model interpretation:\n\n")
            # print the coefficients
            f.write(f"Feature importances: {model.featureImportances}\n")
            # print the tree structure
            f.write(f"Number of nodes: {model.toDebugString}\n")