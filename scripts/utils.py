import os

def log_model_info(model, output_dir, model_type = 'baseline'):
    model_name = model.__class__.__name__
    with open(os.path.join(output_dir, f"summary_{model_type}_{model_name}.txt"), "w") as f:
            # if the model is random forest
            f.write(f"Model name: {model_name}\n\n")
            f.write(f"Model parameters:")
            if model_name == "RandomForestClassifier":
                f.write(f"Number of trees: {model.getNumTrees}\n")
                f.write(f"Max depth: {model._java_obj.getMaxDepth()}\n")
            # if the model is logistic regression
            elif model_name == "LogisticRegressionModel":
                f.write('(regParam): {}\n'.format(model._java_obj.getRegParam()))
                f.write('(elasticNetParam): {}\n'.format(model._java_obj.getElasticNetParam()))
                f.write('(maxIter): {}\n'.format(model._java_obj.getMaxIter()))
                f.write('(tol): {}\n'.format(model._java_obj.getTol()))
            else:
                print('Model not supported')
            
            f.write("=" * 50)
            f.write("\n")

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