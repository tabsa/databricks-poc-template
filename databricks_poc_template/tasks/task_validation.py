from databricks_poc_template.common import Task
from databricks_poc_template import module

import pandas as pd
import numpy as np
import mlflow
import json

# Import of Sklearn packages
from sklearn.metrics import accuracy_score, roc_curve, f1_score, precision_recall_curve, auc, matthews_corrcoef

# Import matplotlib packages
# from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
import seaborn as sns
# import pylab
from pylab import *
# import matplotlib.cm as cm
# import matplotlib.mlab as mlab

def calc_confusion_matrix(y_true, y_pred):
    """Computes confusion matrix.""" 
    
    s_true = pd.Series((1-y_true).astype('int'), name='Actual')
    s_pred = pd.Series((1-y_pred).astype('int'), name='Predicted')

    df_conf_matrix = pd.crosstab(s_true, s_pred).T

    df_conf_matrix.columns = pd.Index(['P', 'N'], name='Actual')
    df_conf_matrix.index = pd.Index(['P', 'N'], name='Predicted')

    return df_conf_matrix

def performance_metric(y, y_hat, metric) -> float:

    if metric == 'roc_curve':
        fpr, tpr, _ = roc_curve(y, y_hat)
        score = auc(fpr, tpr)
    elif metric == 'precision_recall':
        precision, recall, _ = precision_recall_curve(y, y_hat)
        score = auc(recall, precision)
    elif metric == 'f1_score':
        score = f1_score(y, y_hat)
    elif metric == 'matthews_corr':
        score = matthews_corrcoef(y, y_hat)
    else:
        score = None
    
    return score

def show_confusion_matrix(y_true, y_pred, filename, labels=['Fraud', 'Normal']):
    """Visualizes confusion matrix."""
         
    df_conf_matrix = calc_confusion_matrix(y_true, y_pred)
    
    fig = plt.figure(figsize=(6, 6)) 
    
    sns.heatmap(df_conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d",\
                                cbar=False, center=1, vmin=0.5, vmax=0.5, linewidths=.5); 
    
    plt.title("Confusion matrix") 
    plt.xlabel('True class') 
    plt.ylabel('Predicted class') 
    plt.savefig(filename)

    return fig

class ValidationTask(Task):

    # Custom function
    def _validate(self, **kwargs):
        # ===========================
        # 0. Reading the config files
        # ===========================

        # Environment
        env = self.conf["environment"]
        self.logger.info("environment: {0}".format(env))

        # Input
        input_conf = self.conf["data"]["input"]
        self.logger.info("input configs: {0}".format(input_conf))

        db_in = input_conf["database"]  
        test_dataset = input_conf["test_dataset"]            

        # Output
        output_conf = self.conf["data"]["output"]
        self.logger.info("output configs: {0}".format(output_conf))       

        db_out = output_conf["database"]   
       
        # Model configs
        model_conf = self.conf["model"]
        self.logger.info("model configs: {0}".format(model_conf))  
 
        model_name = model_conf["model_name"] 
        experiment = model_conf["experiment_name"] 
        minimal_threshold = model_conf["minimal_threshold"] 
        mlflow.set_experiment(experiment) # Define the MLFlow experiment location

        # ========================
        # 1. Loading the Test data
        # ========================

        # check this
        listing = self.dbutils.fs.ls("dbfs:/")
        for l in listing:
            self.logger.info(f"DBFS directory: {l}")   

        try:
            # Load the raw data and associated label tables
            test_df = spark.table(f"{db_in}.{test_dataset}")
            test_pd = test_df.toPandas()
            # Separate X, y
            X_test = test_pd.drop(['Class'], axis=1)
            y_test = test_pd.Class.values            
    
            # print("Step 1. completed: Loaded Test data")   
            self.logger.info("Step 1. completed: Loaded Test data")   
          
        except Exception as e:
            print("Errored on 1.: data loading")
            print("Exception Trace: {0}".format(e))
            # print(traceback.format_exc())
            raise e    

        # ========================================
        # 2. Load model from MLflow Model Registry
        # ========================================
        try:   
            # Load model from MLflow experiment
            # Conditions:
            # - model accuracy should be higher than pre-defined threshold (defined in model.json)

            # Initialize MLflow client
            client = mlflow.tracking.MlflowClient()
            model_names = [m.name for m in client.search_registered_models()]
            print(model_names)

            # Extracting model & its information (latest model with tag 'None')
            mv = client.get_latest_versions(model_name, ['None'])[0]
            version = mv.version
            run_id = mv.run_id
            artifact_uri = client.get_model_version_download_uri(model_name, version)
            # model = mlflow.pyfunc.load_model(artifact_uri)            
            model = mlflow.xgboost.load_model(artifact_uri)            

            # print("Step 2. completed: load model from MLflow")  
            self.logger.info("Step 2. completed: load model from MLflow")                

        except Exception as e:
            print("Errored on step 2.: model loading from MLflow")
            print("Exception Trace: {0}".format(e))
            print(traceback.format_exc())
            raise e      

        # =============================================================
        # 3. Model validation (and tagging "staging") in Model Registry
        # =============================================================
        try:                      
            # Derive accuracy on TEST dataset
            y_test_prob = model.predict_proba(X_test)[:, 1]
            y_test_pred = model.predict(X_test) 
            test_pd['prediction'] = y_test_pred
            test_pd['score'] = y_test_prob
            test_df_out = spark.createDataFrame(test_pd)            
            test_df_out.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{db_out}.{test_dataset}")             

            # Accuracy and Confusion Matrix
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_roc_score = performance_metric(y_test, y_test_prob, 'roc_curve')
            test_prec_recall = performance_metric(y_test, y_test_prob, 'precision_recall')
            test_f1_score = performance_metric(y_test, y_test_pred, 'f1_score')
            test_matt_score = performance_metric(y_test, y_test_pred, 'matthews_corr')
            print('TEST accuracy = ',test_accuracy)
            print('TEST ROC score = ', test_roc_score)
            print('TEST prec-recall = ', test_prec_recall)
            print('TEST F1 score = ', test_f1_score)
            print('TEST Matthews correlation = ', test_matt_score)
            print('TEST Confusion matrix:')
            # Classes = ['P', 'N']
            # C = confusion_matrix(y_test, y_test_pred)
            # C_normalized = C / C.astype(np.float).sum()        
            # C_normalized_pd = pd.DataFrame(C_normalized,columns=Classes,index=Classes)
            C = calc_confusion_matrix(y_test, y_test_pred)
            # print(C_normalized_pd)   
            print(C)

            # Figure plot
            # fig = show_confusion_matrix(y_test, y_test_pred, 'confusion_matrix_TEST.png')
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # cax = ax.matshow(C,cmap='Blues')
            # plt.title('Confusion matrix of the classifier')
            # fig.colorbar(cax)
            # ax.set_xticklabels([''] + Classes)
            # ax.set_yticklabels([''] + Classes)
            # plt.xlabel('Predicted')
            # plt.ylabel('True')
            # plt.savefig("confusion_matrix_TEST.png")    

            with mlflow.start_run(run_id) as run:

                # Tracking performance metrics on TEST dataset   
                mlflow.log_metric("accuracy_TEST", test_accuracy)
                mlflow.log_metric("roc_score_TEST", test_roc_score)
                mlflow.log_metric("precision_recall_TEST", test_prec_recall)
                mlflow.log_metric("f1_score_TEST", test_f1_score)
                mlflow.log_metric("matth_corr_TEST", test_matt_score)
                # mlflow.log_metric("Confusion matrix", C)
                # mlflow.log_figure(fig, "confusion_matrix_TEST.png")  

                # IF we pass the validation, we push the model to Staging tag 
                print(f"Minimal accuracy threshold: {minimal_threshold:5.2f}")          
                if test_accuracy >= minimal_threshold: 
                    mlflow.set_tag("validation", "passed")
                    if env == 'staging': 
                        client.transition_model_version_stage(name=model_name, version=version, stage="Staging")
                else: 
                    mlflow.set_tag("validation", "failed")

                # Tracking the Test dataset (with predictions)
                test_version = module.get_table_version(spark,f"{db_out}.{test_dataset}")
                mlflow.set_tag("test_version", test_version)
                mlflow.set_tag("test", f"{db_out}.{test_dataset}")                    
                            
            # print("Step 3. completed: model validation")  
            self.logger.info("Step 3. completed: model validation")                

        except Exception as e:
            print("Errored on step 3.: model validation")
            print("Exception Trace: {0}".format(e))
            print(traceback.format_exc())
            raise e                       

    def launch(self):
        self.logger.info("Launching validation task")
        self._validate()
        self.logger.info("Validation task finished!")  

# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = ValidationTask()
    task.launch()

# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == '__main__':
    entrypoint()
