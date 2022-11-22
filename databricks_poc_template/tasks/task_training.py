from databricks_poc_template.common import Task
from databricks_poc_template import module

# General packages
import pandas as pd
import numpy as np
import mlflow
import json
from pyspark.sql.functions import *

# Import matplotlib packages
# from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
import pylab
from pylab import *
import matplotlib.cm as cm
import matplotlib.mlab as mlab

# Sklearn packages
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# XGBoost package
import xgboost as xgb

# Databricks packages
from mlflow.tracking import MlflowClient
import mlflow
import mlflow.sklearn #mlflow.lightgbm
from mlflow.models.signature import infer_signature
from mlflow.tracking.artifact_utils import get_artifact_uri
from databricks import feature_store
from databricks.feature_store import FeatureLookup


class TrainTask(Task):

    # Custom function
    def _train(self, **kwargs):
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
        raw_data_table = input_conf["raw_data_table"]  
        label_table = input_conf["label_table"] 
        fs_db = input_conf["fs_database"] 
        fs_enrich_1_table = input_conf["fs_enrich_1_table"]
        fs_enrich_2_table = input_conf["fs_enrich_2_table"]

        # Output
        output_conf = self.conf["data"]["output"]
        self.logger.info("output configs: {0}".format(output_conf))       

        db_out = output_conf["database"]   
        train_dataset = output_conf["train_dataset"] 
        test_dataset = output_conf["test_dataset"]         
        train_seed = output_conf['random_state']      
        train_ratio = output_conf['train_size']  

        # Model configs
        model_conf = self.conf["model"]
        self.logger.info("model configs: {0}".format(model_conf))  
 
        model_name = model_conf["model_name"] 
        experiment = model_conf["experiment_name"] 
        model_seed = model_conf['hyperparameters_fixed']['random_state']
        mlflow.set_experiment(experiment) # Define the MLFlow experiment location

        # =======================
        # 1. Loading the raw data (from feature store but let's discuss)
        # =======================

        # check this
        listing = self.dbutils.fs.ls("dbfs:/")
        for l in listing:
            self.logger.info(f"DBFS directory: {l}")           

        try:
            # Load the raw data and associated label tables
            raw_data = spark.table(f"{db_in}.{raw_data_table}")
            labels = spark.table(f"{db_in}.{label_table}")
            
            # Joining raw_data and labels
            raw_data_with_labels = raw_data.join(labels, ['trxn_id'])
            display(raw_data_with_labels)
            
            self.logger.info("Step 1.0 completed: Loaded historical raw data and labels")   
          
        except Exception as e:
            print("Errored on 1.0: data loading")
            print("Exception Trace: {0}".format(e))
            # print(traceback.format_exc())
            raise e  

        # ==================================
        # 2. Building the training dataset (by adding enrichment layers)
        # ==================================
        try:
            # Initialize the Feature Store client
            fs = feature_store.FeatureStoreClient()    
            # # Load feature store tables
            # enrich_1_data = fs.read_table(name=enrich_1_table)
            # enrich_2_data = fs.read_table(name=enrich_2_table)
            feature_lookups = [ 
                FeatureLookup( 
                table_name = f"{fs_db}.{fs_enrich_1_table}",
                feature_names = ['V21', 'V22', 'V23', 'V24'],
                lookup_key = 'trxn_id',
                ),
                FeatureLookup( 
                table_name = f"{fs_db}.{fs_enrich_2_table}",
                feature_names = ['V25', 'V26', 'V27', 'V28', 'Amount'],
                lookup_key = 'trxn_id',
                )
            ]

            # Create the training dataset (includes the raw input data merged with corresponding features from feature table)
            exclude_columns = ['trxn_id']
            training_set = fs.create_training_set(
                df = raw_data_with_labels,
                feature_lookups = feature_lookups,
                label = "Class",
                exclude_columns = exclude_columns
            )

            # Load the training dataset into a dataframe
            training_df = training_set.load_df()
            display(training_df)
            training_df.show(5)
        
            # Collect data into a Pandas array for training
            # features_and_label = training_df.columns
            data_pd = training_df.toPandas() #[features_and_label]

            # Do the train-test split
            train, test = train_test_split(data_pd, train_size=0.7, random_state=92)
            print(f'Train size: {train.shape[0]} rows')
            print(f'Test size: {test.shape[0]} rows')

            # Save train dataset
            train_df = spark.createDataFrame(train)
            train_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{db_out}.{train_dataset}")                    
            
            # Save test dataset
            test_df = spark.createDataFrame(test)            
            test_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{db_out}.{test_dataset}") 
     
            self.logger.info("Step 2. completed: Building the training dataset")   
          
        except Exception as e:
            print("Errored on 2.: Building the training dataset")
            print("Exception Trace: {0}".format(e))
            # print(traceback.format_exc())
            raise e  

        # ========================================
        # 1.3 Model training
        # ========================================
        try:            
            with mlflow.start_run() as run:    
                mlflow.xgboost.autolog()                      

                print("Active run_id: {}".format(run.info.run_id))
                self.logger.info("Active run_id: {}".format(run.info.run_id))

                # Model definition  
                base_estimator = xgb.XGBClassifier(objective='binary:logistic',
                                                   random_state=model_seed, 
                                                   n_jobs=-1)

                CV_rfc = GridSearchCV(estimator=base_estimator, 
                                    param_grid=model_conf['hyperparameters_grid'],
                                    cv=5)

                # Remove unneeded data (transaction id)
                x_train = train.drop(["Class"], axis=1)
                y_train = train.Class
                # x_test = test.drop(["target"], axis=1)
                # y_test = test.target

                # Cross validation model fit
                CV_rfc.fit(x_train, y_train)
                print(CV_rfc.best_params_)
                print(CV_rfc.best_score_)
                print(CV_rfc.best_estimator_)
                model = CV_rfc.best_estimator_

                # Tracking the model parameters
                train_version = module.get_table_version(spark,f"{db_out}.{train_dataset}")
                test_version = module.get_table_version(spark,f"{db_out}.{test_dataset}")
                fs_enrich_1_table_version = module.get_table_version(spark,f"{fs_db}.{fs_enrich_1_table}")
                fs_enrich_2_table_version = module.get_table_version(spark,f"{fs_db}.{fs_enrich_2_table}")
                mlflow.set_tag("train_version", train_version)
                mlflow.set_tag("test_version", test_version)
                mlflow.set_tag("fs_enrich_1_table_version", fs_enrich_1_table_version) # Feature store version
                mlflow.set_tag("fs_enrich_2_table_version", fs_enrich_2_table_version) # Feature store version
                mlflow.set_tag("train", f"{db_out}.{train_dataset}")
                mlflow.set_tag("test", f"{db_out}.{test_dataset}")
                mlflow.set_tag("raw_data", f"{db_in}.{raw_data_table}") # Fraud features base
                mlflow.set_tag("raw_labels", f"{db_in}.{label_table}") # Fraud label
                mlflow.set_tag("environment run", f"{env}") # Tag the environment where the run is done
                signature = infer_signature(x_train, model.predict(x_train))  

                # Add an random input example for the model
                input_example = {
                    "V1": 0.1,
                    "V2": -1.5,
                    "V3": -1.4,
                    "V4": 0.2,
                    'V5': 0.3,
                    'V6': 0.5,
                    'V7': -0.2,
                    'V8': 0.8,
                    'V9': -1,
                    'V10': 0.7,
                    'Amount': 1000
                }                               
                
                # Log the model
                # mlflow.sklearn.log_model(model, "model") #, registered_model_name="sklearn-rf")   

                # Register the model to MLflow MR as well as FS MR (should not register in DEV?)
                fs.log_model(
                    model,
                    artifact_path=model_name,
                    flavor=mlflow.xgboost,
                    training_set=training_set,
                    # registered_model_name=model_name,
                )
                
                # Register the model to the Model Registry
                print(mlflow.get_registry_uri())
                mlflow.xgboost.log_model(model, 
                                        model_name,
                                        registered_model_name=model_name,
                                        signature=signature,
                                        input_example=input_example)           

                self.logger.info("Step 3 completed: model training and saved to MLFlow")                

        except Exception as e:
            print("Errored on step 3: model training")
            print("Exception Trace: {0}".format(e))
            print(traceback.format_exc())
            raise e   
               
        
    def launch(self):
        self.logger.info("Launching train task")
        self._train()
        self.logger.info("Train task finished!")  

# if you're using python_wheel_task, you'll need the entrypoint function to be used in setup.py
def entrypoint():  # pragma: no cover
    task = TrainTask()
    task.launch()

# if you're using spark_python_task, you'll need the __main__ block to start the code execution
if __name__ == '__main__':
    entrypoint()



