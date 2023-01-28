import mlflow
import numpy as np
import os
#help(mlflow)

def run_(run_name=''):
    mlflow.set_experiment("miprimermlflow")

    with mlflow.start_run() as run:
        print("mi primer codigo con mlflow")
        print("modelo id rum:", run.info.run_uuid)
        mlflow.set_tag("mlflow.runName", run_name)
        
        mlflow.log_param("param1", np.random.randn())
        mlflow.log_metric("metric1", np.random.randn())
        #mlflow.set_tags("run_introduccion","run by profesor")

        if not os.path.exists("logs"):
            os.makedirs("logs")
        with open("logs/text.txt", "w") as f:
            f.write("holaaa ML flow")
        mlflow.log_artifacts("logs", artifact_path="artifacts")
        mlflow.end_run()


run_()

#correr como: python mlflow_hellow_word.py