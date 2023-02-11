import mlflow
import numpy as np
import os
help(mlflow)
 

def run_(run_name=""):
    mlflow.set_experiment("helloword")

    with mlflow.start_run() as run:
        print("mi primer run con ml flow")
        print("model run: ", run.info.run_uuid)
        mlflow.set_tag("mlflow.runName", run_name)
        
        mlflow.log_param("param1", np.random.randn())
        mlflow.log_metric("metric1", np.random.randint(1, 10, size=1))
        mlflow.set_tags("run_origin","python_profesor")

        if not os.path.exists("logs"):
            os.makedirs("logs")
        with open("logs/text.txt","w") as f:
            f.write("heyy!! mlflow")
        
        mlflow.log_artifacts("logs", artifact_path="artifact")
        mlflow.end_run()

run_("other")







