import mlflow
import numpy as np
import os
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,ConfusionMatrixDisplay,accuracy_score
# Import datasets for use
from sklearn import datasets
import argparse

digits =datasets.load_digits()


def make_model_name(experiment_name, run_name):
    return experiment_name+" "+run_name

if __name__ == "__main__":
    experiment_name ="mlflow_mnist_clasification"

    try:
        exp_id= mlflow.create_experiment(name=experiment_name)
    except Exception as e:
        exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    if not os.path.exists("images"):
        os.makedirs("images")
    descrip="""
            Este es un experimento para clasificacion usando NN
                """
    with mlflow.start_run(experiment_id=exp_id,description=descrip) as run:
        print("mlflow start!!!")
        print("run id", run.info.run_uuid)
        run_name = run.info.run_name

        _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
        for ax, image, label in zip(axes, digits.images, digits.target):
            ax.set_axis_off()
            ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
            ax.set_title("Training: %i" % label)
        plt.savefig("images/mnist_imagen.png")
        plt.close()
        mlflow.log_artifacts("images")

        n_samples =len(digits.images)
        data=digits.images.reshape((n_samples,-1))
        X_train, X_test, y_train, y_test = train_test_split(
            data, digits.target, test_size=0.2,random_state=123)
        
        parser= argparse.ArgumentParser()
        
        parser.add_argument('-act','--activation',type=str,choices=["relu","tanh"],help="este parametro es para activacion", default='relu')
        parser.add_argument('-lr','--alpha',type=float,default=0.01)
        parser.add_argument('-hi','--hidden',action='append',type= int)
        args= parser.parse_args()
        options =vars(args)

        for name in options:
            mlflow.log_param(name, options[name])

        clf=MLPClassifier(hidden_layer_sizes=args.hidden,activation=args.activation, alpha=args.alpha,max_iter=500)
        clf.fit(X_train,y_train)
        predict= clf.predict(X_test)
        reporte_dict =classification_report(y_test,predict,output_dict=True)
        acc =accuracy_score(y_test,predict)

        for name in reporte_dict["0"]:
            mlflow.log_metric(name, reporte_dict["0"][name])
        mlflow.log_metric("acc", acc)

        mlflow.sklearn.log_model(sk_model= clf,artifact_path='mnist-nn', registered_model_name=make_model_name(experiment_name,run_name))
        disp = ConfusionMatrixDisplay.from_predictions(y_test,predict)
        plt.savefig("images/cm.png")
        plt.close()


        

## Correr como: python mlflow_classification_mnist.py --activation "tanh" -lr 0.01 -hi 32 -hi 16
## mlflow ui --port 8999
        

        





    

