import matplotlib.pyplot as plt
import mlflow
import numpy as np
import os
import argparse
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import time
digits = datasets.load_digits()

def make_model_name(experiment_name, run_name):
        clock_time = time.ctime().replace(' ', '-')
        return experiment_name + '_' + run_name + '_'


if __name__ == "__main__":
   
    experiment_name ="clasificacion_mnist"
    try:
        exp_id = mlflow.create_experiment(name=experiment_name)
    except Exception as e:
        exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    if not os.path.exists("images"):
                os.makedirs("images")
    descrip= """
            En este experimento vamos a predecir imagenes de mnist usando NN
            """
    with mlflow.start_run(experiment_id=exp_id, description=descrip) as run:
        print("mi segundo run con ml flow")
        print("model run: ", run.info.run_uuid)
        run_name=run.info.run_name
        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.set_tags({"problema":"classificacion","data":"mnist" })


        _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
        for ax, image, label in zip(axes, digits.images, digits.target):
            ax.set_axis_off()
            ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
            ax.set_title("Training: %i" % label)
        plt.savefig("images/Training_data.png")
        plt.close()
        mlflow.log_artifacts("images")

        n_samples = len(digits.images)
        data = digits.images.reshape((n_samples, -1))
        print("slip dataset")
        X_train, X_test, y_train, y_test = train_test_split(
            data, digits.target, test_size=0.5, shuffle=False)
        
        parser = argparse.ArgumentParser()
        parser.add_argument('-lr', '--alpha', type=float, help='alpha',default=0.01)
        parser.add_argument('-act', '--activation', type=str,choices=['relu', 'tanh'], help='activation',default='relu')
        parser.add_argument('-hi','--hidden', action='append', help='hidden_layer',  type=int,  required=True)
        args = parser.parse_args()
        options = vars(args)
    
        for name in options:
           mlflow.log_param(name, options[name])
        
        print("entrenar MPL")
        clf = MLPClassifier(hidden_layer_sizes=args.hidden,activation=args.activation,\
        solver="adam", alpha= args.alpha,\
         learning_rate="adaptive",early_stopping=True ,max_iter=3000)

        clf.fit(X_train, y_train)

        predicted = clf.predict(X_test)
        
        metrics_dict= metrics.classification_report(y_test, predicted,output_dict=True)
        metrics0 = metrics_dict["0"]

        model_name = make_model_name(experiment_name, run_name) 
        for name in metrics0:
                mlflow.log_metric(name, metrics0[name])

        mlflow.sklearn.log_model(sk_model=clf, artifact_path='mnist-class-model', registered_model_name=model_name)
        print("generar iamgenes CM")
        disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
        disp.figure_.suptitle("Confusion Matrix")
        plt.savefig("images/Confusion_Matrix.png")
        plt.close()
        _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
        for ax, image, prediction in zip(axes, X_test, predicted):
            ax.set_axis_off()
            image = image.reshape(8, 8)
            ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
            ax.set_title(f"Prediction: {prediction}")
        plt.savefig("images/prediction_data.png")



        

    









