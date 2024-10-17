import mlflow
import mlflow.pytorch
from dvc.testing.benchmarks.fixtures import project
from ultralytics import YOLO
import os
import dvc.api

#intialize mlflow

local_mlflow_dir = "mlruns"
mlflow.set_tracking_uri(local_mlflow_dir)
mlflow.set_experiment("YOLOV8m-apple-detection")

# Start an MLflow run
with mlflow.start_run():

    # Load the Yolov8m model
    model = YOLO('yolov8m.pt') #Pretrained YOLOV8m model

    # Load the dataset (if required pull from the DVC version dataset)
    dataset_path = "/home/chaitu/Documents/Code/AppleProject/dataset/data.yaml"

    # Log Yolov8 model hyperparameters to MLflow
    mlflow.log_param("model", "Yolov8m-apple-detection")
    mlflow.log_param("epochs", 25)
    mlflow.log_param("img_size", 640)

    # Train the model
    results = model.train(data=dataset_path, epochs=25, imgsz=640, project="runs/train", name="yolov8m-apple-detection")

    # Log metrics from the results
    mlflow.log_metric("train/box_loss", results.box_loss[-1])
    mlflow.log_metric("train/cls_loss", results.cls_loss[-1])
    mlflow.log_metric("val/precision", results.metrics['precision'])
    mlflow.log_metric("val/recall", results.metrics['recall'])
    mlflow.log_metric("val/mAP_50", results.metrics['map50'])
    mlflow.log_metric("val/mAP_50_95", results.metrics['map'])
    mlflow.log_metric("AP_small", results.metrics['ap50_small'])  # AP for small objects
    mlflow.log_metric("AP_medium", results.metrics['ap50_medium'])  # AP for medium objects
    mlflow.log_metric("AP_large", results.metrics['ap50_large'])  # AP for large objects


    # Save the model to be tracked in MLflow
    model_path = "runs/train/yolov8m_apple/weights/best.pt"
    mlflow.pytorch.log_model(model, artifact_path="model")

print("DONE")