from pathlib import Path
from ultralytics import YOLO
import mlflow
import pandas as pd
import onnx
import mlflow.onnx
import time
from mlflow.tracking import MlflowClient
import os

def get_latest_train_folder(base_path):
    """
    Get the latest train folder based on the highest numerical suffix.

    Args:
        base_path (Path): The base path containing train folders.

    Returns:
        Path: The latest train folder path.
    """
    train_folders = [f for f in base_path.glob('train*') if f.is_dir()]
    if not train_folders:
        raise FileNotFoundError("No train folders found in the specified base path.")

    latest_folder = max(train_folders, key=lambda f: int(f.name.replace('train', '') or '0'))
    return latest_folder


def train_yolov8(cfg_file):
    """
    Trains a YOLOv8 model on a custom dataset and logs artifacts with MLflow.

    Args:
        cfg_file (Path): Path to the YOLOv8 configuration file.
    """
    with mlflow.start_run(log_system_metrics=True) as run:
        # Initialize model (you can choose a pre-trained model here)
        
        model = YOLO("yolov8n.pt")

        # Train the model
        model.train(data=str(cfg_file), epochs=50, imgsz=640)  # Pass cfg_file as keyword argument


        # Usage example of  moving all contents of the detect/train folder as artifacts to MLflow
        base_path = Path("runs") / "detect"
        train_folder = get_latest_train_folder(base_path)
        train_weight_folder = train_folder / "weights"

        # Log artifacts
        for item in train_folder.iterdir():
            if item.is_file():
                mlflow.log_artifact(str(item))
        for item in train_weight_folder.iterdir():
            if item.is_file():
                mlflow.log_artifact(str(item))

        # Load the results.csv file after training is complete
        # df = pd.read_csv("runs\\detect\\train\\results.csv")
        results_csv_path = train_folder / "results.csv"
        df = pd.read_csv(results_csv_path)
        
        # Log parameters from args_file
        params = {}
        with open(args_file, "r") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, value = line.strip().split(":", 1)
                params[key.strip()] = value.strip()
        mlflow.log_params(params)

        # Log metrics from results.csv
        df.columns = [col.strip() for col in df.columns]
        precision = df["metrics/precision(B)"].mean()
        recall = df["metrics/recall(B)"].mean()
        mAP50 = df["metrics/mAP50(B)"].mean()
        mAP50_95 = df["metrics/mAP50-95(B)"].mean()
        train_boxloss = df["train/box_loss"].mean()
        train_clsloss = df["train/cls_loss"].mean()
        train_dflloss = df["train/dfl_loss"].mean()
        val_boxloss = df["val/box_loss"].mean()
        val_clsloss = df["val/cls_loss"].mean()
        val_dflloss = df["val/dfl_loss"].mean()
        lr_pg0 = df["lr/pg0"].mean()
        lr_pg1 = df["lr/pg1"].mean()
        lr_pg2 = df["lr/pg2"].mean()

        mlflow.log_metric("avg_metric-precision", precision)
        mlflow.log_metric("avg_metric-recall", recall)
        mlflow.log_metric("avg_metric-mAP50", mAP50)
        mlflow.log_metric("avg_metric-mAP50-95", mAP50_95)
        mlflow.log_metric("avg_train-boxloss", train_boxloss)
        mlflow.log_metric("avg_train-clsloss", train_clsloss)
        mlflow.log_metric("avg_train-dflloss", train_dflloss)
        mlflow.log_metric("avg_val-boxloss", val_boxloss)
        mlflow.log_metric("avg_val-clsloss", val_clsloss)
        mlflow.log_metric("avg_val-dflloss", val_dflloss)
        mlflow.log_metric("avg_lr-pg0", lr_pg0)
        mlflow.log_metric("avg_lr-pg1", lr_pg1)
        mlflow.log_metric("avg_lr-pg2", lr_pg2)
        
        # Export the model to ONNX format
        model.export(format="onnx")
        onnx_model_path = train_weight_folder / "best.onnx"
        # Log the model as artifact
        mlflow.log_artifact(str(onnx_model_path))
        # Load the ONNX model
        onnx_model = onnx.load(str(onnx_model_path))

        # Log the ONNX model with MLflow
        mlflow.onnx.log_model(onnx_model, "yolo_model")
        
        # Get the run ID for later reference
        run_id = run.info.run_id
        
        # Register the model
        model_name = "Yolo_Model_ONNX"
        result = mlflow.register_model(
            f"runs:/{run_id}/yolo_model",
            model_name
        )
        
        client = MlflowClient()
        while client.get_model_version(model_name, result.version).status != "READY":
            time.sleep(5)
        
        print(f"Model registered and ready: {model_name} version {result.version}")

if __name__ == "__main__":
    cfg_file = Path("data.yaml")
    args_file = Path("runs\\detect\\train\\args.yaml")

    train_yolov8(cfg_file)
    print("YOLOv8 model training complete!")
