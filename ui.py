# from flask import Flask, request, render_template, redirect, render_template_string
# from werkzeug.utils import secure_filename
# from PIL import Image
# import io
# import os
# from ultralytics import YOLO
# import glob
# import shutil
# from mlflow.tracking import MlflowClient
# import mlflow
# from datetime import datetime
# import onnx
# import glob

# app = Flask(__name__)

# # Define allowed file extensions
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif','mp4'}

# # Initialize YOLO model
# #model = YOLO('mlflow_runs/6c41b67c54084659883c77933061b2a0/yolo_model.onnx')

# def get_latest_yolo_model(base_folder='mlflow_runs'):
#     all_folders = glob.glob(os.path.join(base_folder, '*'))
#     # Sort folders by last modified time
#     sorted_folders = sorted(all_folders, key=os.path.getmtime, reverse=True)
#     for folder in sorted_folders:
#         yolo_model_path = os.path.join(folder, 'yolo_model.onnx')
#         if os.path.exists(yolo_model_path):
#             return yolo_model_path
#     return None

# # Get the path of the latest YOLO model
# latest_yolo_model_path = get_latest_yolo_model()

# if latest_yolo_model_path:
#     # Initialize YOLO model with the latest yolo_model.onnx file
#     model = YOLO(latest_yolo_model_path)
#     print("YOLO model initialized successfully from run ID: ",latest_yolo_model_path)
# else:
#     print("No YOLO model found.")

# # Function to check if file extension is allowed
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # Function to get the path of the latest predicted image
# def get_latest_predict_image(base_folder='runs/detect'):
#     predict_folders = glob.glob(os.path.join(base_folder, 'predict*'))
#     if not predict_folders:
#         return None
#     latest_folder = max(predict_folders, key=os.path.getmtime)
#     print("Latest Predict Folder:", latest_folder)
#     latest_image = os.path.join(latest_folder, 'upload.jpg')
#     return latest_image

# @app.route('/', methods=['GET', 'POST'])
# def process_image():
#     if request.method == 'POST':
#         # Check if the post request has the file part
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files['file']
#         # If the user does not select a file, the browser submits an empty file without a filename
#         if file.filename == '':
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             # Perform object detection
#             img_bytes = file.read()
#             img = Image.open(io.BytesIO(img_bytes))
#             img_path = os.path.join('static', 'upload.jpg')
#             img.save(img_path)
#             # Perform object detection and save results
#             results = model(source=img_path, show=True, conf=0.3, save=True)
#             # Copy the latest predicted image to the static folder
#             latest_image = get_latest_predict_image()
#             if latest_image:
#                 shutil.copy(latest_image, 'static')
#                 return render_template('predict.html')
#     # If no image is uploaded or if processing fails, render the upload.html template
#     return render_template('upload.html')

# # Set the tracking URI to the local MLflow server
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# @app.route('/model')
# def model_index():
#     # Initialize the MLflow client
#     client = MlflowClient()

#     # Define the experiment ID for the default experiment
#     experiment_id = "0"  # Experiment ID 0 corresponds to the default experiment

#     # Fetch all runs for the default experiment
#     runs = client.search_runs(
#         experiment_ids=[experiment_id],
#         filter_string="",
#         run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
#         max_results=100,
#         order_by=["end_time DESC"]
#     )

#     # Find the last successful run
#     last_successful_run_id = None
#     for run in runs:
#         if run.info.status == "FINISHED":
#             last_successful_run_id = run.info.run_id
#             break

#     # If no successful runs found, raise an exception
#     if last_successful_run_id is None:
#         return "No successful runs found."

#     # Construct the logged model URI
#     logged_model_uri = f"runs:/{last_successful_run_id}/yolo_model"

#     # Extract the run ID from the logged model URI
#     run_id = last_successful_run_id

#     onnx_model = mlflow.onnx.load_model(logged_model_uri)

#     # Print the input node names
#     input_names = [input.name for input in onnx_model.graph.input]

#     # Get the model's format
#     model_format = mlflow.models.Model.load(logged_model_uri).flavors.get(mlflow.pyfunc.FLAVOR_NAME)

#     # Create the folder structure
#     base_folder = 'mlflow_runs'
#     sub_folder = os.path.join(base_folder, run_id)
#     os.makedirs(sub_folder, exist_ok=True)

#     # Define the local path where you want to save the ONNX model
#     local_model_path = os.path.join(sub_folder, 'yolo_model.onnx')

#     # Save the ONNX model locally
#     onnx.save_model(onnx_model, local_model_path)

#     # Save the model format and input names in a text file
#     info_file_path = os.path.join(sub_folder, 'model_info.txt')
#     with open(info_file_path, 'w') as f:
#         f.write(f"Model Format: {model_format}\n")
#         f.write(f"Input Names: {input_names}\n")


#     # Return the model information and inference results to HTML template
#     return render_template('model_index.html', input_names=input_names, model_format=model_format)



# @app.route('/experiment')
# def experiment():
#     # Initialize the MLflow client
#     client = MlflowClient()

#     # Define the experiment ID for the default experiment
#     experiment_id = "0"  # Experiment ID 0 corresponds to the default experiment

#     # Fetch all runs for the default experiment
#     runs = client.search_runs(
#         experiment_ids=[experiment_id],
#         filter_string="",
#         run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
#         max_results=100,
#         order_by=["end_time DESC"]
#     )

#     # Find the last successful run
#     last_successful_run_id = None
#     last_successful_run_name = None
#     last_successful_run_time = None
#     for run in runs:
#         if run.info.status == "FINISHED":
#             last_successful_run_id = run.info.run_id
#             last_successful_run_name = run.data.tags.get("mlflow.runName")  # Fetch run name if tagged
#             last_successful_run_start_time = datetime.fromtimestamp(run.info.start_time / 1000.0)  # Convert to readable format
#             last_successful_run_end_time = datetime.fromtimestamp(run.info.end_time / 1000.0)  # Convert to readable format
#             last_successful_run_duration = run.info.end_time - run.info.start_time  # Compute duration
#             last_successful_run_status = run.info.status  # Fetch run status
#             last_successful_run_lifecycle_stage = run.info.lifecycle_stage
#             last_successful_run_user_id = run.info.user_id
#             last_successful_run_artifact = run.info.artifact_uri

#             break

#     # Prepare data to pass to HTML template
#     data = {
#         "last_successful_run_id": last_successful_run_id,
#         "last_successful_run_name": last_successful_run_name,
#         "last_successful_run_start_time": last_successful_run_start_time,
#         "last_successful_run_end_time": last_successful_run_end_time,
#         "last_successful_run_duration": last_successful_run_duration,
#         "last_successful_run_status": last_successful_run_status,
#         "last_successful_run_lifecycle_stage": last_successful_run_lifecycle_stage,
#         "last_successful_run_user_id": last_successful_run_user_id,
#         "last_successful_run_artifact": last_successful_run_artifact
#     }

#     # Render HTML template with data
#     return render_template('experiment.html', data=data)


# if __name__ == '__main__':
#     app.run(debug=True, port=5001)




from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import io
import os
from ultralytics import YOLO
import glob
import shutil
from mlflow.tracking import MlflowClient
import mlflow
from datetime import datetime
import onnx

app = Flask(__name__)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to get the latest model path
def get_latest_model_path(base_folder='mlflow_runs'):
    folders = [os.path.join(base_folder, d) for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
    latest_folder = max(folders, key=os.path.getmtime)
    model_path = os.path.join(latest_folder, 'yolo_model.onnx')
    return model_path

# Initialize YOLO model
model_path = get_latest_model_path()
model = YOLO(model_path)

# Function to get the path of the latest predicted image
def get_latest_predict_image(base_folder='runs/detect'):
    predict_folders = glob.glob(os.path.join(base_folder, 'predict*'))
    if not predict_folders:
        return None
    latest_folder = max(predict_folders, key=os.path.getmtime)
    print("Latest Predict Folder:", latest_folder)
    latest_image = os.path.join(latest_folder, 'upload.jpg')
    return latest_image

@app.route('/', methods=['GET', 'POST'])
def process_media():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_ext = filename.rsplit('.', 1)[1].lower()
            if file_ext in {'png', 'jpg', 'jpeg', 'gif'}:
                # Process image file
                img_bytes = file.read()
                img = Image.open(io.BytesIO(img_bytes))
                img_path = os.path.join('static', 'upload.jpg')
                img.save(img_path)
                results = model(source=img_path, show=True, conf=0.3, save=True)
                latest_image = get_latest_predict_image()
                if latest_image:
                    shutil.copy(latest_image, 'static')
                return render_template('predict.html', is_image=True, is_video=False)
            elif file_ext in {'mp4', 'avi', 'mov'}:
                # Process video file
                video_path = os.path.join('static', 'upload.mp4')
                file.save(video_path)
                results = model(source=video_path, show=True, conf=0.3, save=True)
                return render_template('predict.html', is_image=False, is_video=True, video_filename='upload.' + file_ext, video_extension=file_ext)
    return render_template('upload.html')

# Set the tracking URI to the local MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

@app.route('/model')
def model_index():
    # Initialize the MLflow client
    client = MlflowClient()

    # Define the experiment ID for the default experiment
    experiment_id = "0"  # Experiment ID 0 corresponds to the default experiment

    # Fetch all runs for the default experiment
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="",
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        max_results=100,
        order_by=["end_time DESC"]
    )

    # Find the last successful run
    last_successful_run_id = None
    for run in runs:
        if run.info.status == "FINISHED":
            last_successful_run_id = run.info.run_id
            break

    # If no successful runs found, raise an exception
    if last_successful_run_id is None:
        return "No successful runs found."

    # Construct the logged model URI
    logged_model_uri = f"runs:/{last_successful_run_id}/yolo_model"

    # Extract the run ID from the logged model URI
    run_id = last_successful_run_id

    onnx_model = mlflow.onnx.load_model(logged_model_uri)

    # Print the input node names
    input_names = [input.name for input in onnx_model.graph.input]

    # Get the model's format
    model_format = mlflow.models.Model.load(logged_model_uri).flavors.get(mlflow.pyfunc.FLAVOR_NAME)

    # Create the folder structure
    base_folder = 'mlflow_runs'
    sub_folder = os.path.join(base_folder, run_id)
    os.makedirs(sub_folder, exist_ok=True)

    # Define the local path where you want to save the ONNX model
    local_model_path = os.path.join(sub_folder, 'yolo_model.onnx')

    # Save the ONNX model locally
    onnx.save_model(onnx_model, local_model_path)

    # Save the model format and input names in a text file
    info_file_path = os.path.join(sub_folder, 'model_info.txt')
    with open(info_file_path, 'w') as f:
        f.write(f"Model Format: {model_format}\n")
        f.write(f"Input Names: {input_names}\n")

    # Return the model information and inference results to HTML template
    return render_template('model_index.html', input_names=input_names, model_format=model_format)

@app.route('/experiment')
def experiment():
    # Initialize the MLflow client
    client = MlflowClient()

    # Define the experiment ID for the default experiment
    experiment_id = "0"  # Experiment ID 0 corresponds to the default experiment

    # Fetch all runs for the default experiment
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="",
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        max_results=100,
        order_by=["end_time DESC"]
    )

    # Find the last successful run
    last_successful_run_id = None
    last_successful_run_name = None
    last_successful_run_time = None
    for run in runs:
        if run.info.status == "FINISHED":
            last_successful_run_id = run.info.run_id
            last_successful_run_name = run.data.tags.get("mlflow.runName")  # Fetch run name if tagged
            last_successful_run_start_time = datetime.fromtimestamp(run.info.start_time / 1000.0)  # Convert to readable format
            last_successful_run_end_time = datetime.fromtimestamp(run.info.end_time / 1000.0)  # Convert to readable format
            last_successful_run_duration = run.info.end_time - run.info.start_time  # Compute duration
            last_successful_run_status = run.info.status  # Fetch run status
            last_successful_run_lifecycle_stage = run.info.lifecycle_stage
            last_successful_run_user_id = run.info.user_id
            last_successful_run_artifact = run.info.artifact_uri

            break

    # Prepare data to pass to HTML template
    data = {
        "last_successful_run_id": last_successful_run_id,
        "last_successful_run_name": last_successful_run_name,
        "last_successful_run_start_time": last_successful_run_start_time,
        "last_successful_run_end_time": last_successful_run_end_time,
        "last_successful_run_duration": last_successful_run_duration,
        "last_successful_run_status": last_successful_run_status,
        "last_successful_run_lifecycle_stage": last_successful_run_lifecycle_stage,
        "last_successful_run_user_id": last_successful_run_user_id,
        "last_successful_run_artifact": last_successful_run_artifact
    }

    # Render HTML template with data
    return render_template('experiment.html', data=data)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
