"""Anomalib Inferencer Script.

This script performs inference by reading a model config file from
command line, and show the visualization results.
"""

from importlib import import_module
from pathlib import Path

import cv2
import gdown
import numpy as np

from anomalib.config import get_configurable_parameters
from anomalib.deploy.inferencers.base_inferencer import Inferencer
from anomalib.post_processing import superimpose_anomaly_map


def download_model(name):
    if name == "PaDiM":
        model_config_path = "models/padim/config.yaml"
        model_url = "https://drive.google.com/uc?id=1oIZ4-5GVAMaWALqj7fQFfX20U9O6h_Ao"
        weight_path = "models/padim/padim_model.ckpt"

    # Download model from Google Drive into model/ folder
    model_path = Path(weight_path)
    if not model_path.is_file():
        print("Model weights not detected. Downloading weights file ...")
        gdown.download(model_url, weight_path, quiet=False)
        print("Download complete.")

    model_config = {
        "model_config_path": model_config_path,
        "weight_path": weight_path,
    }

    return model_config


def load_model(name="PaDiM"):
    args = download_model(name)
    # args = model_config[name]
    config = get_configurable_parameters(
        model_name=name, config_path=args["model_config_path"]
    )
    # Get the inferencer. We use .ckpt extension for Torch models and (onnx, bin)
    # for the openvino models.
    extension = Path(args["weight_path"]).suffix
    inference: Inferencer
    if extension in (".ckpt"):
        module = import_module("anomalib.deploy.inferencers.torch_inferencer")
        TorchInferencer = getattr(
            module, "TorchInferencer"
        )  # pylint: disable=invalid-name
        inference = TorchInferencer(
            config=config,
            model_source=args["weight_path"],
            meta_data_path=None,  # args["meta_data"]
        )
    # elif extension in (".onnx", ".bin", ".xml"):
    #     module = import_module(".anomalib.deploy.inferencers.openvino")
    #     OpenVINOInferencer = getattr(module, "OpenVINOInferencer")  # pylint: disable=invalid-name
    #     inference = OpenVINOInferencer(config=config, path=args["weight_path"], meta_data_path=args.meta_data)

    else:
        raise ValueError(
            f"Model extension is not supported. Torch Inferencer exptects a .ckpt file,"
            f"OpenVINO Inferencer expects either .onnx, .bin or .xml file. Got {extension}"
        )
    return args, inference


def add_label(
    prediction: np.ndarray, scores: float, font: int = cv2.FONT_HERSHEY_PLAIN
) -> np.ndarray:
    """If the model outputs score, it adds the score to the output image.

    Args:
        prediction (np.ndarray): Resized anomaly map.
        scores (float): Confidence score.

    Returns:
        np.ndarray: Image with score text.
    """
    text = f"Confidence Score {scores:.0%}"
    font_size = (
        prediction.shape[1] // 1024 + 1
    )  # Text scale is calculated based on the reference size of 1024

    (width, height), baseline = cv2.getTextSize(
        text, font, font_size, thickness=font_size // 2
    )

    label_patch = np.zeros((height + baseline, width + baseline, 3), dtype=np.uint8)
    label_patch[:, :] = (225, 252, 134)

    cv2.putText(label_patch, text, (0, baseline // 2 + height), font, font_size, 0)

    prediction[: baseline + height, : baseline + width] = label_patch

    return prediction


def infer(model, image):
    """Perform inference on an input image."""
    # Perform inference for the given image or image path. if image
    # path is provided, `predict` method will read the image from
    # file for convenience.
    predictions = model.predict(image=image)

    anomaly_map = predictions.anomaly_map
    heat_map = superimpose_anomaly_map(
        anomaly_map=anomaly_map, image=predictions.image, normalize=True
    )

    score = predictions.pred_score
    output = add_label(heat_map, score)
    output = cv2.cvtColor(heat_map, cv2.COLOR_RGB2BGR)

    return output, score
