from os import environ
import base64
import numpy as np
import torch
import logging
import requests

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

rollup_server = environ["ROLLUP_HTTP_SERVER_URL"]
logger.info(f"HTTP rollup_server url is {rollup_server}")

seed = 42
torch.manual_seed(seed)

# Load the saved TorchScript model
loaded_model = torch.jit.load('simple_nn_model.pth')

# Set the model to evaluation mode (optional but recommended)
loaded_model.eval()
logger.info("Model loaded properly")


def str2hex(str):
    """
    Encodes a string as a hex string
    """
    return "0x" + str.encode("utf-8").hex()

def hex2str(hex):
    """
    Decodes a hex string into a regular string
    """
    return bytes.fromhex(hex[2:]).decode("utf-8")

def handle_advance(data):
    logger.info(f"Received advance request data {data}")
    

    # Dummy input for inference (same shape as used for training)
    test_inputs = torch.randn(10, 10)  # 10 samples, 10 features
    logger.info(test_inputs)
    # Perform inference
    with torch.no_grad():
        outputs = loaded_model(test_inputs)

    # Get the predicted class (assuming a classification problem)
    predicted_classes = torch.argmax(outputs, dim=1)
    logger.info(f"Predicted classes: {predicted_classes}")

    try:
        response = requests.post(
                    rollup_server + "/notice", json={"payload": str2hex(str({"modelOutputs": str(outputs[0][0])}))}
        )
        logger.info(
            f"Received notice status {response.status_code} body {response.content}"
        )
    except Exception as e:
        logger.error(f"Exception while handling advance:{e}")

    return "accept"


def handle_inspect(data):
    logger.info(f"Received inspect request data {data}")
    return "accept"


handlers = {
    "advance_state": handle_advance,
    "inspect_state": handle_inspect,
}

finish = {"status": "accept"}

while True:
    logger.info("Sending finish")
    response = requests.post(rollup_server + "/finish", json=finish)
    logger.info(f"Received finish status {response.status_code}")
    if response.status_code == 202:
        logger.info("No pending rollup request, trying again")
    else:
        rollup_request = response.json()
        data = rollup_request["data"]
        handler = handlers[rollup_request["request_type"]]
        finish["status"] = handler(rollup_request["data"])
