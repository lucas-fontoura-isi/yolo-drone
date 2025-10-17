import yaml
from pathlib import Path
from ultralytics import YOLO

# Load configuration file
CONFIG_PATH = Path(__file__).parent / "config_files" / "train_config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


def train_yolo(data: Path) -> None:
    model = YOLO(config["model"])

    results = model.train(
        data=data,
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        device=config["device"],
        batch=config["batch"],
        seed=config["seed"],
        project=config["project"],
        name=config["name"]
    )

    print(results)