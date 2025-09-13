import os, subprocess, json
from pathlib import Path
from google.colab import drive

REPO_URL   = "https://github.com/Wangjx1995/Object_Detection_Tutorial.git"
REPO_DIR   = "/content/Object_Detection_Tutorial"
BRANCH     = "main"

DRIVE_MOUNT = "/content/drive"
DATA_ROOT   = "/content/drive/MyDrive/colab_data/odt/my_dataset"
CLASSES     = ["person", "car"]   
EPOCHS      = 10                  
TRAIN_CMD   = "yolo task=detect mode=train model=yolo11n.pt data={dataset_yaml} epochs={epochs} imgsz=640"



drive.mount(DRIVE_MOUNT)


subprocess.run('git config --global user.name "wjx"', shell=True, check=True)
subprocess.run('git config --global user.email "wangjx951029@gmail.com"', shell=True, check=True)
subprocess.run('git config --global --add safe.directory "*"', shell=True, check=True)


subprocess.run(f"rm -rf '{REPO_DIR}' && git clone -b {BRANCH} {REPO_URL} '{REPO_DIR}'",shell=True, check=True)


subprocess.run("python -m pip install -U pip", shell=True, check=True)
subprocess.run(f"python -m pip install -r {REPO_DIR}/requirements.txt", shell=True, check=True)


for p in [
    f"{DATA_ROOT}/images/train",
    f"{DATA_ROOT}/images/val",
    f"{DATA_ROOT}/labels/train",
    f"{DATA_ROOT}/labels/val",
]:
    Path(p).mkdir(parents=True, exist_ok=True)

dataset_yaml_path = f"{DATA_ROOT}/dataset.yaml"
dataset_yaml = {
    "path": DATA_ROOT,
    "train": "images/train",
    "val": "images/val",
    "names": CLASSES,
}
Path(dataset_yaml_path).write_text(
    "path: {}\ntrain: {}\nval: {}\nnames: {}\n".format(
        dataset_yaml["path"], dataset_yaml["train"], dataset_yaml["val"], json.dumps(dataset_yaml["names"], ensure_ascii=False)
    ),
    encoding="utf-8"
)


project_link = f"{REPO_DIR}/data/dataset"
subprocess.run(f"rm -rf {project_link}", shell=True, check=False) 
os.symlink(DATA_ROOT, project_link, target_is_directory=True)


cmd = TRAIN_CMD.format(dataset_yaml=dataset_yaml_path, epochs=EPOCHS)
print("\n=== RUN ===\n", cmd)
subprocess.run(cmd, shell=True, check=True, cwd=REPO_DIR)

print("\n✅ Done.")