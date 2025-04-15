import json
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_result_json_files(dir_path):
    trainer_state_path = os.path.join(dir_path, "trainer_state.json")

    # Kiểm tra nếu trainer_state.json đã nằm trực tiếp trong dir_path
    if os.path.exists(trainer_state_path):
        return trainer_state_path

    # Đường dẫn đến thư mục chứa các checkpoint
    base_dir = dir_path

    # Tìm checkpoint mới nhất
    checkpoints = [f for f in os.listdir(base_dir) if f.startswith("checkpoint-")]
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))

    # Đường dẫn đến file trainer_state.json trong checkpoint mới nhất
    trainer_state_path = os.path.join(base_dir, latest_checkpoint, "trainer_state.json")
    return trainer_state_path

def plot_image(dir_path, output_file):
    input_file = get_result_json_files(dir_path)
    # Đọc file JSON
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Trích xuất thông tin từ log_history
    log_history = data["log_history"]

    # Lọc thông tin liên quan đến train và eval
    train_data = []
    eval_data = []

    for entry in log_history:
        if "loss" in entry and "epoch" in entry:
            train_data.append({
                "epoch": entry["epoch"],
                "loss": entry["loss"]
            })
        if "eval_loss" in entry and "epoch" in entry:
            eval_data.append({
                "epoch": entry["epoch"],
                "eval_loss": entry["eval_loss"],
                "eval_accuracy": entry["eval_acc"]
            })

    # Tạo DataFrame
    train_df = pd.DataFrame(train_data)
    eval_df = pd.DataFrame(eval_data)

    # Plotting three subplots in one figure

    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    # Train Loss subplot
    axs[0].plot(train_df["epoch"], train_df["loss"], marker='o', color='red', label="Train Loss")
    axs[0].set_title("Train Loss over Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].grid(True)
    axs[0].legend()

    # Eval Loss subplot
    axs[1].plot(eval_df["epoch"], eval_df["eval_loss"], marker='o', color='blue', label="Eval Loss")
    axs[1].set_title("Evaluation Loss over Epochs")
    axs[1].set_ylabel("Eval Loss")
    axs[1].grid(True)
    axs[1].legend()

    # Eval Accuracy subplot
    axs[2].plot(eval_df["epoch"], eval_df["eval_accuracy"], marker='o', color='green', label="Eval Accuracy")
    axs[2].set_title("Evaluation Accuracy over Epochs")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Accuracy")
    axs[2].grid(True)
    axs[2].legend()

    # Adjust layout and save the image
    fig.tight_layout()
    plt.savefig(output_file)
