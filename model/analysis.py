import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def load_jsonl_labels(file_path: str, true_label: str):
    y_true_list = []
    y_pred_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pred = obj.get("label", None)
            if pred is None:
                continue
            pred = pred.lower()
            y_pred_list.append(pred)
            y_true_list.append(true_label.lower())
    return y_true_list, y_pred_list

def collect_labels(no_search_files: list, search_files: list):
    results = {}
    for group_name, file_list in [("no_search", no_search_files), ("search", search_files)]:
        y_true_all = []
        y_pred_all = []
        for path in file_list:
            base = os.path.basename(path)
            if "_real" in base:
                true_label = "real"
            else:
                true_label = "fake"
            t, p = load_jsonl_labels(path, true_label)
            y_true_all.extend(t)
            y_pred_all.extend(p)
        results[group_name] = (y_true_all, y_pred_all)
    return results

def compute_metrics_per_class(y_true: list, y_pred: list):
    """
    計算 Accuracy、F1(real)、F1(fake)。
    """
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=["real", "fake"],
        average=None
    )
    f1_real = float(f[0])
    f1_fake = float(f[1])
    return acc, f1_real, f1_fake

def plot_metrics_group_by_metric(metrics_dict: dict):
    metrics = ["Accuracy", "F1_real", "F1_fake"]
    # No Search / Search 各自的數值
    no_search_vals = [metrics_dict["no_search"][0],
                      metrics_dict["no_search"][1],
                      metrics_dict["no_search"][2]]
    search_vals    = [metrics_dict["search"][0],
                      metrics_dict["search"][1],
                      metrics_dict["search"][2]]

    x = range(len(metrics))
    width = 0.35  # 長條寬度

    # 和雷達圖相同的色碼
    color_ns = "#1f77b4"  # 藍色 (No Search)
    color_s  = "#ff7f0e"  # 橘色 (Search)
    alpha_ns = 0.4        # 藍色半透明
    alpha_s  = 0.4       # 橘色較不透明

    plt.figure(figsize=(8, 5))
    # No Search
    plt.bar(
        [i - width/2 for i in x],
        no_search_vals,
        width=width,
        label="No Search",
        color=color_ns,
        alpha=alpha_ns,
        edgecolor="black",
        linewidth=0.5
    )
    # Search
    plt.bar(
        [i + width/2 for i in x],
        search_vals,
        width=width,
        label="Search",
        color=color_s,
        alpha=alpha_s,
        edgecolor="black",
        linewidth=0.5
    )

    plt.xticks(x, metrics, fontsize=12)
    plt.ylim(0, 1.0)
    plt.ylabel("Score", fontsize=12)
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig('analysis.png', dpi=300)

if __name__ == "__main__":
    # 請自行修改成真實路徑
    no_search_files = [
        "log/logs_no_search_real.jsonl",
        "log/logs_no_search_fake.jsonl"
    ]
    search_files = [
        "log/logs_search_real.jsonl",
        "log/logs_search_fake.jsonl"
    ]

    # 1. 收集 y_true 與 y_pred
    data_dict = collect_labels(no_search_files, search_files)

    # 2. 計算兩組的 (Accuracy, F1_real, F1_fake)
    metrics_dict = {}
    for group in ["no_search", "search"]:
        y_true, y_pred = data_dict[group]
        acc, f1_real, f1_fake = compute_metrics_per_class(y_true, y_pred)
        metrics_dict[group] = (acc, f1_real, f1_fake)

    # 3. 印出數值
    for group in ["no_search", "search"]:
        acc, f1_r, f1_f = metrics_dict[group]
        print(f"--- {group} 結果 ---")
        print(f"Accuracy = {acc:.4f}")
        print(f"F1 (real) = {f1_r:.4f}")
        print(f"F1 (fake) = {f1_f:.4f}\n")

    # 4. 繪圖
    plot_metrics_group_by_metric(metrics_dict)