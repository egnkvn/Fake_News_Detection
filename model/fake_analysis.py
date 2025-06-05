import json
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def load_jsonl_preds(file_path: str, true_label: str, 
                     per_cat_correct: dict, per_cat_total: dict):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pred = obj.get("label", None)
            if pred is None:
                continue
            pred = pred.lower().strip()

            cat = obj.get("category", None)
            if cat is None:
                cat = "unknown"

            per_cat_total[cat] += 1
            if pred == true_label:
                per_cat_correct[cat] += 1


def compute_search_accuracy(search_files: list):
    s_corr  = defaultdict(int)
    s_total = defaultdict(int)

    # 處理 Search 檔案
    for path in search_files:
        basename = os.path.basename(path).lower()
        if "_real" in basename:
            t_label = "real"
        else:
            t_label = "fake"
        load_jsonl_preds(path, t_label, s_corr, s_total)

    # 收集所有出現過的 category
    cats = sorted(s_total.keys())

    # 計算每個 category 的 Search accuracy
    acc_search = []
    for c in cats:
        tot = s_total[c]
        corr = s_corr[c]
        acc_s = corr / tot if tot > 0 else 0.0
        acc_search.append(acc_s)

    return cats, acc_search


def plot_radar_polygon(cats, acc_user, acc_search, save_path="radar_search_vs_manual.png"):
    N = len(cats)
    # 雷達圖需要在最後「再把第一個值 append 到陣列尾端」，以形成閉合形狀
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 閉合

    # 閉合資料
    vals_user   = [acc_user[c] for c in cats]      + [acc_user[cats[0]]]
    vals_search = acc_search + [acc_search[0]]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    # 讓第一個 category 從頂端開始 (π/2)，並逆時針方向
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 設定雷達圖上的標籤 (去掉最後一個閉合標籤)
    ax.set_thetagrids(np.degrees(angles[:-1]), cats, fontsize=11)

    # 設定 radial 範圍 & 同心網格
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.5)

    # **畫多邊形外框**
    # ax.plot(angles, [1.0]*len(angles), color="black", linewidth=1.0)

    # 低飽和度配色
    # Manual (用者手動提供)
    ax.plot(angles, vals_user,   color="C2", linewidth=1.5, linestyle="solid", label="BERT")
    ax.fill(angles, vals_user,   color="C2", alpha=0.25)
    # Search (模型執行 Search 後的結果)
    ax.plot(angles, vals_search, color="C1", linewidth=1.5, linestyle="solid", label="Search Agent")
    ax.fill(angles, vals_search, color="C1", alpha=0.25)

    # 圖例放在右上角
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    # -----------------------------
    # 1. 請修改成你實際的 Search JSONL 檔路徑
    search_files = [
        # "log/logs_search_real.jsonl",
        "log/logs_search_fake.jsonl"
    ]
    # -----------------------------

    # 2. 先計算 Search 檔案各 category 的 Accuracy
    cats, acc_search = compute_search_accuracy(search_files)

    # 3. 手動提供的 Accuracy（從你提供的文字中整理而來）
    #    注意：keys 必須與 cats 中元素一致，若某 category 在 cats 裡找不到就給 0.0
    manual_acc = {
        "satire":               1.0000,
        "false_context":        0.9500,
        "impostor":             1.0000,
        "false_connection":     1.0000,
        "misleading&manipulated": 1.0000,
        "fabricated":           1.0000
    }
    # 確保若某些 category 在 manual_acc 沒有給值，預設為 0.0
    for c in cats:
        if c not in manual_acc:
            manual_acc[c] = 0.0

    # 4. 畫雷達圖 (多邊形外框) 並比較 Manual 與 Search
    plot_radar_polygon(cats, manual_acc, acc_search, save_path="radar_search_vs_bert.png")

    print("完成：已將雷達圖儲存為 radar_search_vs_manual.png")