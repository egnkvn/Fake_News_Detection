import google.generativeai as genai
import prompts
import json
import os
import time # Import the time module for sleep
from google.api_core import exceptions # Import exceptions for API errors
from tqdm import tqdm
from google.generativeai.types import HarmCategory, HarmBlockThreshold # 導入 HarmCategory 和 HarmBlockThreshold



genai.configure(api_key="AIzaSyDx6djjovqSry-kA8c3_71JWkWzyUcbo4g")
model = genai.GenerativeModel('gemini-2.0-flash') 

generation_config = {
    "candidate_count": 1,  # 通常我們只需要一個最佳答案
    "max_output_tokens": 800, # 限制最大輸出 token 數，避免過長解釋
    "temperature": 0.0,    # 控制輸出隨機性，0.0 為最不隨機（更確定性），1.0 為最隨機
    "top_p": 0.8,          # 控制取樣時包含的機率質量，結合 top_k 使用
    "top_k": 40            # 控制取樣時考慮的最高機率 token 數量
}

safety_settings = [
    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    # 對於新的 Gemini 模型 (如 1.5 系列)，可能還有其他類別，例如 CIVIC_INTEGRITY, UNSPECIFIED 等
]


def call_gemini_api(data_entry, max_retries=5, initial_delay=1):
    """
    呼叫 Gemini API 進行假新聞判斷。

    Args:
        data_entry (dict): 包含 'original_news', 'fake_news', 'type' 的字典。

    Returns:
        dict: 包含模型輸出結果和原始輸入的字典。
    """
    # 替換 Prompt 中的佔位符
    formatted_prompt = prompts.DATA_VERIFICATION_PROMPT.format(
        original_title=data_entry['original_title'],
        original_news=data_entry['original_content'],
        fake_title=data_entry['fake_title'],
        fake_news=data_entry['fake_content'],
        type=data_entry['category']
    )

    retries = 0
    delay = initial_delay

    while retries < max_retries:
        try:
            response = model.generate_content(
                formatted_prompt,
                generation_config=generation_config,
                safety_settings=safety_settings,
            )

            response_text = response.text.strip()

            if "輸出：" in response_text and "解釋：" in response_text:
                lines = response_text.split('\n')
                output_line = next((line for line in lines if line.startswith("輸出：")), "")
                explanation_line = next((line for line in lines if line.startswith("解釋：")), "")

                is_match_str = output_line.replace("輸出：", "").strip()
                explanation = explanation_line.replace("解釋：", "").strip()

                return {
                    "original_title": data_entry['original_title'],
                    "original_content": data_entry['original_content'],
                    "fake_title": data_entry['fake_title'],
                    "fake_content": data_entry['fake_content'],
                    "url": data_entry['url'],
                    "date": data_entry['date'],
                    "author": data_entry['author'],
                    "id": data_entry['id'],
                    "category": data_entry['category'],
                    "is_match": is_match_str.lower() == "true",
                    "explanation": explanation,
                    "raw_gemini_output": response_text
                }
            else:
                return {
                    "original_title": data_entry['original_title'],
                    "original_content": data_entry['original_content'],
                    "fake_title": data_entry['fake_title'],
                    "fake_content": data_entry['fake_content'],
                    "url": data_entry['url'],
                    "date": data_entry['date'],
                    "author": data_entry['author'],
                    "id": data_entry['id'],
                    "category": data_entry['category'],
                    "is_match": None,
                    "explanation": "無法解析 Gemini 輸出格式",
                    "raw_gemini_output": response_text
                }

        except exceptions.ResourceExhausted as e: # Catch Rate Limit Error (429)
            retries += 1
            print(f"遇到 Rate Limit 錯誤 (429)。第 {retries}/{max_retries} 次重試，等待 {delay} 秒...")
            time.sleep(delay)
            delay *= 2 # Exponential backoff
        except Exception as e:
            print(f"呼叫 Gemini API 時發生非 Rate Limit 錯誤: {e}")
            return {
                "original_title": data_entry['original_title'],
                "original_content": data_entry['original_content'],
                "fake_title": data_entry['fake_title'],
                "fake_content": data_entry['fake_content'],
                "url": data_entry['url'],
                "date": data_entry['date'],
                "author": data_entry['author'],
                "id": data_entry['id'],
                "category": data_entry['category'],
                "is_match": None,
                "explanation": f"API 呼叫失敗 (非 Rate Limit 錯誤): {e}",
                "raw_gemini_output": ""
            }
    
    # If all retries fail
    print(f"超過最大重試次數 ({max_retries})，無法完成 API 呼叫。")
    return {
        "original_title": data_entry['original_title'],
        "original_content": data_entry['original_content'],
        "fake_title": data_entry['fake_title'],
        "fake_content": data_entry['fake_content'],
        "url": data_entry['url'],
        "date": data_entry['date'],
        "author": data_entry['author'],
        "id": data_entry['id'],
        "category": data_entry['category'],
        "is_match": None,
        "explanation": f"API 呼叫失敗: 超過最大重試次數。",
        "raw_gemini_output": ""
    }

def process_data(real_news_folder, fake_news_folder, debug=False):
    
    all_real_news = []
    for file in os.listdir(real_news_folder):
        if file.endswith('.json'):
            with open(os.path.join(real_news_folder, file), 'r', encoding='utf-8') as f:
                all_real_news.extend(json.load(f))
    
    all_fake_news = []
    for file in os.listdir(fake_news_folder):
        if file.endswith('.json'):
            with open(os.path.join(fake_news_folder, file), 'r', encoding='utf-8') as f:
                all_fake_news.extend(json.load(f))
    
    if debug:
        all_fake_news = all_fake_news[:10]  # 只處理前10條假新聞

    data_entries = []
    for fake_news_entry in all_fake_news:
        for real_news_entry in all_real_news:
            if real_news_entry['id'] == fake_news_entry['id']:
                # 找到對應的真實新聞
                break
        else:
            print(f"警告：在真實新聞中找不到 ID 為 {fake_news_entry['id']} 的條目，將跳過此假新聞。")
            continue
        data_entry = {
            "original_title": real_news_entry['title'],
            "original_content": real_news_entry['content'],
            "fake_title": fake_news_entry['title'],
            "fake_content": fake_news_entry['content'],
            "url": real_news_entry['url'],
            "date": real_news_entry['date'],
            "author": real_news_entry['author'],
            "id": real_news_entry['id'],
            "category": fake_news_entry['category']
        }
        data_entries.append(data_entry)

    results = []
    for i, data_entry in enumerate(tqdm(data_entries)):
        result = call_gemini_api(data_entry, max_retries=5, initial_delay=1)
        results.append(result)

    try:
        output_json_path = os.path.join(fake_news_folder, 'verification.json')
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"\n所有處理結果已成功寫入 {output_json_path}")
    except Exception as e:
        print(f"錯誤：寫入輸出檔案 {output_json_path} 時發生問題: {e}")

# --- 主程式執行區塊 ---
if __name__ == "__main__":

    process_data(fake_news_folder='fake_data/train/impostor',
                 real_news_folder='new_data',
                 debug=True)  # 設定 debug=True 以僅處理前10條假新聞