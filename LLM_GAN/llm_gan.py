# llm_gan.py
import logging
from typing import Tuple
from tqdm import tqdm
from openai import OpenAI
import os
import json
import re
# Set the environment variable
os.environ["OPENAI_API_KEY"] = ""

client = OpenAI()

fake_news_types = {
    "satire": {
        "name": "諷刺或惡搞（Satire or Parody）",
        "definition": "以幽默或諷刺為目的創作的內容，無意誤導，但可能被誤解為真實新聞。",
        "note": "原意為娛樂，但易被誤解。"
    },
    "false_context": {
        "name": "錯誤上下文（False Context）",
        "definition": "真實內容被移植到錯誤的時空背景，導致誤導。",
        "note": "內容本身可能為真，但使用時的脈絡錯誤。"
    },
    "fabricated": {
        "name": "捏造內容（Fabricated Content）",
        "definition": "完全虛構的新聞或事件，與現實無關。",
        "note": "屬於惡意創作，目的是欺騙或操縱輿論。"
    },
    "false_connection":{
        "name": "錯誤連結（False Connection）",
        "definition": "標題、圖片或導語與實際內容不符。",
        "note": "吸引點擊，內容與標題不符。"
    },
    "misleading":{
        "name": "誤導內容（Misleading Content）",
        "definition": "利用真實資訊，以誤導性的方式呈現，造成錯誤解讀。",
        "note": "資訊真實，但呈現方式誤導。"
    },
    "impostor":{
        "name": "冒名內容（Impostor Content）",
        "definition": "冒用可信來源的名義，發布虛假資訊。",
        "note": "偽裝成可信來源，增加可信度。"
    },
    "manipulated":{
        "name": "操弄內容（Manipulated Content）",
        "definition": "對真實資訊進行修改或剪輯，改變其原意。",
        "note": "修改原始內容，誤導觀眾。"
    }

}

# ─── Logger Setup ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLM-GAN")

# ─── news extract ──────────────────────────────────────────────────────────────
def extract_news(text):
    match = re.match(r"標題：(.+?)\n?內容：(.+)", text, re.DOTALL)
    if match:
        title = match.group(1).strip()
        content = match.group(2).strip()
        result = {"title": title, "content": content}
        return result
    else:
        print("fake news格式不正確")
        input()

# ─── Prompt Templates ──────────────────────────────────────────────────────────
def generator_prompt(real_news: object, fake_news_type: str, fake_strategy: str = None) -> str:
    t = fake_news_types[fake_news_type]
    return f"""你是一位假新聞生成器。
根據以下真實新聞，創造一則看似合理但實際為虛假的新聞版本。

本次的造假類型為：{t['name']}
- 定義：{t['definition']}
- 重點：{t['note']}

真實新聞：
標題：{real_news['title']}
內容：{real_news['content']}

造假策略：{fake_strategy or "請加入與該類型對應的誤導技巧，例如幽默、斷章取義、捏造細節等"}

請以以下格式回覆：
<fake_news>
標題：...
內容：...
</fake_news>
<explanation>...（解釋你如何根據{t['name']}的特性進行造假）...</explanation>"""

def detector_prompt(news: object, detect_strategy: str = None) -> str:
    return f"""你是一位假新聞偵測器。
新聞資訊：
標題：{news['title']}
內容：{news['content']}
偵測策略：{detect_strategy or "使用常識與事實一致性進行判斷。"}
請以以下格式回覆：
<label>Real／Fake</label>
<explanation>...（解釋你做出判斷的依據）...</explanation>"""

def strategy_update_prompt(success: bool, news: object, explanation: str, role: str, previous_strategy = None, fake_news_type = None) -> str:
    if success and role == "Generator":
        t = fake_news_types[fake_news_type]
        return f"""你是一位假新聞策略設計師。
    偵測器已經識別出這則是假新聞。

    本次的假新聞類型為：{t['name']}
    - 定義：{t['definition']}
    - 重點：{t['note']}

    新聞資訊：
    標題：{news['title']}
    內容：{news['content']}

    偵測器的說明：{explanation}

    你在上一次使用的策略如下：
    <previous_strategy>{previous_strategy}</previous_strategy>

    請根據偵測器的說明與新聞類型，**修改上述策略**，讓它更難被識破，但仍保有原來該類型的特性）。

    請以以下格式回覆：
    <strategy>...（你的更新後造假策略）...</strategy>"""

    elif not success and role == "Detector":
        return f"""你是一位假新聞偵測策略專家。
偵測器未能識別出這則假新聞。
新聞資訊：
標題：{news['title']}
內容：{news['content']}
造假說明：{explanation}
請改進偵測策略，以提升識別假新聞的能力。
請以以下格式回覆：
<strategy>...（你提出的改進策略）...</strategy>"""
    else:
        return ""

# ─── Utility Functions ──────────────────────────────────────────────────────────
def call_gpt(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    # Extract token usage
    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens

    # Pricing (GPT-4.1-mini as of May 2025)
    input_cost_per_token = 0.0000004     # $0.40 / 1M
    output_cost_per_token = 0.0000016    # $1.60 / 1M

    # Calculate costs
    input_cost = prompt_tokens * input_cost_per_token
    output_cost = completion_tokens * output_cost_per_token
    total_cost = input_cost + output_cost
    # Print usage and cost
    print(f"Total tokens: {total_tokens}")
    print(f"Estimated cost: ${total_cost:.8f}")
    return response.choices[0].message.content

def extract_between_tags(text: str, tag: str) -> str:
    import re
    # Try to match both <tag>...</tag> and handle <tag> without a closing tag
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no closing tag, try to find what comes after <tag>
    match_open_only = re.search(f"<{tag}>(.*)", text, re.DOTALL)
    return match_open_only.group(1).strip() if match_open_only else ""

def extract_news_and_explanation(output: str) -> Tuple[str, str]:
    return extract_between_tags(output, "fake_news"), extract_between_tags(output, "explanation")

def extract_label_and_explanation(output: str) -> Tuple[str, str]:
    return extract_between_tags(output, "label"), extract_between_tags(output, "explanation")

def extract_strategy(output: str) -> str:
    return extract_between_tags(output, "strategy")

# ─── Main Training Loop ─────────────────────────────────────────────────────────
def run_llm_gan(real_news_samples, num_rounds=2):
    
    fake_news_samples = []

    fake_news_category = list(fake_news_types.keys())

    category_index = 0
    count = 0
    detect_strategy = ""

    while category_index != len(fake_news_category):
        fake_news_type = fake_news_category[category_index]
        real_news = real_news_samples[count]
        count += 1
        if count % 40 == 0:
            category_index += 1
    
        fake_strategy = ""
        # detect_strategy = ""
        for round in range(num_rounds):
            logger.info(f"\n=== Round {round + 1} ===")

            # 1. Generator creates fake news
            gen_output = call_gpt(generator_prompt(real_news, fake_news_type, fake_strategy))
            fake_news, fake_explanation = extract_news_and_explanation(gen_output)
            fake_news = extract_news(fake_news)
            logger.info(f"Generated Fake News: {fake_news}\nExplanation: {fake_explanation}")


            if fake_explanation == "":
                print("fake_explanation格式出現錯誤")
                input()

            # 2. Detector attempts classification
            det_output = call_gpt(detector_prompt(fake_news, detect_strategy))
            predicted_label, det_explanation = extract_label_and_explanation(det_output)

            if predicted_label == "":
                predicted_label = det_output.split('\n')[0].strip()
            
            if det_explanation == "":
                det_explanation = det_output.split('\n')[1].strip()

            if predicted_label == "":
                print("predicted_label格式出現錯誤")
                input() 
            
            if det_explanation == "":
                print("predicted_label格式出現錯誤")
                input()

            logger.info(f"Prediction: {predicted_label}\nExplanation: {det_explanation}")

            # Assume the ground truth is "Fake" for all generated news
            correct = predicted_label.lower() == "fake"

            # 3. Update strategy based on success
            if correct:
                logger.info("Detector succeeded. Updating Generator strategy...")
                strat_prompt = strategy_update_prompt(True, fake_news, det_explanation, role="Generator", previous_strategy = fake_strategy, fake_news_type = fake_news_type)
                fake_strategy = extract_strategy(call_gpt(strat_prompt))
                if fake_strategy == "":
                    print(f"fake_strategy格式錯誤")
                    input()
            else:
                logger.info("Detector failed. Updating Detector strategy...")
                strat_prompt = strategy_update_prompt(False, fake_news, fake_explanation, role="Detector")
                detect_strategy = extract_strategy(call_gpt(strat_prompt))
                if detect_strategy == "":
                    print(f"detect_strategy格式錯誤")
                    input()


            logger.info(f"Updated Fake Strategy: {fake_strategy}")
            logger.info(f"Updated Detect Strategy: {detect_strategy}")

        fake_news["url"] = real_news["url"]
        fake_news["date"] = real_news["date"]
        fake_news["author"] = real_news["author"]
        fake_news["id"] = real_news["id"]
        fake_news["category"] = fake_news_type
        fake_news_samples.append(fake_news)

        with open("../fake_data/test/generated_news.json", "w", encoding="utf-8") as f:
            json.dump(fake_news_samples, f, ensure_ascii=False, indent=4)

        with open(f"../fake_data/test/detect_strategy.txt", "w") as f:
            print(detect_strategy, file=f)

    logger.info("Training complete.")

def run_llm_gan_train_set(real_news_samples, num_rounds=2):
    
    fake_news_samples = []

    # fake_news_category = list(fake_news_types.keys())

    fake_news_category = ['fabricated', 'false_connection', 'misleading', 'impostor', 'manipulated']

    for fake_news_type in fake_news_category:
        if not os.path.exists(f"/data2/jerome/web_mining/final/Fake_News_Detection/fake_data/train/{fake_news_type}"):
            os.makedirs(f"/data2/jerome/web_mining/final/Fake_News_Detection/fake_data/train/{fake_news_type}")
            fake_news_samples = []
            exist_ids = set()
        else:
            with open(f'/data2/jerome/web_mining/final/Fake_News_Detection/fake_data/train/{fake_news_type}/generated_news.json', 'r', encoding='utf-8') as f:
                fake_news_samples = json.load(f)

            exist_ids = set([ fake_news_sample['id'] for fake_news_sample in fake_news_samples])

        detect_strategy = ""

        for real_news in tqdm(real_news_samples):
            if real_news['id'] in exist_ids:
                continue

            fake_strategy = ""
            # detect_strategy = ""
            for round in range(num_rounds):
                logger.info(f"\n=== Round {round + 1} ===")

                # 1. Generator creates fake news
                gen_output = call_gpt(generator_prompt(real_news, fake_news_type, fake_strategy))
                fake_news, fake_explanation = extract_news_and_explanation(gen_output)
                fake_news = extract_news(fake_news)
                logger.info(f"Generated Fake News: {fake_news}\nExplanation: {fake_explanation}")


                if fake_explanation == "":
                    print("fake_explanation格式出現錯誤")
                    input()

                # 2. Detector attempts classification
                det_output = call_gpt(detector_prompt(fake_news, detect_strategy))
                predicted_label, det_explanation = extract_label_and_explanation(det_output)

                if predicted_label == "":
                    predicted_label = det_output.split('\n')[0].strip()
                
                if det_explanation == "":
                    det_explanation = det_output.split('\n')[1].strip()

                if predicted_label == "":
                    print("predicted_label格式出現錯誤")
                    input() 
                
                if det_explanation == "":
                    print("predicted_label格式出現錯誤")
                    input()

                logger.info(f"Prediction: {predicted_label}\nExplanation: {det_explanation}")

                # Assume the ground truth is "Fake" for all generated news
                correct = predicted_label.lower() == "fake"

                # 3. Update strategy based on success
                if correct:
                    logger.info("Detector succeeded. Updating Generator strategy...")
                    strat_prompt = strategy_update_prompt(True, fake_news, det_explanation, role="Generator", previous_strategy = fake_strategy, fake_news_type = fake_news_type)
                    fake_strategy = extract_strategy(call_gpt(strat_prompt))
                    if fake_strategy == "":
                        print(f"fake_strategy格式錯誤")
                        input()
                else:
                    logger.info("Detector failed. Updating Detector strategy...")
                    strat_prompt = strategy_update_prompt(False, fake_news, fake_explanation, role="Detector")
                    detect_strategy = extract_strategy(call_gpt(strat_prompt))
                    if detect_strategy == "":
                        print(f"detect_strategy格式錯誤")
                        input()

                logger.info(f"Updated Fake Strategy: {fake_strategy}")
                logger.info(f"Updated Detect Strategy: {detect_strategy}")

            fake_news["url"] = real_news["url"]
            fake_news["date"] = real_news["date"]
            fake_news["author"] = real_news["author"]
            fake_news["id"] = real_news["id"]
            fake_news["category"] = fake_news_type
            # fake_news["explanation"] = fake_explanation
            fake_news_samples.append(fake_news)

            with open(f"/data2/jerome/web_mining/final/Fake_News_Detection/fake_data/train/{fake_news_type}/generated_news.json", "w", encoding="utf-8") as f:
                json.dump(fake_news_samples, f, ensure_ascii=False, indent=4)

            with open(f"/data2/jerome/web_mining/final/Fake_News_Detection/fake_data/train/{fake_news_type}/detect_strategy.txt", "w") as f:
                print(detect_strategy, file=f)

    logger.info("Training complete.")