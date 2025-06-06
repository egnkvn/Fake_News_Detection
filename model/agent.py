# main.py
import os
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  
from langchain_community.utilities import SerpAPIWrapper, GoogleSerperAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.agents import Tool, initialize_agent, AgentType
from datetime import datetime
from utils.search_engine import search_online
import json
from collections import defaultdict
from tqdm import tqdm

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.0, 
    openai_api_key=OPENAI_API_KEY
)
# search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
# search = GoogleSerperAPIWrapper()

def make_chain(template: str, output_key: str):
    return LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(template),
        output_key=output_key
    )

# 4️⃣ 定義每個步驟的 PromptTemplate
url_prompt = PromptTemplate.from_template(
    "URL 工具 – 描述以下網域概覽，並判斷是否可信：\nURL: {url}"
)
phrase_prompt = PromptTemplate.from_template(
    "Phrase 工具 – 判斷下列新聞是否含有聳動、誇張或情緒化詞彙，並簡要說明：\nNews: {news}"
)
language_prompt = PromptTemplate.from_template(
    "Language 工具 – 找出下列新聞中的拼寫、語法錯誤，或不當 ALL CAPS，並簡要說明：\nNews: {news}"
)
commonsense_prompt = PromptTemplate.from_template(
    "Commonsense 工具 – 文章發布於 {date}，根據常識判斷下列新聞是否合理？指出任何與常識衝突之處：\nNews: {news}"
)
title_content_prompt = PromptTemplate.from_template(
     "Title 工具 – 判斷下列新聞【標題】與【內容】是否相關一致？指出任何不一致之處：\n標題: {title}\n內容: {news}"
)
search_prompt = PromptTemplate.from_template(
    """Search 工具 –  
新聞摘要：{news}  
發布日期：{date}  

以下是截至 {date}，針對「{news}」搜尋到的前十筆結果：  
{search_results}

請逐一比較這些結果，看它們是否**完全支持**原始標題的說法：  
- 如果大多數（≥3/5）結果支持，請回覆：Match （並簡要說明支持依據）  
- 如果大多數結果都找不到對應報導或內容明顯不符，請回覆：NoMatch （並簡要說明矛盾點）  
"""
)
final_prompt = PromptTemplate.from_template(
    """綜合驗證（文章發布於 {date}）：

以下是各工具的檢測結果：
1. 搜尋結果一致性：{search_result}
2. 常識合理性檢測：{commonsense_result}
3. 標題-內容一致性：{title_content_result}
4. 網域可信度：{url_result}
5. 煽動性語言檢測：{phrase_result}
6. 語言錯誤檢測：{language_result}

請模擬人類事實查核流程，重點評估「搜尋結果一致性」與「常識合理性」。再來，若標題-內容不一致，則判斷為fake。綜合判斷新聞真偽。請以以下格式回覆。：
Label: <real/fake>
Reason: <簡要說明最關鍵的判斷依據，150 字以內>
"""
)

# 5️⃣ 把 Prompt 與 LLM 組成 Runnable
url_runnable        = url_prompt | llm
phrase_runnable     = phrase_prompt | llm
language_runnable   = language_prompt | llm
commonsense_runnable= commonsense_prompt | llm
title_content_runnable = title_content_prompt | llm
search_runnable     = search_prompt | llm
final_runnable      = final_prompt | llm

# 6️⃣ 為每支工具寫一個 func，內部呼叫 .invoke()
def url_tool(url: str) -> str:
    out = url_runnable.invoke({"url": url})
    return out.content.strip()

def phrase_tool(news: str) -> str:
    out = phrase_runnable.invoke({"news": news})
    return out.content.strip()

def language_tool(news: str) -> str:
    out = language_runnable.invoke({"news": news})
    return out.content.strip()

def commonsense_tool(news: str, date: str) -> str:
    out = commonsense_runnable.invoke({"news": news, "date": date})
    return out.content.strip()

def title_content_tool(title: str, news: str) -> str:
    out = title_content_runnable.invoke({"title": title, "news": news})
    return out.content.strip()

def search_tool(news: str, date: str) -> str:
    summarize_query_prompt = PromptTemplate.from_template(
        "請用一句話（不超過50字）概括下列新聞，以便後續搜尋：\n\n{content}"
    )
    summarize_query_chain = LLMChain(llm=llm, prompt=summarize_query_prompt)
    query = summarize_query_chain.predict(content=news).strip()
    # print(f"=== 搜尋關鍵字 ===\n{query}\n")
    try:
        cutoff = datetime.fromisoformat(date).date()
    except ValueError:
        cutoff = datetime.now().date()
    docs = search_online(query, cutoff_date=cutoff, num_results=5)

    summarize_prompt = PromptTemplate.from_template(
        "請用一句話（不超過50字）概括下列文章內容：\n\n{content}"
    )
    summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)
    summaries = []
    for i, text in enumerate(docs, start=1):
        excerpt = text[:300].replace("\n", " ")
        summary = summarize_chain.predict(content=excerpt)
        summaries.append(f"{i}. {summary or '（內容不足或抓取失敗）'}")

    joined = "\n".join(summaries)
    # print(f"=== 搜尋結果摘要 ===\n{joined}\n")
    out = search_runnable.invoke({
        "news":      query,
        "date":      date,
        "search_results": joined
    })
    return out.content.strip()

def fact_agent_pipeline(url: str, title: str, news: str, date: str) -> str:
    url_res         = url_tool(url)
    phrase_res      = phrase_tool(news)
    language_res    = language_tool(news)
    commonsense_res = commonsense_tool(news, date)
    title_content_res = title_content_tool(title, news)
    search_res      = search_tool(news, date)

    # 最後跑 Checklist
    final_inputs = {
        "url_result":         url_res,
        "phrase_result":      phrase_res,
        "language_result":    language_res,
        "commonsense_result": commonsense_res,
        "title_content_result": title_content_res,
        "search_result":      search_res,
        "date":               date,
    }
    final_out = final_runnable.invoke(final_inputs)
    return final_out.content.strip()

# 8️⃣ 執行示例
if __name__ == "__main__":
    with open('../fake_data/test/real_news.json') as f:
        data = json.load(f)
    print(len(data), "條測試數據")

    logs = []
    logs_path = "log/logs.jsonl"

    with open(logs_path, "a", encoding="utf-8") as log_f:
        for d in tqdm(data, desc="Processing"):
            rec = {
                "id":       d.get("id"),
                "category": d.get("category"),
                "label":    None,
                "Reason":  None,
                "error":    None,
            }
            try:
                verdict = fact_agent_pipeline(
                    url=d["url"],
                    title=d["title"],
                    news=d["content"],
                    date=d["date"]
                )
                for line in verdict.splitlines():
                    line = line.strip()
                    if line.startswith("Label:"):
                        rec["label"] = line.split("Label:",1)[1].strip()
                    elif line.startswith("Reason:"):
                        rec["Reason"] = line.split("Reason:",1)[1].strip()
                if rec["label"] is None:
                    rec["label"] = "Unknown"
                if rec["Reason"] is None:
                    rec["Reason"] = ""
            except Exception as e:
                rec["error"] = str(e)

            log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log_f.flush()
    