# main.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  
from langchain_community.utilities import SerpAPIWrapper, GoogleSerperAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.agents import Tool, initialize_agent, AgentType

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.0, 
    openai_api_key=OPENAI_API_KEY
)
# search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
search = GoogleSerperAPIWrapper()

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
    "Commonsense 工具 – 根據常識判斷下列新聞是否合理？指出任何與常識衝突之處：\nNews: {news}"
)
standing_prompt = PromptTemplate.from_template(
    "Standing 工具 – 若此新聞為政治新聞，請分析其是否表達偏頗立場；若非政治新聞回覆 Not Applicable：\nNews: {news}"
)
search_prompt = PromptTemplate.from_template(
    "Search 工具 – 請根據以下搜尋結果，判斷是否有資訊與新聞相互矛盾，並簡要說明：\n{search_results}"
)
final_prompt = PromptTemplate.from_template(
    """最終 Checklist 判定：
1. 網域 URL 不可信 → fake
2. 聳動/誇張語言 → fake
3. 拼寫/語法錯誤或 ALL CAPS → fake
4. 與常識衝突或像八卦 → fake
5. 政治偏頗 → fake
6. 搜尋結果衝突 → fake

【工具輸出】
- URL: {url_result}
- Phrase: {phrase_result}
- Language: {language_result}
- Commonsense: {commonsense_result}
- Standing: {standing_result}
- Search: {search_result}

請返回 'real' 或 'fake'，並說明理由："""
)

# 5️⃣ 把 Prompt 與 LLM 組成 Runnable
url_runnable        = url_prompt | llm
phrase_runnable     = phrase_prompt | llm
language_runnable   = language_prompt | llm
commonsense_runnable= commonsense_prompt | llm
standing_runnable   = standing_prompt | llm
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

def commonsense_tool(news: str) -> str:
    out = commonsense_runnable.invoke({"news": news})
    return out.content.strip()

def standing_tool(news: str) -> str:
    out = standing_runnable.invoke({"news": news})
    return out.content.strip()

def search_tool(news: str) -> str:
    sr = search.run(news)  # SerpAPIWrapper 回傳 raw string
    out = search_runnable.invoke({"search_results": sr})
    return out.content.strip()

def fact_agent_pipeline(url: str, news: str) -> str:
    url_res         = url_tool(url)
    phrase_res      = phrase_tool(news)
    language_res    = language_tool(news)
    commonsense_res = commonsense_tool(news)
    standing_res    = standing_tool(news)
    search_res      = search_tool(news)

    # 最後跑 Checklist
    final_inputs = {
        "url_result":         url_res,
        "phrase_result":      phrase_res,
        "language_result":    language_res,
        "commonsense_result": commonsense_res,
        "standing_result":    standing_res,
        "search_result":      search_res,
    }
    final_out = final_runnable.invoke(final_inputs)
    return final_out.content.strip()

# 8️⃣ 執行示例
if __name__ == "__main__":
    example_url  = "https://tw.nextapple.com/sports/20250521/0A9E1D720B06E7D9E96FCDCD55DEE943"
    example_news = "「台灣怪力男」李灝宇奪3A上周MVP　教頭曝他上大聯盟最大優勢"

    verdict = fact_agent_pipeline(example_url, example_news)
    print("=== 最終判定 ===\n", verdict)