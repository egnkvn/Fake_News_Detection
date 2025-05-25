from llm_gan import run_llm_gan
import json

with open('new_data/taiwan_bbc_news.json', 'r', encoding='utf-8') as f:
    real_news_examples = json.load(f)

real_news_examples = real_news_examples[:5]

run_llm_gan(real_news_examples, num_rounds=3)
