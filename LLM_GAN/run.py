from llm_gan import run_llm_gan_train_set, run_llm_gan
import json
import random

with open('../new_data/international_bbc_news.json', 'r', encoding='utf-8') as f:
    real_news_examples_1 = json.load(f)
with open('../new_data/taiwan_bbc_news.json', 'r', encoding='utf-8') as f:
    real_news_examples_2 = json.load(f)

store_pos = ""
real_news_examples = real_news_examples_1.extend(real_news_examples_2)

run_llm_gan_train_set(real_news_examples, store_pos, num_rounds=1)

