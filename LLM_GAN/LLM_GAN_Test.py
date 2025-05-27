from llm_gan import run_llm_gan_train_set, run_llm_gan
import json
with open('../fake_data/train/fake_news.json', 'r', encoding='utf-8') as f:
    real_news_examples = json.load(f)


run_llm_gan_train_set(real_news_examples, num_rounds=1)
