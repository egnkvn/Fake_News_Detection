from llm_gan import run_llm_gan
import json
import random

random.seed(40)

with open('../new_data/taiwan_bbc_news.json', 'r', encoding='utf-8') as f:
    taiwan_bbc_examples = json.load(f)
with open('../new_data/international_bbc_news.json', 'r', encoding='utf-8') as f:
    international_bbc_examples = json.load(f)

total = taiwan_bbc_examples + international_bbc_examples

test_set = random.sample(total, 320)

news_id = set([news['id'] for news in test_set])

with open("../fake_data/test/fake_news.json", "w", encoding="utf-8") as f:
    json.dump(test_set[:280], f, ensure_ascii=False, indent=4)
with open("../fake_data/test/real_news.json", "w", encoding="utf-8") as f:
    json.dump(test_set[280:], f, ensure_ascii=False, indent=4)

fake_news_train = []
for example in total:
    if example['id'] not in news_id:
        fake_news_train.append(example)

with open("../fake_data/train/fake_news.json", "w", encoding="utf-8") as f:
    json.dump(fake_news_train[:1000], f, ensure_ascii=False, indent=4)
with open("../fake_data/train/real_news.json", "w", encoding="utf-8") as f:
    json.dump(fake_news_train[1000:], f, ensure_ascii=False, indent=4)


