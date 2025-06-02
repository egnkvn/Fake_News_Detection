from llm_gan import run_llm_gan_train_set, run_llm_gan
import json
import random

with open('../fake_data/train/fake_news.json', 'r', encoding='utf-8') as f:
    real_news_examples = json.load(f)

# with open('../old/test/generated_news.json', 'r', encoding='utf-8') as f:
#     generated_news_examples = json.load(f)

# target_data = []
# for generated_news_example in generated_news_examples:
#     if generated_news_example['category'] == 'misleading' or generated_news_example['category'] == 'manipulated':
#         for real_news_example in real_news_examples:
#             if real_news_example['id'] == generated_news_example['id']:
#                 target_data.append(real_news_example)

# target_data = random.sample(target_data, 40)

run_llm_gan_train_set(real_news_examples, num_rounds=1)

