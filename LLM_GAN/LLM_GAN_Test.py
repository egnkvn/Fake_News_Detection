from llm_gan import run_llm_gan_train_set, run_llm_gan
import json
with open('../fake_data/test/fake_news.json', 'r', encoding='utf-8') as f:
    real_news_examples = json.load(f)

# with open('../old/generated_news.json', 'r', encoding='utf-8') as f:
#     generated_news_examples = json.load(f)

# target_data = []
# for generated_news_example in generated_news_examples:
#     if generated_news_example['category'] == 'false_connection':
#         for real_news_example in real_news_examples:
#             if real_news_example['id'] == generated_news_example['id']:
#                 target_data.append(real_news_example)

# print(len(target_data))
# input()
run_llm_gan_train_set(real_news_examples, num_rounds=1)

