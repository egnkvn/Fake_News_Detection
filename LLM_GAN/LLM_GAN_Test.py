from llm_gan import run_llm_gan_train_set
import json
# with open('../fake_data/train/fake_news.json', 'r', encoding='utf-8') as f:
#     real_news_examples = json.load(f)

with open('../new_data/international_bbc_news.json', 'r', encoding='utf-8') as f:
    real_news_examples = json.load(f)

demo_data = []
for real_news_example in real_news_examples:
    if real_news_example['title'] == "特朗普晶片戰略的風險：美國難以撼動亞洲的地位":
        demo_data.append(real_news_example)

run_llm_gan_train_set(demo_data, num_rounds=1)
