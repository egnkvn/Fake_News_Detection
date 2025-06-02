import json


fake_news_types = {
    "satire": 0,
    "false_context": 0,
    "fabricated": 0,
    "false_connection":0,
    "misleading&manipulated":0,
    "impostor":0,
    "manipulated":0
}

fake_news_examples = {
    "satire": [],
    "false_context": [],
    "fabricated": [],
    "false_connection": [],
    "misleading&manipulated": [],
    "impostor": [],
    "manipulated": []
}

with open('/data2/jerome/web_mining/final/Fake_News_Detection/fake_data/test/generated_news.json', 'r', encoding='utf-8') as f:
    examples = json.load(f)

for example in examples:
    fake_news_types[example["category"]] += 1
    fake_news_examples[example["category"]].append(example)

print(fake_news_types)
# for category, fake_news_example in fake_news_examples.items():

#     if len(fake_news_example) != 0:
#         with open(f"/data2/jerome/web_mining/final/Fake_News_Detection/fake_data/train/{category}/generated_news.json", "w", encoding="utf-8") as f:
#             json.dump(fake_news_example, f, ensure_ascii=False, indent=4)