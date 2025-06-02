import json

with open('new_data/international_bbc_news.json', 'r', encoding='utf-8') as f:
    real_examples_1 = json.load(f)

with open('new_data/taiwan_bbc_news.json', 'r', encoding='utf-8') as f:
    real_examples_2 = json.load(f)

real_examples = real_examples_1 + real_examples_2
with open('/data2/jerome/web_mining/final/Fake_News_Detection/fake_data/test/generated_news.json', 'r', encoding='utf-8') as f:
    fake_examples = json.load(f)
print(len(fake_examples))


# for fake_example in fake_examples:
#     if fake_example['category'] == "impostor":
#         for real_example in real_examples:
#             if fake_example['id'] == real_example['id']:
#                 print(real_example)
#         print("============================================")
#         print(fake_example)
#         input()

new_data = []
for fake_example in fake_examples:
    if fake_example['category'] == "fabricated":
        pass
    else:
        new_data.append(fake_example)

print(len(new_data))
input()
with open(f"/data2/jerome/web_mining/final/Fake_News_Detection/fake_data/test/generated_news_2.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)
