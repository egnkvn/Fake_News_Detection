import json

with open('new_data/international_bbc_news.json', 'r', encoding='utf-8') as f:
    real_examples = json.load(f)

with open('fake_data_test/demo2.json', 'r', encoding='utf-8') as f:
    fake_examples = json.load(f)

for fake_example in fake_examples:
    for real_example in real_examples:
        if fake_example['id'] == real_example['id']:
            print(real_example)
    print("============================================")
    print(fake_example)
    input()