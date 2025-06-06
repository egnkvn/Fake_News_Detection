# Fake_News_Detection

## Project structure
### Data generation
![Model Comparison](img/data_pipeline.png)
### LLM agent module
![Model Comparison](img/agent_module.png)
### Search tool pipline
![Model Comparison](img/search_pipeline.png)

## How to run code
Create an .env file and Paste your OpenAI key in that file.

### Web Scraping
Install the `scrapy` package with the command
```
pip install scrapy
```

Change directory to `web_scarping/`
```
cd web_scraping
```

If you want to scrape news about China, change the directory by
```
cd ./bbc_china_news
```
Then you can run the code with the desired output file path
```
scrapy crawl bbc_china -o chian_bbc_news.json
```

### Dataset
In LLM GAN directory and run the following command

```
# specify your real_news.json path and store directory in run.py
python run.py
```
### Model
Change directory to `model/`, run:
```
python agent.py
```
Then the agent results would be saved under `log/`. You can change the testing news data in code by changing the path of dataset.

Run:
```
python baseline_truc.py
```
can train a BERT model on our dataset, saved as `bert_model_our.pt`

Run:
```
python baseline_pkl.py
```
can train a BERT model on weibo21 dataset, saved as `bert_model_weibo.pt`

