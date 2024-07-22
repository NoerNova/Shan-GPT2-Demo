# Shan GPT2 finetuned model inference demo

A Shan GPT2 finetuned model with news domain datasets.


## Model
[shan_gpt2_news](https://huggingface.co/NorHsangPha/shan_gpt2_news)


## Datasets

- [shannews.org](https://shannews.org) [datasets](https://huggingface.co/datasets/NorHsangPha/shan-news-shannews_org)
- [taifreedom.com](https://taifreedom.com) [datasets](https://huggingface.co/datasets/NorHsangPha/shan-news-taifreedom_com)
- [ssppssa.org](https://ssppssa.org) [datasets](https://huggingface.co/datasets/NorHsangPha/shan-news-ssppssa_org)
- [shan.shanhumanrights.org](https://shan.shanhumanrights.org) [datasets](https://huggingface.co/datasets/NorHsangPha/shan-news-shanhumanrights_org)


## Usage
```bash
# install requirements
pip install -r requirements.txt
```

```bash
# run
python app.py

# or gradio debug mode
gradio app.py
```
