# coding:utf-8
import json
from flask import Flask, request, render_template
import sys
import numpy as np
from com.chinaso.recog.app import app, server

reload(sys)
sys.setdefaultencoding("utf-8")


@app.route('/hello')
def hello_world():
    return render_template('hello.html')

#短文本分类
@app.route('/short_text', methods=['GET', 'POST'])
def get_text_classification():
    data = request.data
    j_data = json.loads(data)
    title = j_data['title']
    rst = server.get_text_classification(title)
    return str(rst)

#长文本分类
@app.route('/long_text', methods=['GET', 'POST'])
def get_longtext_classification():
    data = request.data
    j_data = json.loads(data)
    title = j_data['title']
    text = j_data['text']
    if title == '' and text == '':
        return str('None')
    rst = server.get_longtext_classification(title + text)
    return str(rst)

# @app.route('/topic', methods=['GET', 'POST'])
# def get_topic_kewords():
#     result = {}
#     data = request.data
#     j_data = json.loads(data)
#     title = j_data['title']
#     text = j_data['text']
#     topics = server.get_topics(title + text)
#     keywords = ' '.join(server.get_keywords(title + text))
#     result['topics'] = topics
#     result['keywords'] = keywords
#     return json.dumps(result)
#情感分析
@app.route('/sentiment', methods=['GET', 'POST'])
def get_topic():
    data = request.data
    j_data = json.loads(data)
    title = j_data['title']
    text = j_data['text']
    segs = server.get_keywords(text)
    score = server.get_sentiment_score(text, segs, title)
    result = 1 - 1.0/(1 + np.exp(-score / 5))
    return str(result)
#话题分类
@app.route('/topic_classify', methods=['GET', 'POST'])
def get_classification():
    data = request.data
    j_data = json.loads(data)
    title = j_data['title']
    text = j_data['text']
    rst = server.get_topic_type(title + text)
    return str(rst)

@app.route('/human_name', methods=['GET', 'POST'])
def get_human_name():
    data = request.data
    j_data = json.loads(data)
    title = j_data['title']
    text = j_data['text']
    rst = server.get_human_name(title + text)
    rst = sorted(rst.items(), key=lambda x: x[1], reverse=True)
    result = [item[0] for item in rst]
    if len(result) >= 3:
        return str(' '.join(result[:3]))
    return str(' '.join(result))


