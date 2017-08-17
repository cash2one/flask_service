# -*- coding:utf-8 -*-

import jieba
import pymysql.cursors
import json
import datetime
#删除停用词
def del_stopwords(seg_sent):
    stopwords = [line.rstrip().decode('utf-8') for line in open('dicts/emotion_dict/stopwords.txt')]
    new_sent = []
    for word in seg_sent:
        if word.rstrip() in stopwords or not word.rstrip():
            continue
        else:
            new_sent.append(word)
    return new_sent
#插入数据库 表名 T_wordCount
def insertTodb(countofNews, startTime, endTime, curTime, js_content):
    db = pymysql.connect(host='10.10.160.146',  # your host, usually localhost
                         port=4306,
                         user='sentiment',  # your username
                         passwd='sentiment_123',  # your password
                         db='sentiment',
                         charset='utf8')  # name of the data base
    cur = db.cursor()
    query = "INSERT INTO T_wordCount(countofNews, startTime, endTime, time, content) VALUES(%s,%s,%s,%s,%s)"
    args = (countofNews, startTime, endTime, curTime, js_content)
    cur.execute(query, args)
    db.commit()
    db.close()

def main():
    word_dic = {}
    #读取内容，以后改成从ES中读取
    train_contents = [line.rstrip() for line in open('data/baidu_hotdata_test.txt')]

    content = ''
    for c in train_contents:
        content += c
    #分词并删除停用词
    word_list = del_stopwords(list(jieba.cut(content, cut_all=True)))
    #wordcount统计
    for word in word_list:
        if word in word_dic:
            word_dic[word]+=1
        else:
            word_dic[word]=1
    #转换成tuple并排序
    #word_dic2item = sorted(word_dic.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    #转换成json串
    js_content = json.dumps(word_dic).decode('utf-8')
    # js_content = ''
    # for word in word_dic2item:
    #     js_content +=  word[0]
    #     js_content +=  word[1]
    curTime = datetime.datetime.now()
    #插入数据库
    insertTodb(len(train_contents), curTime, curTime, curTime, js_content)

if __name__ == '__main__':
    main()

