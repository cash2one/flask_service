# coding:utf-8
import sys
import urllib2
import elasticsearch

reload(sys)
sys.setdefaultencoding("utf-8")
from gensim import corpora, models, similarities
import re
import pandas as pd
import jieba.analyse
from k_means_plus_plus import *
import json
import pymysql.cursors
import logging


log = logging.getLogger()
log.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


'''过滤HTML中的标签
#将HTML中标签等信息去掉
#@param htmlstr HTML字符串.'''

def filter_tag(htmlstr):
    re_cdata = re.compile('<!DOCTYPE HTML PUBLIC[^>]*>', re.I)
    re_script = re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>', re.I)  # 过滤脚本
    re_style = re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>', re.I)  # 过滤style
    re_br = re.compile('<br\s*?/?>')
    re_h = re.compile('</?\w+[^>]*>')
    re_comment = re.compile('<!--[\s\S]*-->')
    s = re_cdata.sub('', htmlstr)
    s = re_script.sub('', s)
    s = re_style.sub('', s)
    s = re_br.sub('\n', s)
    s = re_h.sub(' ', s)
    s = re_comment.sub('', s)
    blank_line = re.compile('\n+')
    s = blank_line.sub('\n', s)
    s = re.sub('\s+', ' ', s)
    s = replaceCharEntity(s)
    return s

'''##替换常用HTML字符实体.
#使用正常的字符替换HTML中特殊的字符实体.
#你可以添加新的实体字符到CHAR_ENTITIES中,处理更多HTML字符实体.
#@param htmlstr HTML字符串.'''
def replaceCharEntity(htmlstr):
    CHAR_ENTITIES = {'nbsp': '', '160': '',
                     'lt': '<', '60': '<',
                     'gt': '>', '62': '>',
                     'amp': '&', '38': '&',
                     'quot': '"''"', '34': '"'}
    re_charEntity = re.compile(r'&#?(?P<name>\w+);')  # 命名组,把 匹配字段中\w+的部分命名为name,可以用group函数获取
    sz = re_charEntity.search(htmlstr)
    while sz:
        # entity=sz.group()
        key = sz.group('name')  # 命名组的获取
        try:
            htmlstr = re_charEntity.sub(CHAR_ENTITIES[key], htmlstr, 1)  # 1表示替换第一个匹配
            sz = re_charEntity.search(htmlstr)
        except KeyError:
            htmlstr = re_charEntity.sub('', htmlstr, 1)
            sz = re_charEntity.search(htmlstr)
    return htmlstr

# keams++ 聚类
def kmeanspp_cluster(data, num):
    kmpp = KMeansPlusPlus(data, num)
    kmpp.cluster()
    return kmpp

def train_model(train_set, demention=50):
    dictionary = corpora.Dictionary(train_set)
    dictionary.filter_extremes(no_below=1, no_above=1, keep_n=None)
    corpus = [dictionary.doc2bow(text) for text in train_set]
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=demention, alpha=0.3, eta=0.2)
    return corpus, lda


def cluster2dic(lda, num, corpus, train_content):
    data_input = []
    for i in xrange(len(corpus)):
        line = lda.get_document_topics(corpus[i], minimum_probability=0.0)
        data_input_row = []
        for f in line:
            data_input_row.append(float(f[1]))
        data_input.append(data_input_row)
    print 'total data: ' + str(len(data_input))
    data = pd.DataFrame(data_input)
    kmpp = kmeanspp_cluster(data, num)
    dic = {}
    cluster_titles = {}
    for i, x in enumerate(kmpp.clusters):
        if x not in dic:
            dic[x] = []
            cluster_titles[x] = []
            dic[x].append(train_content[i])
            cluster_titles[x].append(train_content[i].split('\t')[0])
        else:
            dic[x].append(train_content[i])
            cluster_titles[x].append(train_content[i].split('\t')[0])
    return dic, cluster_titles

def es_search(index, start_time, end_time):
    contents = []
    es = elasticsearch.Elasticsearch(hosts=[{'host': '10.10.66.86', 'port': 9200},
                                            {'host': '10.10.66.87', 'port': 9200},
                                            {'host': '10.10.66.91', 'port': 9200}])
    # count = es.count(index=index,
    #                  body={"query": {"range": {"createdAt": \
    #                  {"from": start_time, "to": end_time}}}})
    count = es.count(index="cz_news",
                     body={"query": {
                         "range": {"createdAt": {"from": start_time, "to": end_time}}}})
    page_num = count['count']
    result = es.search(index=index,
                       body={"query": {"range": {"createdAt": \
                       {"from": start_time, "to": end_time}}}},
                       params={"size": page_num})
    for hit in result['hits']['hits']:
        try:
            title = hit['_source']['titleCN']
            summary = filter_tag(hit['_source']['summary'])
        except:
            continue
        contents.append(title + '\t' + summary)
    return contents

def segmentation(sentence):
        seg_list = jieba.cut(sentence)
        seg_result = []
        punt_list = ',.!?;~，。！？；～… ：:...-'.decode('utf8')
        for w in seg_list:
            if w.decode('utf8') not in punt_list and w.rstrip():
               seg_result.append(w)
        return seg_result

def del_stopwords(seg_sent):
    stopwords = [line.rstrip().decode('utf-8') for line in open('dicts/emotion_dict/stopwords.txt')]
    new_sent = []
    for word in seg_sent:
        if word.rstrip() in stopwords or not word.rstrip():
            continue
        else:
            new_sent.append(word)
        return new_sent

def get_topicTpye(url, data):
    headers = {'Content-Type': 'application/json'}
    request = urllib2.Request(url=url, headers=headers, data=json.dumps(data))
    return urllib2.urlopen(request).read()

def find_lcsubstr(s1, s2):
    m=[[0 for i in range(len(s2)+1)]  for j in range(len(s1)+1)]  #生成0矩阵，为方便后续计算，比字符串长度多了一列
    mmax=0   #最长匹配的长度
    p=0  #最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i]==s2[j]:
                m[i+1][j+1]=m[i][j]+1
                if m[i+1][j+1]>mmax:
                    mmax=m[i+1][j+1]
                    p=i+1
    return s1[p-mmax:p],mmax   #返回最长子串及其长度


def get_hot_topic(titles):
    topic_dic = {}
    for i in xrange(len(titles)):
        for j in xrange(i+1,len(titles)):
            str1,str2 = titles[i], titles[j]
            str, longest = find_lcsubstr(titles[i], titles[j])
            if longest <= 9: continue
            #str = str1[start:start + longest]
            if str in topic_dic:
                topic_dic[str] += 1
            else:
                topic_dic[str] = 1
    if not topic_dic: return '暂无话题'
    return max(topic_dic, key=topic_dic.get)

def insertTodb(all_topics, best_description, all_regoin, relative_newscount, topic_types, hot_topics):
    db = pymysql.connect(host='10.10.160.146',    # your host, usually localhost
                     port=4306,
                     user='sentiment',         # your username
                     passwd='sentiment_123',  # your password
                     db='sentiment',
                     charset='utf8')        # name of the data base
    cur = db.cursor()
    #content = [line.rstrip() for line in open('emotion_dict/pos_all_dict.txt')]

    query = "INSERT INTO T_HotTopic(name, description, region, keywords, relativeNewsCount, type, category) VALUES(%s,%s,%s,%s,%s,%s,%s)"
    for i in xrange(10):
        name = hot_topics[i]
        description = best_description[i]
        region = ','.join(all_regoin[i])
        keywords = ','.join(all_topics[i])
        relativeNewsCount = relative_newscount[i]
        type = '1'
        category = ','.join(topic_types[i])
        args = (name, description, region, keywords, relativeNewsCount, type, category)
        cur.execute(query, args)
        db.commit()
    db.close()
def main(demention, num_cluster, index, start_time, end_time):
    #get corpus from ES
    #train_contents = es_search(index, start_time, end_time)
    #------------------------用百度搜索热点话题做测试--------------------------------
    train_contents = [line.rstrip() for line in open('data/baidu_hotdata_test.txt')]
    #train_contents.extend([line.rstrip() for line in open('data/baidu_hotdata_t1.txt')])
    #train_contents.extend([line.rstrip() for line in open('data/baidu_hotdata_t2.txt')])

    log.info('get contents is done')
    #cut all contents
    train_set = [del_stopwords(segmentation(line.rstrip())) for line in train_contents]
    log.info('cutting is done')
    corpus, lda = train_model(train_set, demention=demention)
    log.info('training LDA model is done')
    num = num_cluster
    cluster_dic, cluster_titles = cluster2dic(lda, num, corpus, train_contents)
    log.info('kmeans++ clustering is done')
    tup = []
    for i in xrange(num):
        tup.append((i, len(cluster_dic[i])))
    tup = sorted(tup, key=lambda t: t[1], reverse=True)
    #filter all location name
    jieba.analyse.set_stop_words('dicts/region.txt')
    regoins = [line.rstrip().decode('utf-8') for line in open('dicts/region.txt')]
    all_topics = []
    all_regoin = []
    best_description = {}
    relative_newscount = []
    topic_types = []
    hot_topics = []
    for i, t in enumerate(tup[:24]):
        max_count = 0
        relative_newscount.append(t[1])
        sentences = cluster_dic[t[0]]
        titles = cluster_titles[t[0]]
        hot_topics.append(get_hot_topic(titles))
        sentence = ''.join(sentences)
        temp = []
        for regoin in regoins:
            count = sentence.count(regoin)
            if count:
                temp.append((count, regoin))
        temp = sorted(temp, key=lambda t: t[0], reverse=True)
        rst = [te[1] for te in temp[:5]]
        all_regoin.append(rst)

        log.info('话题' + str(i + 1) + '----------------------')
        log.info('地域: ' + ','.join(rst))
        tags = jieba.analyse.extract_tags(sentence, topK=10, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
        map_topicTpye = {}
        url = 'http://10.10.16.15:8000/topic_classify'
        for s in sentences:
            data = {"title":s, "text":''}
            ttype = get_topicTpye(url, data)
            if ttype != '暂未分类':
                if ttype in map_topicTpye:
                    map_topicTpye[ttype] += 1
                else:
                    map_topicTpye[ttype] = 1
            count = 0
            for ta in tags:
                count += s.count(ta)
            if count > max_count:
                max_count = count
                best_description[i] = s
        topic_type = []
        if map_topicTpye:
            items =  sorted(map_topicTpye.items(), key=lambda d: d[1])
            if len(items) > 3: len_items = 3
            else: len_items = len(items)
            for idx in xrange(len_items):
                topic_type.append(items[idx][0])
        else:
            topic_type.append('暂未分类')
        topic_types.append(topic_type)
        all_topics.append(tags)
        log.info('热点话题：' + hot_topics[i])
        log.info('话题标签：' + ','.join(tags))
        log.info('最佳描述：' + best_description[i])
        log.info('可选分类：' + ','.join(topic_type))
        log.info('相关新闻数：' + str(t[1]))
    #insertTodb(all_topics, best_description, all_regoin, relative_newscount, topic_types, hot_topics)
def usage():
    print '-h                           print help' + '\n' + \
          '-d                           demention for each content, default value is 200' + '\n' + \
          '-n                           number of clustering, default value is 100' + '\n' + \
          '-i                           index name for ES, default value is cz_news' + '\n' + \
          '-s                           start time for ES, default value is one day ago(YY-MM-DD HH:MM:SS)' + '\n' + \
          '-e                           end time for ES, default value is current time(YY-MM-DD HH:MM:SS)' + '\n'

def start_extaction():
    import datetime
    import sys, getopt


    opts, args = getopt.getopt(sys.argv[1:], "hd:n:i:s:e:")#['data=']
    demention = 200  #demention for each data, default value is 200
    num_cluster = 50   #number of clustering, default value is 100
    index = "cz_news" #index name from ES, default value is cs_news
    oneDayAgo = (datetime.datetime.now() - datetime.timedelta(days = 1))
    start_time = oneDayAgo.strftime("%Y-%m-%d %H:%M:%S") #start time for content, default value is one day ago
    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") #end time for content, default value is current time
    for op, value in opts:
        if op == "-d":
            demention = int(value)
        elif op == "-n":
            num_cluster = int(value)
        elif op == "-i":
            index = value
        elif op == "-s":
            start_time = value
        elif op == "-e":
            end_time = value
        elif op == "-h":
            usage()
            sys.exit()
        else:
            print 'unhandled option'
            sys.exit(3)
    main(demention, num_cluster, index, start_time, end_time)