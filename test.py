# coding:utf-8
import sys
import urllib2
import json
import elasticsearch
import re
import logging
reload(sys)
import jieba
import jieba.posseg as pseg
sys.setdefaultencoding("utf-8")


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

def request_post(url, data):
    headers = {'Content-Type': 'application/json'}
    request = urllib2.Request(url=url, headers=headers, data=json.dumps(data))
    response = urllib2.urlopen(request).read()
    return response


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
        contents.append(title + summary)
    return contents


def initindexs(char, string):
    index = []
    length = len(string)
    for i in range(length):
        if char == string[i]:
            index.append(i + 1)  # 保存相同字符坐标+1的位置
    return index


def Substring(str1, str2):
    str1_len = len(str1)
    str2_len = len(str2)
    length = 0
    longest = 0
    startposition = 0
    start = 0
    for i in range(str1_len):
        start = i
        index = initindexs(str1[i], str2)
        index_len = len(index)
        for j in range(index_len):
            end = i + 1
            while end < str1_len and index[j] < str2_len and str1[end] == str2[index[j]]:  # 保证下标不会超出列表范围
                end += 1
                index[j] += 1
            length = end - start
            if length > longest:
                longest = length
                startposition = start

    return startposition, longest

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
        for j in xrange(1, len(titles)):
            str1, str2 = titles[i], titles[j]
            start, longest = Substring(titles[i], titles[j])
            str = str1[start:start + longest]
            if str in topic_dic:
                topic_dic[str] += 1
            else:
                topic_dic[str] = 1
    return max(topic_dic, key=topic_dic.get)

def find_lcseque(s1, s2):
    # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
    m = [[0 for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
    # d用来记录转移方向
    d = [[None for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            if s1[p1] == s2[p2]:  # 字符匹配成功，则该位置的值为左上方的值加1
                m[p1 + 1][p2 + 1] = m[p1][p2] + 1
                d[p1 + 1][p2 + 1] = 'ok'
            elif m[p1 + 1][p2] > m[p1][p2 + 1]:  # 左值大于上值，则该位置的值为左值，并标记回溯时的方向
                m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
                d[p1 + 1][p2 + 1] = 'left'
            else:  # 上值大于左值，则该位置的值为上值，并标记方向up
                m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
                d[p1 + 1][p2 + 1] = 'up'
    (p1, p2) = (len(s1), len(s2))
    #print numpy.array(d)
    s = []
    while m[p1][p2]:  # 不为None时
        c = d[p1][p2]
        if c == 'ok':  # 匹配成功，插入该字符，并向左上角找下一个
            s.append(s1[p1 - 1])
            p1 -= 1
            p2 -= 1
        if c == 'left':  # 根据标记，向左找下一个
            p2 -= 1
        if c == 'up':  # 根据标记，向上找下一个
            p1 -= 1
    s.reverse()
    return ''.join(s)




    # curPaht = '/Users/yanpeiying/Desktop/workspace/emotion_analysis/nlp_recognition/'
# print "加载主题模型..."
# lda_model = models.ldamodel.LdaModel.load(curPaht + 'models/sogou_corpus/sogou_corpus_segs_lda.model')
# dictionary = corpora.Dictionary.load(curPaht + 'models/sogou_corpus/sogou_corpus_segs_lda.dic')
# print "加载用户词典..."
# jieba.load_userdict(curPaht + 'dicts/emotion_dict/pos_all_dict.txt')
# jieba.load_userdict(curPaht + 'dicts/emotion_dict/neg_all_dict.txt')
# posdict = obj.read_lines(curPaht + "dicts/emotion_dict/pos_all_dict.txt")
# negdict = obj.read_lines(curPaht + "dicts/emotion_dict/neg_all_dict.txt")
# print "加载程度副词词典..."# 程度副词词典
# mostdict = obj.read_lines(curPaht + 'dicts/degree_dict/most.txt')   # 权值为2
# verydict = obj.read_lines(curPaht + 'dicts/degree_dict/very.txt')   # 权值为1.5
# moredict = obj.read_lines(curPaht + 'dicts/degree_dict/more.txt')   # 权值为1.25
# ishdict = obj.read_lines(curPaht + 'dicts/degree_dict/ish.txt')   # 权值为0.5
# insufficientdict = obj.read_lines(curPaht + 'dicts/degree_dict/insufficiently.txt')  # 权值为0.25
# inversedict = obj.read_lines(curPaht + 'dicts/degree_dict/inverse.txt')  # 权值为-1
#
# topic_classification_dict = obj.get_topic_dict()
#
# new_grocery = Grocery(curPaht + 'models/short_text_classification')
# new_grocery.load()
# classifier = fasttext.load_model(curPaht + 'models/longtext/chinaso_corpus.model.bin')
def cut_sentence(words):
    words = words.decode('utf8')
    start = 0
    i = 0
    token = 'meaningless'
    sents = []
    punt_list = ',.!?;~，。！？；～… '.decode('utf8')
    for word in words:
        if word not in punt_list:  # 如果不是标点符号
            i += 1
            token = list(words[start:i + 2]).pop()
        elif word in punt_list and token in punt_list:  # 处理省略号
            i += 1
            token = list(words[start:i + 2]).pop()
        else:
            sents.append(words[start:i + 1])  # 断句
            start = i + 1
            i += 1
    if start < len(words):  # 处理最后的部分
        sents.append(words[start:])
    return sents

if __name__ == '__main__':
    for word in cut_sentence("张学友被喊刘德华歌，留给你一个屁股自行体会"):
        print word

    # words = pseg.cut("张学友被喊刘德华 歌神：留给你一个屁股自行体会")
    # for word, flag in words:
    #     print('%s %s' % (word, flag))
    #print get_hot_topic(titles[:20])
    # word = '洛克菲勒'
    # print len(word.decode('utf-8'))
    # #print len(word)
    # for i in xrange(len(word) - 1):
    #     x1 = word[:i+1]
    #     x2 = word[i+1:]
    #     print str(x1), str(x2)
    l1 = ['a','b']
    l2 = ['c', 'd']
    l1.extend(l2)
    print l1
