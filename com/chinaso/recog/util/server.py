# -*- coding:utf-8 -*-
import sys
from tgrocery import Grocery

reload(sys)
sys.setdefaultencoding("utf-8")
import os
from gensim import corpora, models
from numpy import *
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import fasttext
from pypinyin import lazy_pinyin, TONE2
import jieba
import jieba.analyse
import jieba.posseg as pseg
import Levenshtein

class Util:

    #短文本分类
    def get_text_classification(self, text):
        return str(new_grocery.predict(text))

    #基于fasttext的长文本分类
    def get_longtext_classification(self, text):
        text_pinying = ' '.join(lazy_pinyin(text, style=TONE2))
        sentence1 = [text_pinying]
        labels1 = classifier.predict(sentence1)
        class1 = labels1[0][0]
        #print("Sentence: ", sentence1[0])
        #print("Label: %d; label name: %s" %(int(class1[-1]), dicts[int(class1[-1])]))
        return str(dict[int(class1[-1])])

    #获取主题词
    # def get_topics(self, text):
    #     seg_sent = self.segmentation(text)   # 分词
    #     seg_sent = self.del_stopwords(seg_sent)[:]
    #     bow_vector = dictionary.doc2bow(seg_sent)
    #     topics = ''
    #     for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1 * tup[1]):
    #         topics += ' '.join([a.split('*')[-1] for a in str(lda_model.print_topic(index, 5)).split('+')])
    #     # #print "Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 10))
    #     return topics

    #计算指定词库的话题分类
    def get_topic_type(self, text):
        seg_sent = self.segmentation(text)
        seg_sent = self.del_stopwords(seg_sent)[:]
        final_result = 0
        max_socre = 0
        for i, dic in enumerate(topic_classification_dict):
            score = 0
            for word in seg_sent:
                for d in dic:
                    if Levenshtein.ratio(word, d) > 0.9:
                        score += 1
                        break
            if score > max_socre:
                max_socre = score
                final_result = i + 1
        return topic_dict[final_result]

    #获取人名
    def get_human_name(self, text):
        #jieba.load_userdict("dict.txt")
        words = pseg.cut(text)
        dict_name = {}
        for w in words:
            if len(w.word) < 2: continue
            if w.flag == "nr" or w.flag == "nrfg" :
                if w.word in dict_name:
                    dict_name[w.word] += 1
                else:
                    dict_name[w.word] = 1
        return dict_name


    #计算文章情感分数
    def get_sentiment_score(self, sent, keywords, title):
        result = 0
        cuted_review = self.cut_sentence(sent)  # 句子切分，单独对每个句子进行分析
        tfisf_matrix = mat(self.get_tfisf(cuted_review, title))#计算生成tfisf矩阵
        sum_row = tfisf_matrix.sum(axis=1)
        index_cuted_review = 1
        for sent in cuted_review:
            Wp, Wt, Wh, Wf, Wk = 0, 0, 0, 0, 0 		#句子权重值
            Wp = float(1) / min(index_cuted_review, len(cuted_review) - index_cuted_review + 1)
            seg_sent = self.segmentation(sent)   # 分词
            seg_sent = self.del_stopwords(seg_sent)[:]
            Wt = sum_row[index_cuted_review, 0]
            Wh = self.cosine_similarity(tfisf_matrix[0].tolist()[0], tfisf_matrix[index_cuted_review].tolist()[0])
            if '我' in seg_sent or '我们' in seg_sent:
                wf = 1
            else:
                wf = 0
            i = 0    # 记录扫描到的词的位置
            s = 0    # 记录情感词的位置
            poscount = 0    # 记录该分句中的积极情感得分
            negcount = 0    # 记录该分句中的消极情感得分
            for word in seg_sent:
                if word in keywords:
                    Wk += 1
                if word in posdict:
                    poscount += 1
                    for w in seg_sent[s:i]:
                        poscount = self.match(w, poscount)
                    s = i + 1  # 记录情感词的位置变化
                elif word in negdict:  # 如果是消极情感词
                    negcount += 1
                    for w in seg_sent[s:i]:
                        negcount = self.match(w, negcount)
                    s = i + 1
                # 如果是感叹号，表示已经到本句句尾
                elif word == "！".decode("utf-8") or word == "!".decode('utf-8'):
                    for w2 in seg_sent[::-1]:  # 倒序扫描感叹号前的情感词，发现后权值+2，然后退出循环
                        if w2 in posdict:
                            poscount += 2
                            break
                        elif w2 in negdict:
                            negcount += 2
                            break
                i += 1
            index_cuted_review += 1
            poscount, negcount = self.transform_to_positive_num(poscount, negcount)
            result += (0.15 * Wp + 0.1 * Wt + 0.5 * Wh + 0.1 * Wk + 0.15 * Wf) * (poscount - negcount)
        result = round(result, 1)
        return result
    #分词
    def segmentation(self, sentence):
        seg_list = jieba.cut(sentence)
        seg_result = []
        punt_list = ',.!?;~，。！？；～… ：:...-'.decode('utf8')
        for w in seg_list:
            if w.decode('utf8') not in punt_list and w.rstrip():
               seg_result.append(w)
        return seg_result

    #获取关键词
    def get_keywords(self, text):
        keywords = jieba.analyse.extract_tags(text, topK=20, withWeight=False, allowPOS=())
        return keywords

    #分句
    def cut_sentence(self, words):
        words = words.decode('utf8')
        start = 0
        i = 0
        token = 'meaningless'
        sents = []
        punt_list = ',.!?;~，。！？；～… '.decode('utf8')
        for word in words:
            if word not in punt_list:   # 如果不是标点符号
                i += 1
                token = list(words[start:i+2]).pop()
            elif word in punt_list and token in punt_list:  # 处理省略号
                i += 1
                token = list(words[start:i+2]).pop()
            else:
                sents.append(words[start:i+1])   # 断句
                start = i + 1
                i += 1
        if start < len(words):   # 处理最后的部分
            sents.append(words[start:])
        return sents

    # 以句子为单位转换整篇文章为矩阵
    def get_tfisf(self, cuted_review, title):
        corpus = []
        title_seg = self.segmentation(title)
        corpus.append(' '.join(title_seg))
        for sen in cuted_review:
            seg_sent = self.segmentation(sen)
            seg_sent = self.del_stopwords(seg_sent)[:]
            corpus.append(' '.join(seg_sent))
        vectorizer = CountVectorizer()
        counts_train = vectorizer.fit_transform(corpus)
        transformer = TfidfTransformer()
        tfidf_train = transformer.fit_transform(counts_train)
        train_set = tfidf_train.toarray()
        return train_set

    # 计算句子之间余弦相似度
    def cosine_similarity(self, v1, v2):
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(v1)):
            x = v1[i]; y = v2[i]
            sumxx += x*x
            sumyy += y*y
            sumxy += x*y
        if sumxx*sumyy == 0:
            return 0.0
        return sumxy/math.sqrt(sumxx*sumyy)

    # 去除停用词
    def del_stopwords(self, seg_sent):
        stopwords = [line.rstrip().decode('utf-8') for line in open('dicts/emotion_dict/stopwords.txt')]
        new_sent = []   # 去除停用词后的句子
        for word in seg_sent:
            if word.rstrip() in stopwords or not word.rstrip():
                continue
            else:
                new_sent.append(word)
        return new_sent

    # 程度副词处理，根据程度副词的种类不同乘以不同的权值
    def match(self, word, sentiment_value):
        if word in mostdict:
            sentiment_value *= 2.0
        elif word in verydict:
            sentiment_value *= 1.75
        elif word in moredict:
            sentiment_value *= 1.5
        elif word in ishdict:
            sentiment_value *= 1.2
        elif word in insufficientdict:
            sentiment_value *= 0.5
        elif word in inversedict:
            sentiment_value *= -1
        return sentiment_value

    # 情感得分的最后处理，防止出现负数
    # Example: [5, -2] →  [7, 0]; [-4, 8] →  [0, 12]
    def transform_to_positive_num(self, poscount, negcount):
        pos_count = 0
        neg_count = 0
        if poscount < 0 and negcount >= 0:
            neg_count += negcount - poscount
            pos_count = 0
        elif negcount < 0 and poscount >= 0:
            pos_count = poscount - negcount
            neg_count = 0
        elif poscount < 0 and negcount < 0:
            neg_count = -poscount
            pos_count = -negcount
        else:
            pos_count = poscount
            neg_count = negcount
        return (pos_count, neg_count)

    def read_lines(self, filename):
        fp = open(filename, 'r')
        lines = []
        for line in fp.readlines():
            line = line.strip()
            line = line.decode("utf-8")
            lines.append(line)
        fp.close()
        return lines
    def get_topic_dict(self):
        result = []
        path = 'dicts/topic_classification_dict/'
        #filenNames= [str(item[1]) for item in topic_dict.items()]
        #topic_names = {}
        for fileName in xrange(1, 10):
            fullfilename = path + str(fileName) + '.txt'
            curr = self.read_lines(fullfilename)
            result.append(curr)
        return result
obj = Util()
topic_dict = {
        0:'暂无分类',
        1:'公共安全',
        2:'医疗卫生',
        3:'城乡建设',
        4:'涉军社警',
        5:'涉官涉腐',
        6:'涉民族宗教',
        7:'涉网涉宣',
        8:'环境保护',
        9:'社会保障',
        10:'经济发展',
}
dict = {
        1:'国际',
        2:'政务',
        3:'军事',
        4:'财经',
        5:'科技',
        6:'社会',
        7:'体育',
        8:'娱乐',
        9:'军事',
        10:'健康',
        11:'汽车',
        12:'互联',
        13:'生活',
        14:'旅游',
        15:'教育'
        }

#curPath = '/Users/yanpeiying/Desktop/workspace/emotion_analysis/nlp_recognition/'
# print "加载主题模型..."
# lda_model = models.ldamodel.LdaModel.load('models/sogou_corpus/sogou_corpus_segs_lda.model')
# dictionary = corpora.Dictionary.load('models/sogou_corpus/sogou_corpus_segs_lda.dic')
print "加载用户词典..."
jieba.load_userdict('dicts/emotion_dict/pos_all_dict.txt')
jieba.load_userdict('dicts/emotion_dict/neg_all_dict.txt')
posdict = obj.read_lines("dicts/emotion_dict/pos_all_dict.txt")
negdict = obj.read_lines("dicts/emotion_dict/neg_all_dict.txt")
print "加载程度副词词典..."# 程度副词词典
mostdict = obj.read_lines('dicts/degree_dict/most.txt')   # 权值为2
verydict = obj.read_lines('dicts/degree_dict/very.txt')   # 权值为1.5
moredict = obj.read_lines('dicts/degree_dict/more.txt')   # 权值为1.25
ishdict = obj.read_lines('dicts/degree_dict/ish.txt')   # 权值为0.5
insufficientdict = obj.read_lines('dicts/degree_dict/insufficiently.txt')  # 权值为0.25
inversedict = obj.read_lines('dicts/degree_dict/inverse.txt')  # 权值为-1

topic_classification_dict = obj.get_topic_dict()

new_grocery = Grocery('models/short_text_classification')
new_grocery.load()
classifier = fasttext.load_model('models/longtext/chinaso_corpus.model.bin')
