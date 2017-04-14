# encoding=utf-8
__author__ = 'hikonga'
import random
import os
import sys
from sklearn.feature_extraction.text import CountVectorizer
import jieba
from  sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# http://www.jianshu.com/p/516f009c0875  全面的介绍的sklearn
reload(sys)
sys.setdefaultencoding('utf-8')
rdm = random.randint(1, 10)  # 生成随机数，用来提取测试集


def extact(dir):
    docs = []
    categoryDict = {}
    contents = []
    labels = []
    for parent, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            cat, docId = extract_filename(filename)
            if not categoryDict.has_key(cat):
                categoryDict[cat] = len(categoryDict)
            docs.append(docId)
            labels.append(categoryDict[cat])
            with open(parent + '/' + filename) as file:
                contList = file.readlines()
                contents.append(''.join(contList))
    return docs, categoryDict, contents, labels


def extract_filename(filename):
    names = filename.split('_')
    category = names[0]
    docId = names[1].split('.')[0]
    return category, category + "_" + docId


stopwords = ["的", "了", "在", "是", "我", "有", "和", "就",
             "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
             "会", "着", "没有", "看", "好", "自己", "这"]


# 向量化
def vectorize(contents):
    jieba_tokenizer = lambda x: jieba.cut(x)  # cut_all默认为false,false为精确模式，true为全模式
    v = CountVectorizer(tokenizer=jieba_tokenizer, min_df=2,
                        stop_words=stopwords)  # CountVectorizer,HashingVectorizer,TfidfVectorizer
    vecs = v.fit_transform(contents)
    return vecs


# 评价模型
def evaluete(model, x_data, y_data, lables, targetnames):
    x_pred = model.predict(x_data)
    print model.score(y_data, x_pred)
    print classification_report(y_data, x_pred, labels=lables, target_names=targetnames)
    print metrics.precision_score(y_data, x_pred,
                                  average='macro')


dirPath = unicode("F:/dataset/corpus_6_4000",
                  "utf8")
docs, categorys, contents, targets = extact(dirPath)
xdata = vectorize(contents)
x_train, x_test, y_train, y_test = train_test_split(xdata, targets, test_size=0.25)
if not os.path.exists('MultinomialNB'):
    model = MultinomialNB()  # 朴素贝叶斯分为了高斯型，多项式型和伯努利型
    model.fit(x_train, y_train)
    joblib.dump(model, 'MultinomialNB')
else:
    model = joblib.load('MultinomialNB')
lables = []
targetnames = []
for key in categorys:  # pyhon只支持对key的遍历，不用能用for key,value in 这种形式，这时候会提示ValueError:too many values to unpack
    lables.append(categorys[key])
    targetnames.append(key)
print '----train evaluate---'
evaluete(x_train, y_train, lables, targetnames)

print '----test evaluate---'
evaluete(x_test, y_test, lables, targetnames)
