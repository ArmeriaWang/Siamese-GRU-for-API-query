import math
import pickle
import utils
import numpy as np


class CORPUS:
    def __init__(self, title, snippets, embedding):
        self.titles_raw = []           # 原始问句
        self.titles_tok = []           # token化后的的问句
        self.titles_vec = []           # 转化为向量序列后的问句
        self.titles_idf = []           # 句子中每个词的idf
        self.titles_id = []            # 对应序号问句的id
        self.id_snippets = {}          # id对应的代码片段
        self.f = []                    # 列表的每一个元素是一个dict，dict存储着一个文档中每个词的出现次数
        self.df = {}                   # 存储每个词及出现了该词的文档数量
        self.idf = {}                  # 存储每个词的idf值
        self.w2v = {}                  # 词到词向量的字典
        self.embedding_dim = 0         # 词向量维度

        self.titles_count = 0          # 问句的个数
        self.titles_average_len = 0    # 问句平均长度 sum([len(doc) + 0.0 for doc in docs]) / self.D
        self.load_embedding(embedding) # 加载词向量
        self.init(title, snippets)

    def init(self, title, snippets):
        with open(title, "rb") as f:
            p = pickle.load(f)
            self.titles_count = len(p)

            # 打乱数据
            sidx = np.random.permutation(self.titles_count)
            shffled_original_data = [(p[i][0], p[i][1], p[i][2]) for i in sidx]

            for _ in shffled_original_data:
                self.titles_raw.append(_[0].strip().split())
                self.titles_tok.append(utils.remove_stop_words(_[1].strip().split()))
                self.titles_id.append(_[2])

        self.titles_average_len = sum([len(self.titles_tok)+0.0 for doc in self.titles_tok]) / self.titles_count

        with open(snippets, "rb") as f:
            self.id_snippets = pickle.load(f)

        for title in self.titles_tok:
            tmp = {}
            for word in title:
                tmp[word] = tmp.get(word, 0) + 1  # 存储每个文档中每个词的出现次数
            self.f.append(tmp)
            self.titles_vec.append(utils.sent2matrix(self.w2v, self.embedding_dim, title))
            for k in tmp.keys():
                self.df[k] = self.df.get(k, 0) + 1
        for k, v in self.df.items():
            self.idf[k] = math.log(self.titles_count-v+0.5)-math.log(v+0.5)
        for title in self.titles_tok:
            self.titles_idf.append(utils.sentence_idf_vector(self.idf, title))

    def load_embedding(self, text_vectors):
        self.w2v = {}
        with open(text_vectors, "r", encoding='utf-8') as f:
            for _ in f.readlines():
                ws = _.strip().split(' ')
                if len(self.w2v) == 0 and len(ws) > 0:  # 开始插一个0向量
                    v = [0] * len(ws[1:])
                    self.w2v["pad"] = v
                    self.embedding_dim = len(v)
                else:
                    v = [float(w) for w in ws[1:]]
                    self.w2v[ws[0]] = v

    def get_snippets(self, qid):
        return self.id_snippets.get(qid, "")
