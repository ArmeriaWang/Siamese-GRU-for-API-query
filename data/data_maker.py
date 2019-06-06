import heapq
import pickle
import utils
import time
import numpy as np
from corpus import CORPUS
from itertools import groupby
import random
import time

class BIKER:
    def __init__(self, corp):
        self.corpus = corp

    # 计算p,q的相似度
    def sim(self, p, q,  idf_p, idf_q):
        sim12 = (idf_p * (abs(p.dot(q.T).max(axis=1)))).sum() / idf_p.sum()
        sim21 = (idf_q * (abs(q.dot(p.T).max(axis=1)))).sum() / idf_q.sum()
        return 2 * sim12 * sim21 / (sim12 + sim21)

# 构造数据集，lower、upper为采集范围，range_cnt为区间采集上限
def make_data(lower, upper, range_cnt, data_path_name):
    eps = 1e-6
    cur_count = upper - lower + 1

    scores = []                # 数据集（元组的列表），
                               # 每个元组第零维为相似度值
                               # 第一维为问题1编号，第二维为问题1向量矩阵
                               # 第三维为问题2编号，第四维为问题2向量矩阵
    scores_val = []            # 相似度值的列表，用于统计
    scores_num = []            # 每0.2区间的采集上限
    scores_cnt = []            # 每0.2区间的当前采集数
    scores_stat = []           # 每0.1区间的当前采集数


    scores_num = [range_cnt, range_cnt, range_cnt, range_cnt, range_cnt]
    scores_cnt = [0 for n in range(10)]
    scores_stat = [0 for n in range(15)] 
    scores_val = []
    ids = [i for i in range(lower, upper)]

    # 采集方式：
    # 扫描采集范围内所有问句，对每个问句A都在采集范围内随机再抽取一个问句B与之配对
    # 计算A和B的相似度，若对应0.2区间尚未采集完成，则将其加入数据集，否则跳过
    while sum(scores_cnt) < sum(scores_num):
        for id1 in range(lower, upper):

            # 输出当前采集进度
            if (id1 %  ((upper - lower + 1) / 3) == 0):
                print(scores_cnt[:5])
                print("total =", sum(scores_cnt))
                print("current:", int(sum(scores_cnt)/(sum(scores_num)*1.0)*100), "%\n")

            if sum(scores_cnt) == sum(scores_num):
                break

            # 随机抽取一个问句id2与id1配对
            for id2 in random.sample(ids, 1):
                score = biker.sim(biker.corpus.titles_vec[id1], biker.corpus.titles_vec[id2], 
                    biker.corpus.titles_idf[id1], biker.corpus.titles_idf[id2])
                sid = int((score - eps) / 0.2)
                stid = int((score - eps) / 0.1)

                # 若对应区间采集未完成，则加入数据集
                if scores_cnt[sid] < scores_num[sid]:
                    scores_cnt[sid] += 1
                    scores_stat[stid] += 1
                    scores_val.append(score)
                    scores.append((score, 
                        biker.corpus.titles_id[id1], biker.corpus.titles_vec[id1],
                        biker.corpus.titles_id[id2], biker.corpus.titles_vec[id2]))
    print("Creating done")

    # 打乱数据集
    sidx = np.random.permutation(range_cnt*5)
    new_scores = [(scores[i][0], scores[i][1], scores[i][2], scores[i][3], scores[i][4]) for i in sidx]
    print("Shuffling done")

    # 写入pickle文件
    score_file = open(data_path_name, 'wb')
    pickle.dump(new_scores, score_file)
    score_file.close()
    print("Writing done")

    print("=" * 35)

    # 以0.1为跨度输出采集统计结果
    for i in range(10):
        print("%.1f ~ %.1f :: %d, %.1f%%" % (i*0.1, (i+1)*0.1, scores_stat[i], scores_stat[i]/(range_cnt*5.0)*100))
    print("total created:", len(scores), "\n")

if __name__ == '__main__':

    random.seed(time.time())

    print("Loading original data ...")
    corpus = CORPUS("corpus/title_token_id.pickle", "corpus/id_to_code.pickle", "corpus/title_w2v")
    biker = BIKER(corpus)
    print("Loading done", "\n")

    total = biker.corpus.titles_count
    print(total)

    # 构造训练&验证集
    make_data(lower=0, upper=120000, range_cnt=10000, data_path_name="./train_valid.pickle")

    # 构造测试集
    make_data(lower=120000, upper=150000, range_cnt=2000, data_path_name="./test.pickle")

    print("Making all_data...")
    all_data = [(biker.corpus.titles_id[i], biker.corpus.titles_vec[i]) for i in range(total)]
    all_file = open("./all.pickle", 'wb')
    pickle.dump(all_data, all_file)
    all_file.close()

    print("ALL Data making done")
