# -*- coding:utf8 -*-
import tensorflow as tf
import numpy as np
import os
import time
import data_helper
import pickle
from gruRNN import GRURNN
from scipy.stats import pearsonr

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 128, 'the batch_size of the training procedure')
flags.DEFINE_float('lr', 0.00015, 'the learning rate')
flags.DEFINE_integer('max_grad_norm', 5, 'max_grad_norm')
flags.DEFINE_integer('emdedding_dim', 20, 'embedding dim')
flags.DEFINE_integer('hidden_neural_size', 25, 'GRU hidden neural size')
flags.DEFINE_integer('max_len', 25, 'max_len of training sentence')
flags.DEFINE_integer('num_epoch', 1500, 'num epoch')

flags.DEFINE_string('out_dir', os.path.abspath(os.path.join(os.path.curdir, "run_gen")), 'output directory')
flags.DEFINE_integer('check_point_every', 1, 'checkpoint every num epoch ')

# 载入数据集
print("Loading dataset...")

data = data_helper.load_data(max_len=FLAGS.max_len, data_path='./data/train_valid.pickle',
                             embed_dim=FLAGS.emdedding_dim)
test_data = data_helper.load_data(max_len=FLAGS.max_len, data_path='./data/test.pickle', embed_dim=FLAGS.emdedding_dim)
all_data = data_helper.load_all_data(max_len=FLAGS.max_len, data_path='./data/all.pickle', embed_dim=FLAGS.emdedding_dim)


print("length of train set:", len(data[0]))
print("length of test set:", len(test_data[0]))


class Config(object):
    hidden_neural_size = FLAGS.hidden_neural_size
    embed_dim = FLAGS.emdedding_dim
    lr = FLAGS.lr
    batch_size = FLAGS.batch_size
    num_step = FLAGS.max_len
    max_grad_norm = FLAGS.max_grad_norm
    num_epoch = FLAGS.num_epoch
    out_dir = FLAGS.out_dir
    max_len = FLAGS.max_len
    checkpoint_every = FLAGS.check_point_every


def cut_data(data, rate):
    x1, x2, y, mask_x1, mask_x2 = data

    n_samples = len(x1)

    # 打散数据集
    sidx = np.random.permutation(n_samples)

    ntrain = int(np.round(n_samples * (1.0 - rate)))

    train_x1 = [x1[s] for s in sidx[:ntrain]]
    train_x2 = [x2[s] for s in sidx[:ntrain]]
    train_y = [y[s] for s in sidx[:ntrain]]
    train_m1 = [mask_x1[s] for s in sidx[:ntrain]]
    train_m2 = [mask_x2[s] for s in sidx[:ntrain]]

    valid_x1 = [x1[s] for s in sidx[ntrain:]]
    valid_x2 = [x2[s] for s in sidx[ntrain:]]
    valid_y = [y[s] for s in sidx[ntrain:]]
    valid_m1 = [mask_x1[s] for s in sidx[ntrain:]]
    valid_m2 = [mask_x2[s] for s in sidx[ntrain:]]

    # 打散划分好的训练和测试集
    train_data = [train_x1, train_x2, train_y, train_m1, train_m2]
    valid_data = [valid_x1, valid_x2, valid_y, valid_m1, valid_m2]

    return train_data, valid_data


# 验证
def evaluate(model, session, data, config, global_steps=None, summary_writer=None):
    x1, x2, y, mask_x1, mask_x2 = data

    x1 = x1[:config.batch_size]
    x2 = x2[:config.batch_size]
    y = y[:config.batch_size]
    mask_x1 = mask_x1[:config.batch_size]
    mask_x2 = mask_x2[:config.batch_size]

    fetches = [model.mse, model.sim, model.target]
    feed_dict = {}
    feed_dict[model.input_data_s1] = x1
    feed_dict[model.input_data_s2] = x2
    feed_dict[model.target] = y
    feed_dict[model.mask_s1] = mask_x1
    feed_dict[model.mask_s2] = mask_x2

    mse, sim, target = session.run(fetches, feed_dict)

    pearson_r = pearsonr(sim, target)[0]

    dev_summary = tf.summary.scalar(name="dev_pearson_r", tensor=pearson_r)

    dev_summary = session.run(dev_summary)
    if summary_writer:
        summary_writer.add_summary(dev_summary, global_steps)
        summary_writer.flush()
    return mse, pearson_r


def run_epoch(model, session, data, global_steps, valid_model, valid_data, train_summary_writer,
              valid_summary_writer=None):
    for step, (s1, s2, y, mask_s1, mask_s2) in enumerate(data_helper.batch_iter(data, batch_size=FLAGS.batch_size)):
        
        if (len(s1) < FLAGS.batch_size):
            continue

        feed_dict = {}
        feed_dict[model.input_data_s1] = s1
        feed_dict[model.input_data_s2] = s2
        feed_dict[model.target] = y
        feed_dict[model.mask_s1] = mask_s1
        feed_dict[model.mask_s2] = mask_s2
        fetches = [model.mse, model.sim, model.target, model.train_op, model.summary]
        mse, sim, target, _, summary = session.run(fetches, feed_dict)

        pearson_r = pearsonr(sim, target)[0]

        train_summary_writer.add_summary(summary, global_steps)
        train_summary_writer.flush()

        if (global_steps % 100 == 0):
            valid_cost, valid_pearson_r = evaluate(valid_model, session, valid_data, config, global_steps, valid_summary_writer)
            print(
                "the %i step, train cost is: %f, the train pearson_r is %f, the valid cost is %f, the valid pearson_r is %f" % (
                    global_steps, mse, pearson_r, valid_cost, valid_pearson_r))

        global_steps += 1

    return global_steps


def train_step():
    config = Config()
    eval_config = Config()
    eval_config.batch_size = len(test_data[0])
    stat_config = Config()
    stat_config.batch_size = len(all_data[0])

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.initializers.glorot_normal(seed=20000623, dtype=tf.float64)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = GRURNN(config=config, sess=session, is_training=True)

        with tf.variable_scope("model", reuse=True, initializer=initializer):
            valid_model = GRURNN(config=eval_config, sess=session, is_training=False)
            test_model = GRURNN(config=eval_config, sess=session, is_training=False)
            stat_model = GRURNN(config=stat_config, sess=session, is_training=False)

        # 创建摘要
        train_summary_dir = os.path.join(config.out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, session.graph)

        dev_summary_dir = os.path.join(eval_config.out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, session.graph)

        # 创建检查点
        pre_epoch = 0
        global_steps = 1;

        checkpoint_dir = os.path.abspath(os.path.join(config.out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")

        count_rec_dir_name = os.path.abspath(os.path.join(checkpoint_dir, "count_rec.txt"))

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        if ckpt != None:
            saver.restore(session, ckpt)
            # 从count_rec.txt里读取上次训练的global_steps和epoch编号
            with open(count_rec_dir_name, 'r') as f:
                pre_epoch, global_steps = (f.read()).split()
                pre_epoch = int(pre_epoch)
                global_steps = int(global_steps)
            print("Checkpoint restored")
        else:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            tf.global_variables_initializer().run()

        print("Prepare finished")

        begin_time = int(time.time())

        for i in range(pre_epoch, config.num_epoch):
            print("the %d epoch training..." % (i + 1))
            lr = config.lr

            train_data, valid_data = cut_data(data, 0.1)

            global_steps = run_epoch(model, session, train_data, global_steps, valid_model, valid_data,
                                     train_summary_writer, dev_summary_writer)

            if i % config.checkpoint_every == 0:
                path = saver.save(session, checkpoint_prefix, global_steps)
                with open(count_rec_dir_name, 'w') as f:
                    f.write(str(i + 1) + " " + str(global_steps + 1))
                print("Saved model chechpoint to{}\n".format(path))

        print("the train is finished")
        end_time = int(time.time())
        print("training takes %d seconds already\n" % (end_time - begin_time))

        test_cost, test_pearson_r = evaluate(test_model, session, test_data, eval_config)
        print("the test data cost is %f" % test_cost)
        print("the test data pearson_r is %f" % test_pearson_r)

        print("Writing prediction")

        # 原始问句矩阵（模型输入）
        sent_vec = [all_data[1][i] for i in range(stat_config.batch_size)]
        print("total: ", len(sent_vec))

        # 获得问句特征向量（模型输出）
        sent_out = session.run(stat_model.sent1, feed_dict={stat_model.input_data_s1:sent_vec, stat_model.mask_s1:all_data[2]})
        
        # 问句id到问句特征向量的字典
        sent_represent = {}
        for i in range(stat_config.batch_size):
            sent_represent[all_data[0][i]] = sent_out[i]

        # pickle输出文件
        out_file = open('./sent_represent.pickle', 'wb')
        pickle.dump(sent_represent, out_file)
        out_file.close()

        print("Writing done")


        print("program end!")


def main(_):
    train_step()


if __name__ == "__main__":
    tf.app.run()
