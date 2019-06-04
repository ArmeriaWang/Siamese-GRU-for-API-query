# -*- coding:utf8 -*-
import tensorflow as tf


class GRURNN(object):
    def singleRNN(self, x, with_attention, attn_len, scope, cell='gru', reuse=None):
        if cell == 'gru':
            with tf.variable_scope('grucell' + scope, reuse=reuse, dtype=tf.float64):
                used_cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_neural_size,
                                                   reuse=tf.get_variable_scope().reuse,
                                                   kernel_initializer=tf.initializers.glorot_normal(seed=19260817,
                                                                                                    dtype=tf.float64))
                if with_attention:
                    used_cell = tf.contrib.rnn.AttentionCellWrapper(used_cell, attn_length=attn_len, reuse=reuse)
        
        else:
            with tf.variable_scope('lstmcell' + scope, reuse=reuse, dtype=tf.float64):
                used_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_neural_size, forget_bias=1.0, state_is_tuple=True,
                                                         reuse=tf.get_variable_scope().reuse)
                if with_attention:
                    used_cell = tf.contrib.rnn.AttentionCellWrapper(used_cell, attn_length=attn_len, reuse=reuse)

        with tf.variable_scope('cell_init_state' + scope, reuse=reuse, dtype=tf.float64):
            self.cell_init_state = used_cell.zero_state(batch_size=self.batch_size, dtype=tf.float64)

        with tf.name_scope('RNN_' + scope), tf.variable_scope('RNN_' + scope, dtype=tf.float64):
            outs, _ = tf.nn.dynamic_rnn(cell=used_cell, inputs=x, initial_state=self.cell_init_state, time_major=False,
                                        dtype=tf.float64)

        return outs

    def __init__(self, config, sess, is_training=True):
        
        self.lr = config.lr
        self.batch_size = config.batch_size

        num_step = config.num_step
        embed_dim = config.embed_dim
        self.input_data_s1 = tf.placeholder(tf.float64, [self.batch_size, num_step, embed_dim])
        self.input_data_s2 = tf.placeholder(tf.float64, [self.batch_size, num_step, embed_dim])
        self.target = tf.placeholder(tf.float64, [self.batch_size])
        self.mask_s1 = tf.placeholder(tf.float64, [self.batch_size, num_step])
        self.mask_s2 = tf.placeholder(tf.float64, [self.batch_size, num_step])

        self.hidden_neural_size = config.hidden_neural_size
        self.rnn_cell_tpye = 'gru'
        self.rnn_with_attention = False
        # cell_outputs2的reuse为True，表示两个RNN共用一套参数
        with tf.name_scope('gru_output_layer'):
            self.cell_outputs1 = self.singleRNN(x=self.input_data_s1, with_attention=self.rnn_with_attention, 
                                                attn_len=config.max_len, scope='side1',
                                                cell=self.rnn_cell_tpye, reuse=None)
            self.cell_outputs2 = self.singleRNN(x=self.input_data_s2, with_attention=self.rnn_with_attention, 
                                                attn_len=config.max_len, scope='side1',
                                                cell=self.rnn_cell_tpye, reuse=True)

        with tf.name_scope('Sentence_Layer'):
            # 此处得到句子向量，通过调整mask，可以改变句子向量的组成方式
            # 由于mask是用于指示句子的结束位置，所以此处使用sum函数而不是mean函数
            self.sent1 = tf.reduce_sum(self.cell_outputs1 * self.mask_s1[:, :, None], axis=1)
            self.sent2 = tf.reduce_sum(self.cell_outputs2 * self.mask_s2[:, :, None], axis=1)

        with tf.name_scope('loss'):
            diff = tf.abs(tf.subtract(self.sent1, self.sent2), name="err_l1")
            diff = tf.reduce_sum(diff, axis=1)
            # 预测结果被定义为两个RNN末尾输出的差向量的一阶范数的exp
            self.sim = tf.clip_by_value(tf.exp(-1.0 * diff), 1e-7, 1.0 - 1e-7)
            # 损失函数为MSE
            self.mse = tf.reduce_mean(tf.square(tf.subtract(self.sim, self.target)))

        # 加入摘要
        mse_summary = tf.summary.scalar(name="train_mse", tensor=self.mse)

        if not is_training:
            return

        self.globle_step = tf.Variable(0, name="globle_step", trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.mse, tvars), config.max_grad_norm)

        grad_summaries = []
        for g, v in zip(grads, tvars):
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        self.grad_summaries_merged = tf.summary.merge(grad_summaries)
        self.summary = tf.summary.merge([mse_summary, self.grad_summaries_merged])

        optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr, epsilon=1e-6)
        # optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-6)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

        with tf.name_scope('train'):
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
