import numpy as np
import pickle


# 载入数据集
def load_set(data_path, embed_dim):
    with open(data_path, 'rb') as f:
        data_list = pickle.load(f)
    s0 = []
    s1 = []
    labels = []
    for item in data_list:
        s0.append(item[2])
        s1.append(item[4])
        labels.append(float(item[0]))
    return [s0, s1, labels]

def load_all_set(data_path, embed_dim):
    with open(data_path, 'rb') as f:
        data_list = pickle.load(f)
    sent_id = []
    sent_vec = []
    for item in data_list:
        sent_id.append(item[0])
        sent_vec.append(item[1])
    return [sent_id, sent_vec]


# 根据max_len对数据集格式化
def load_data(max_len, data_path, embed_dim):
    data_set = load_set(data_path, embed_dim)

    data_set_x1, data_set_x2, data_set_y = data_set

    n_samples = len(data_set_x1)

    # 打散数据集
    sidx = np.random.permutation(n_samples)

    data_set_x1 = [data_set_x1[s] for s in sidx]
    data_set_x2 = [data_set_x2[s] for s in sidx]
    data_set_y = [data_set_y[s] for s in sidx]

    data_set = [data_set_x1, data_set_x2, data_set_y]

    new_data_set_x1 = np.zeros([n_samples, max_len, embed_dim], dtype=float)
    new_data_set_x2 = np.zeros([n_samples, max_len, embed_dim], dtype=float)
    new_data_set_y = np.zeros([n_samples], dtype=float)

    # mask用于标记句子结束位置
    mask_x1 = np.zeros([n_samples, max_len])
    mask_x2 = np.zeros([n_samples, max_len])

    def padding_and_generate_mask(x1, x2, y, new_x1, new_x2, new_y, mask_x1, mask_x2):
        for i, (x1, x2, y) in enumerate(zip(x1, x2, y)):
            new_y[i] = y
            if len(x1) <= max_len:
                new_x1[i, 0:len(x1)] = x1
                mask_x1[i, len(x1) - 1] = 1
            else:
                new_x1[i, :, :] = (x1[0:maxlen])
                mask_x1[i, max_len - 1] = 1
            if len(x2) <= max_len:
                new_x2[i, 0:len(x2)] = x2
                mask_x2[i, len(x2) - 1] = 1
            else:
                new_x2[i, :, :] = (x2[0:maxlen])
                mask_x2[i, max_len - 1] = 1

        new_set = [new_x1, new_x2, new_y, mask_x1, mask_x2]
        del new_x1, new_x2, new_y
        return new_set

    final_set = padding_and_generate_mask(data_set[0], data_set[1], data_set[2], new_data_set_x1, new_data_set_x2,
                                          new_data_set_y, mask_x1, mask_x2)
    return final_set


def load_all_data(max_len, data_path, embed_dim):
    data_set = load_all_set(data_path, embed_dim)

    n_samples = len(data_set[0])

    new_id = np.zeros([n_samples])
    new_x  = np.zeros([n_samples, max_len, embed_dim], dtype=float)
    # mask用于标记句子结束位置
    mask_x = np.zeros([n_samples, max_len])

    def padding_and_generate_mask(data_id, data_x, new_id, new_x, mask_x):
        for i, (sid, x) in enumerate(zip(data_id, data_x)):
            new_id[i] = sid
            if len(x) <= max_len:
                new_x[i, 0:len(x)] = x
                mask_x[i, len(x) - 1] = 1
            else:
                new_x[i, :, :] = (x[0:maxlen])
                mask_x[i, max_len - 1] = 1

        new_set = [new_id, new_x, mask_x]
        del new_id, new_x, mask_x
        return new_set

    final_set = padding_and_generate_mask(data_set[0], data_set[1], new_id, new_x, mask_x)
    return final_set

# 划分batch
def batch_iter(data, batch_size):
    x1, x2, y, mask_x1, mask_x2 = data
    x1 = np.array(x1)
    x2 = np.array(x2)
    y = np.array(y)
    data_size = len(x1)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for batch_index in range(num_batches_per_epoch):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_size)
        return_x1 = x1[start_index:end_index]
        return_x2 = x2[start_index:end_index]
        return_y = y[start_index:end_index]
        return_mask_x1 = mask_x1[start_index:end_index]
        return_mask_x2 = mask_x2[start_index:end_index]

        yield [return_x1, return_x2, return_y, return_mask_x1, return_mask_x2]
