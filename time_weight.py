import numpy as np
import os


from preprocess import load_sparse, save_sparse
import _pickle as pickle

def normalize_adj(adj):
    s = adj.max(axis=-1, keepdims=True)
    s[s == 0] = 1
    result = adj / s
    return result

if __name__ == '__main__':
    path = 'data/mimic4/standard/train'
    code_x = load_sparse(os.path.join(path, 'code_x.npz'))
    code_num = len(code_x[-1][-1])
    visit_lens = np.load(os.path.join(path, 'visit_lens.npz'))['lens']
    code_x_timestamp = np.load(os.path.join(path, 'code_x_timestamp.npz'), allow_pickle=True)['time']

    for i, code in enumerate(code_x_timestamp):
        for j in range(0, visit_lens[i]):
            if j > 0:
                code_x_timestamp[i][j] = code_x_timestamp[i][j] - code_x_timestamp[i][0]
                code_x_timestamp[i][j] = code_x_timestamp[i][j].days
        code_x_timestamp[i][0] = 0

    x = normalize_adj(code_x_timestamp)
    for person, x_admi in enumerate(x):
        for admi in range(0, visit_lens[person]):
            x[person][admi] = np.exp(x[person][admi])
    x = normalize_adj(x)
    time_weight = np.zeros((len(code_x), code_num, 1))
    for p in range(0, len(code_x)):
        temp = np.where(code_x[p] != 0)
        S = []
        for sc in np.unique(temp[1]):
            sum = x[p][temp[0]][np.where(temp[1] == sc)[0]].sum()
            S.append(sum)
        S_arr = np.array(S)
        S_arr_ave = S_arr / S_arr.max()
        time_weight_temp = np.zeros((code_num, 1))

        for num in range(0, len(S)):
            time_weight_temp[np.unique(temp[1])[num]] = S_arr_ave[num]
        time_weight[p] = time_weight_temp
    save_sparse(os.path.join(os.path.dirname(path), 'time_weight'), time_weight)

