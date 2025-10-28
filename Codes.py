# -*- coding: utf-8 -*-
import os
import numpy as np
import torch

def get_convolutional_generator_matrix(code, SubMatrixList):
    k = code.k
    n = code.n
    l = code.l
    block_length = code.blk_len

    generator_matrix = np.zeros((block_length * k, block_length * n))
    for i in range(block_length):
        for j in range(block_length):
            if 0 <= j - i <= l - 1:
                generator_matrix[i * k:(i + 1) * k, j * n:(j + 1) * n] = SubMatrixList[j - i]

    return generator_matrix

def get_convolutional_parity_matrix(code, SubMatrixList):
    k = code.k
    n = code.n
    l = code.l
    block_length = code.blk_len

    P_submatrices = []
    for i in range(l):
        P_i = SubMatrixList[i][:, k:]
        Trans = P_i.transpose()
        zero_matrix = np.zeros((n - k, n - k)) if i != 0 else np.eye(n - k)
        Sub_Check = np.concatenate((Trans, zero_matrix), axis=1)
        P_submatrices.append(Sub_Check)

    parity_matrix = np.zeros((block_length * (n - k), block_length * n))
    for i in range(block_length):
        for j in range(block_length):
            if 0 <= i - j <= l - 1:
                parity_matrix[i * (n - k):(i + 1) * (n - k), j * n:(j + 1) * n] = P_submatrices[i - j]

    return parity_matrix

def get_generator_and_parity(code):
    n, k, l, b = code.n, code.k, code.l, code.blk_len
    path = os.path.join('Codes_DB', f'Conv_N{n}_K{k}_L{l}.txt')
    try:
        Sub = np.loadtxt(path)
        print(Sub)
        # 将 Sub 分割成子矩阵列表
        SubMatrixList = [Sub[i * k:(i + 1) * k] for i in range(l)]
        G = get_convolutional_generator_matrix(code, SubMatrixList)
        H = get_convolutional_parity_matrix(code, SubMatrixList)
        return G.astype(float), H.astype(float)
    except FileNotFoundError:
        print(f"File {path} inaccessible.")
        return None, None


if __name__ == "__main__":
    np.set_printoptions(threshold = np.inf)
    class Code():
        pass
    code = Code()

    code.n = 3              # 分组编码长度
    code.k = 2              # 信息位长度
    code.l = 3              # 约束长度
    code.blk_len = 5        # 块长度

    G0 = np.array([[1, 0, 1],
                   [0, 1, 1]])
    G1 = np.array([[0, 0, 0],
                   [0, 0, 1]])
    G2 = np.array([[0, 0, 1],
                   [0, 0, 0]])

    # 计算生成矩阵
    generator_matrix = get_convolutional_generator_matrix(code,[G0, G1, G2])
    print("Generator Matrix:")
    print(generator_matrix)

    parity_matrix = get_convolutional_parity_matrix(code,[G0, G1, G2])
    print("Parity Matrix:")
    print(parity_matrix)

    assert np.all(np.mod(np.matmul(generator_matrix, parity_matrix.transpose()), 2)) == 0

    G, H = get_generator_and_parity(code)
    print(G)
    print(H)
