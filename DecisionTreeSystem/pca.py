import numpy as np
import pickle
inf = float(-2**31)


## PCA计算主成分
def pca(X, Y, labels):
    PCA_path = 'pca_data.txt'
    n, m = np.shape(X)
    mean = np.mean(X, axis=0)  # 计算每一列的均值
    mean = np.tile(mean,n).reshape(n,m)
    A = X -mean
            
    # 计算协方差矩阵C
    C = np.matmul(A.T,A) / (n-1)
    
    w, v = np.linalg.eig(C)  # 计算C的特征值w和特征向量v
    w = np.abs(w)
    w_sum = np.sum(w)  # 计算特征值之和
    w1 = w / w_sum  # 计算每个特征值的占比
    
    # 计算累计贡献率得出主成分
    s , p = 0, 0
    while s < 0.8:  # 将前80%的特征作为主成分的特征向量
        s += w1[p]
        p += 1
        
    v1 = v[:,:p-1]  # 主成分的特征向量

    X1 = np.matmul(X,v1)  # 计算主成分
        
    return X1