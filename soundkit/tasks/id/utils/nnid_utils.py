"""
Correlation function for speaker verification
"""
import numpy as np
import tensorflow as tf

def cross_entropy_nnid(
        target,       # trget
        estimation,   # estimator
        eps = tf.constant(10**-5, dtype = tf.float32)): # epsilon
    """
    Cross entropy loss for nnid
    """
    batchsize,_ = target.shape
    loss = -tf.reduce_sum(
        target * tf.math.log(tf.math.maximum(estimation, eps)) )
    steps = batchsize
    ave_loss = loss / steps
    return ave_loss, steps

def get_corr(
        feats,
        num_group_ppls,
        num_prons,
        eps = 10**-5):
    """
    Correlation function
    """
    nfeats = feats / (np.sqrt(np.sum(feats**2, axis=-1, keepdims=True)) + eps)
    mat_corr = np.zeros((num_group_ppls*num_prons,num_group_ppls), dtype=np.float32)
    for i in range(num_group_ppls):
        center = np.sum(nfeats[i*num_prons:num_prons * (i+1),:], axis=0)
        # print(center)
        for j in range(num_group_ppls):
            for k in range(num_prons):
                nfeat = nfeats[k + j * num_prons,:]
                if i == j:
                    center0 = (center - nfeat) / (num_prons - 1)
                else:
                    center0 = center / num_prons
                norm_center0 = np.sum(center0**2)

                corr = np.matmul(nfeat, center0)
                corr = corr / (np.sqrt(norm_center0) + eps)
                mat_corr[k + j * num_prons, i] = corr
    return mat_corr

def get_corr_fast(
        feats,
        targets,
        num_group_ppls,
        num_prons,
        eps = 10**-5):
    """
    Fast Correlation function
    """
    nfeats = feats / tf.maximum(
                tf.sqrt(tf.reduce_sum(feats**2, axis=-1, keepdims=True)),
                eps)
    centers = tf.matmul(tf.transpose(nfeats),targets)

    norm2_feat = tf.identity(targets)

    mat_corr = tf.matmul(nfeats, centers)
    mat_corr = mat_corr - norm2_feat

    norm2_centers = tf.reduce_sum(centers**2, axis=0, keepdims=True)

    norm2_centers = tf.tile(
        norm2_centers,
        [num_group_ppls * num_prons,1]) * (1 - targets)

    ncenters_targets = tf.transpose(tf.matmul(centers, tf.transpose(targets)))
    norm2_centers_targets = tf.reduce_sum((ncenters_targets - nfeats)**2, axis=-1, keepdims=True)

    norm2_centers_targets = tf.tile(
        norm2_centers_targets,
        [1,num_group_ppls]) * targets
    norm2_centers = norm2_centers + norm2_centers_targets
    mat_corr = mat_corr / (tf.sqrt(norm2_centers) + eps)

    return mat_corr

def gen_target_nnid(num_group_ppls, num_prons):
    """ 
    Generate diag for nnid
    """
    return tf.constant(np.kron(np.eye(num_group_ppls), np.ones((num_prons,1))), dtype=np.float32)

if __name__=="__main__":
    NUM_GROUP_PPLS  = 3
    NUM_PRONS       = 4
    DIM_FEAT        = 6
    EPS             = 0
    TARGET = gen_target_nnid(NUM_GROUP_PPLS, NUM_PRONS)
    # print(diag)

    FEATS = tf.constant(
        np.random.randint(1,100, size = (NUM_GROUP_PPLS * NUM_PRONS, DIM_FEAT)),
        dtype=np.float32)

    mat = get_corr_fast(
        FEATS,
        TARGET,
        NUM_GROUP_PPLS,
        NUM_PRONS,
        eps = EPS).numpy()

    mat_ref = get_corr(
        FEATS.numpy(),
        NUM_GROUP_PPLS,
        NUM_PRONS,
        eps=EPS)

    print("corr:")
    print(mat)
    print("corr: ref")
    print(mat_ref)

    print("err")
    print(mat-mat_ref)
    print(np.abs(mat-mat_ref).max())
