import numpy as np
import tensorflow as tf
import argparse
import gpflow
from scipy.cluster.vq import kmeans2
import os
from math import ceil
import pickle
import pdb
import data
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score



###### Preparing TF and GPUs
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

##### Load dataset,normalize,define some variables
# path_data = "/data/BasesDeDatos/AI4SkIN_features/"
# path_data = "/home/jose/crowdsourcing/features/Features_dsmil/"
# path_data = "/home/jose/crowdsourcing/features/Features_random/"

A = 7
K = 4
M = 512
mb_size = 128
n_epochs= 100

# Load and select data
data_loaded = data.load_data()
data_loaded.select('grxcr', 'vlc')
X_tr = data_loaded.X_train
Y_cr = (data_loaded.y_train).astype('int64')
S = Y_cr.shape[1]
Y_mask = data_loaded.grx.train_mask

X_ts = data_loaded.sicap.X_test
z_test = data_loaded.sicap.y_test

X_mean = np.mean(X_tr,axis=0)
X_std = np.std(X_tr,axis=0)
X_std[X_std<=0] = 1
X_tr_nor = (X_tr-X_mean)/X_std
X_ts_nor = (X_ts-X_mean)/X_std
N, x_dim = X_tr_nor.shape
float_type = tf.float64

##### Initializations of q_raw and alpha_tilde_raw and alpha (they are _raw because these values must be constrained)
counts_init = np.array([np.bincount(y[m==1,1], minlength=K) for y,m in zip(Y_cr,Y_mask)])
counts_init = counts_init + np.ones(counts_init.shape)
q_raw_init = np.log(counts_init/np.sum(counts_init,axis=1,keepdims=True))
#q_raw_init = np.log(np.exp(counts_init/np.sum(counts_init,axis=1,keepdims=True))-1.0)
def _init_behaviors(probs, Y_cr, Y_mask):
    alpha_tilde = np.ones((A,K,K))/K
    counts = np.ones((A,K))
    for n in range(len(Y_cr)):
        for a,c,m in zip(Y_cr[n][:,0], Y_cr[n][:,1], Y_mask[n]):
            if m==1:
                alpha_tilde[a,c,:] += probs[n,:]
                counts[a,c] += 1
    alpha_tilde=alpha_tilde/counts[:,:,None]
    alpha_tilde = (counts/np.sum(counts,axis=1,keepdims=True))[:,:,None]*alpha_tilde
    return alpha_tilde/np.sum(alpha_tilde,axis=1,keepdims=True)
# alpha_tilde_raw_init = np.load(path_data+"alpha_tilde_raw_init.npy")  # The initialization of alpha_tilde is as in the TPAMI code (I just saved it in a .npy file to import it easily)
alpha_tilde_raw_init = _init_behaviors(q_raw_init, Y_cr, Y_mask)
#alpha_tilde_raw_init_2 = np.log(np.exp(np.exp(alpha_tilde_raw_init))-1.0)
alpha = tf.ones((A,K,K),dtype=float_type)

###### Variables that will be optimized
q_raw = tf.Variable(q_raw_init,dtype=float_type)  # N,K
alpha_tilde_raw = tf.Variable(alpha_tilde_raw_init,dtype=float_type) # A,K,K
#alpha_tilde_raw = tf.Variable(alpha_tilde_raw_init_2,dtype=float_type) # A,K,K
model = gpflow.models.SVGP(kernel=gpflow.kernels.SquaredExponential(lengthscales=2.0, variance=2.0),
                           likelihood=gpflow.likelihoods.MultiClass(K),
                           inducing_variable=kmeans2(X_tr_nor,M,minit='points')[0],
                           num_latent_gps=K,
                           num_data=N)
trainable_variables = (alpha_tilde_raw,q_raw)+model.trainable_variables # (the latter contains Z,q_mu,q_sqrt and both kernel params)
optimizer = tf.optimizers.Adam(learning_rate=1e-2)

###### This is the optimization step. Basic idea:
######  * We compute the ELBO (the five terms in eq.15 TPAMI), obtain the gradient, and give one step
######  * Gradients are recorded for the computations inside the tf.GradientTape block
######  * The @tf.function decorator makes things faster, since the function is implemented internally as a graph
@tf.function
def optimization_step(X_tr_nor_mb,Y_cr_mb,Y_mask_mb,idxs_mb):
    scale = N/tf.cast(X_tr_nor_mb.shape[0],float_type) # scale to take into account the minibatch size
    with tf.GradientTape() as tape:   # tf.GradientTape() records the gradients. All the computations for which gradients are required must go in such a block
        q_mb = tf.nn.softmax(tf.gather(q_raw,idxs_mb)) # N_mb,K (constraint: positive and adds to 1 by rows). Gather selects the minibatch
        # q_mb = tf.math.softplus(tf.gather(q_raw,idxs_mb)) # N_mb,K (adds to 1 and positive by rows)
        # q_mb = q_mb/tf.reduce_sum(q_mb,axis=1,keepdims=True)
        alpha_tilde = tf.exp(alpha_tilde_raw)  # A,K,K  (cosntraint: positive)
        #alpha_tilde = tf.math.softplus(alpha_tilde_raw)  # A,K,K
        # Annotations term (term 1 in eq.15 TPAMI)
        expect_log = tf.math.digamma(alpha_tilde)-tf.math.digamma(tf.reduce_sum(alpha_tilde,1,keepdims=True)) # A,K,K
        tnsr_expCrow = tf.gather_nd(expect_log,Y_cr_mb)*tf.cast(Y_mask_mb[:,:,None],float_type) # (N_mb,S,K) = (N_mb,S,K)*(N_mb,S,1)
        annot_term = tf.reduce_sum(tnsr_expCrow*q_mb[:,None,:])*scale # scalar
        # SVGP likelihood term (terms 2 in eq.15 TPAMI)
        f_mean, f_var = model.predict_f(X_tr_nor_mb, full_cov=False, full_output_cov=False) # N,K ; N,K
        liks = [model.likelihood.variational_expectations(X_tr_nor_mb,f_mean,f_var,
                                                            c*tf.ones((f_mean.shape[0],1),dtype=tf.int32))
                 for c in np.arange(K)] # [(N_mb),....,(N_mb)]
        lik_term = scale*tf.reduce_sum(q_mb*tf.stack(liks,axis=1))  # 1 <- reduce_sum[(N_mb,K)*(N_mb,K)]
        # Entropy term (term 3 in eq.15 TPAMI)
        entropy_term = -tf.reduce_sum(q_mb*tf.math.log(q_mb))*scale  #scalar
        # KL SVGP term (term 4 in eq.15 TPAMI)
        KL_svgp_term = model.prior_kl()
        # KL annot term (term 5 in eq.15 TPAMI)
        alpha_diff = alpha_tilde-alpha
        KL_annot_term=(tf.reduce_sum(alpha_diff*tf.math.digamma(alpha_tilde))-
                  tf.reduce_sum(tf.math.digamma(tf.reduce_sum(alpha_tilde,1))*tf.reduce_sum(alpha_diff,1))+
                  tf.reduce_sum(tf.math.lbeta(tf.linalg.matrix_transpose(alpha))-tf.math.lbeta(tf.linalg.matrix_transpose(alpha_tilde))))
        negELBO = -(annot_term + lik_term + entropy_term - KL_svgp_term - KL_annot_term)
    grads = tape.gradient(negELBO,trainable_variables)          # The gradients are obtained
    optimizer.apply_gradients(zip(grads,trainable_variables))   # The gradients are applied with the optimizer
    return -negELBO            # The ELBO is returned just to have access to it and print it in the main program (but the minimization takes place in the line above; here we could just return nothing)

####### Functions to predict and evaluate
@tf.function
def predict():
    y_pred,_ = model.predict_y(X_ts_nor)  # y_pred is N_ts,K (test probabilities, adds 1 by rows)
    return y_pred

def evaluate():
    y_pred = predict()
    y_pred = y_pred.numpy()
    y_bin = np.argmax(y_pred,axis=1)
    acc = np.mean(y_bin==z_test)
    f1 = f1_score(z_test,y_bin,average='macro')
    prec = precision_score(z_test, y_bin,average='macro')
    rec = recall_score(z_test, y_bin,average='macro')
    auc = roc_auc_score(z_test, y_pred,average='macro',multi_class='ovr')
    return f1,acc,auc,prec,rec

########## MAIN PROGRAM: Perhaps the best place to start reading
X_tr_nor = tf.convert_to_tensor(X_tr_nor,dtype=float_type)
X_ts_nor = tf.convert_to_tensor(X_ts_nor,dtype=float_type)
Y_cr = tf.convert_to_tensor(Y_cr)
Y_mask = tf.convert_to_tensor(Y_mask)
for ep in range(n_epochs):
    idxs = np.random.permutation(N)
    idxs_iter = iter(tf.data.Dataset.from_tensor_slices(idxs).batch(mb_size))
    for _, idxs_mb in enumerate(idxs_iter):
        X_tr_nor_mb = tf.gather(X_tr_nor,idxs_mb)     # N_mb,D
        Y_cr_mb = tf.cast(tf.gather(Y_cr,idxs_mb),tf.int32)            # N_mb,S,2
        Y_mask_mb = tf.gather(Y_mask,idxs_mb)       # N_mb,S
        ELBO = optimization_step(X_tr_nor_mb,Y_cr_mb,Y_mask_mb,idxs_mb)
        if _%100==0:
            f1,acc,auc,prec,rec = evaluate()
            print("Epoch {} | Iter {} | Test ACC: {} | Test F1: {} | Test auc: {}".format(ep+1,_+1,acc,f1,auc))
with open('state_dict.pickle', 'wb') as f:
    pickle.dump(gpflow.utilities.parameter_dict(model),f)