import gpflow
import tensorflow as tf
import numpy as np
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import gpflow
from scipy.cluster.vq import kmeans2
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score, accuracy_score, classification_report
from typing import Tuple
import data

figure_of_merit = 'kap2'
float_type = tf.float64

def create_setup(X, lengthscale, variance, lr, num_inducing, K):
    model = gpflow.models.SVGP(kernel=gpflow.kernels.SquaredExponential(lengthscales=lengthscale, variance=variance),
                            likelihood=gpflow.likelihoods.MultiClass(K),
                            inducing_variable=kmeans2(X,num_inducing,minit='points')[0],
                            num_latent_gps=K,
                            num_data=len(X))
    optimizer = tf.optimizers.Adam(learning_rate=lr)
    return model, optimizer

def _init_behaviors(probs, Y_cr, Y_mask, A, K):
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

##### Initializations of q_raw and alpha_tilde_raw and alpha (they are _raw because these values must be constrained)
def crowd_setup(Y_cr, Y_mask, q_raw_init, A, K):

    alpha_tilde_raw_init = _init_behaviors(q_raw_init, Y_cr, Y_mask, A, K)
    alpha = tf.ones((A,K,K),dtype=float_type)

    q_raw = tf.Variable(q_raw_init,dtype=float_type)  # N,K
    alpha_tilde_raw = tf.Variable(alpha_tilde_raw_init,dtype=float_type) # A,K,K

    return alpha_tilde_raw, q_raw

A = 7
K = 4
data_loaded_aux = data.load_data()
data_loaded_aux.select('grxcr', 'grxmv')
Y_cr = data_loaded_aux.y_train
Y_cr = Y_cr.astype('int64')
Y_mask = data_loaded_aux.grx.train_mask


alpha = tf.ones((A,K,K),dtype=float_type)

counts_init = np.array([np.bincount(y[m==1,1], minlength=K) for y,m in zip(Y_cr,Y_mask)])
counts_init = counts_init + np.ones(counts_init.shape)
q_raw_init = np.log(counts_init/np.sum(counts_init,axis=1,keepdims=True))

alpha_tilde_raw, q_raw = crowd_setup(Y_cr, Y_mask, q_raw_init, A=7, K=4)
model = gpflow.models.SVGP(kernel=gpflow.kernels.SquaredExponential(lengthscales=2.0, variance=2.0),
                            likelihood=gpflow.likelihoods.MultiClass(K),
                            inducing_variable=kmeans2(data_loaded_aux.X_train,100,minit='points')[0],
                            num_latent_gps=K,
                            num_data=len(data_loaded_aux.X_train))
optimizer = tf.optimizers.Adam(learning_rate=1e-2)
trainable_variables = (alpha_tilde_raw, q_raw) + model.trainable_variables

######################################################################################################################################
def crowd_optimization_step(X_tr, Y_cr_mb,Y_mask_mb,idxs_mb,  N):
    scale = N/tf.cast(X_tr.shape[0],float_type) # scale to take into account the minibatch size
    with tf.GradientTape() as tape:   # tf.GradientTape() records the gradients. All the computations for which gradients are required must go in such a block
        q_mb = tf.nn.softmax(tf.gather(q_raw,idxs_mb)) # N_mb,K (constraint: positive and adds to 1 by rows). Gather selects the minibatch
        alpha_tilde = tf.exp(alpha_tilde_raw)  # A,K,K  (cosntraint: positive)
        # Annotations term (term 1 in eq.15 TPAMI)
        expect_log = tf.math.digamma(alpha_tilde)-tf.math.digamma(tf.reduce_sum(alpha_tilde,1,keepdims=True)) # A,K,K
        tnsr_expCrow = tf.gather_nd(expect_log,Y_cr_mb)*tf.cast(Y_mask_mb[:,:,None],float_type) # (N_mb,S,K) = (N_mb,S,K)*(N_mb,S,1)
        annot_term = tf.reduce_sum(tnsr_expCrow*q_mb[:,None,:])*scale # scalar
        # SVGP likelihood term (terms 2 in eq.15 TPAMI)
        f_mean, f_var = model.predict_f(X_tr, full_cov=False, full_output_cov=False) # N,K ; N,K
        print("AAAAAAAAAAAAAAAAAAAAAA")
        liks = [model.likelihood.variational_expectations(X_tr, f_mean,f_var, c*tf.ones((f_mean.shape[0],1),dtype=tf.int32)) for c in tf.range(K)] # [(N_mb),....,(N_mb)]
        print("BBBBBBBBBBBBBB")
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
    
    print("CCC", annot_term.numpy(), lik_term, entropy_term, KL_svgp_term)
    print(negELBO)
    print(trainable_variables)
    grads = tape.gradient(negELBO,trainable_variables)          # The gradients are obtained
    optimizer.apply_gradients(zip(grads,trainable_variables))   # The gradients are applied with the optimizer
    return -negELBO



def run_adam_cr(aa, optimizer, n_epochs, batch_size, X_tr=None, y_tr=None, Y_mask=None, X_vl=None, y_vl=None, save_path=None):
    """
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param interations: number of iterations
    """
    N = len(X_tr)
    for epoch in range(n_epochs):
        print('    Epoch: {:2d} / {}'.format(epoch+1, n_epochs))
        idxs = np.random.permutation(N)
        idxs_iter = iter(tf.data.Dataset.from_tensor_slices(idxs).batch(batch_size))

        for _, idxs_mb in enumerate(idxs_iter):
            X_tr_mb = tf.gather(X_tr, idxs_mb)     # N_mb,D
            y_tr_mb = tf.cast(tf.gather(y_tr, idxs_mb),tf.int32)            # N_mb,S,2
            Y_mask_mb = tf.gather(Y_mask, idxs_mb)       # N_mb,S
            ELBO = crowd_optimization_step(X_tr_mb, y_tr_mb, Y_mask_mb, idxs_mb, N)
            if _ % 20 == 0:
                train_step_metrics = evaluate(model, X_tr, y_tr)
                val_step_metrics = evaluate(model, X_vl, y_vl)
                for key in train_metrics.keys():
                    train_metrics[key].append(train_step_metrics[key])
                    val_metrics[key].append(val_step_metrics[key])
                print(_,'/', iter_per_epochs)
                print('ELBO: ', -ELBO)

                # Save metric
                if val_metrics[figure_of_merit][-1] > best_val_metric:
                    best_val_metric = val_metrics[figure_of_merit][-1]
                    best_params = gpflow.utilities.parameter_dict(model)
                    with open(save_path + 'best_svgp.pickle', 'wb') as handle:
                        pickle.dump(best_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    cr_params = {'q': q_raw, 'alpha': alpha_tilde_raw}
                    print(cr_params)
                    with open(save_path + 'cr_params.pickle', 'wb') as handle:
                        pickle.dump(cr_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Load best model
    gpflow.utilities.multiple_assign(model, best_params)
    print('Best cr_params: ', cr_params)

    # Save logs
    logs = {'train': train_metrics, 'val': val_metrics}
    with open(save_path + 'logs.pickle', 'wb') as handle:
                pickle.dump(logs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return model, (figure_of_merit, best_val_metric)




####### Functions to predict and evaluate
@tf.function
def predict(model,X_ts_nor):
    y_pred,_ = model.predict_y(X_ts_nor)  # y_pred is N_ts,K (test probabilities, adds 1 by rows)
    return y_pred

def evaluate(model,X_ts_nor,y_test):
    indexes = (y_test!=-1)
    y_test = y_test[indexes]

    y_pred_ = predict(model,X_ts_nor)
    y_pred_ = y_pred_.numpy()[indexes]
    y_pred = np.argmax(y_pred_,axis=1)

    acc = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_test,y_pred, average='macro')
    kap = cohen_kappa_score(y_test, y_pred)
    kap2 = cohen_kappa_score(y_test, y_pred, weights='quadratic')
    row_sums = y_pred_.sum(axis=1)
    y_pred_ = y_pred_ / row_sums[:, np.newaxis]
    auc = roc_auc_score(y_test, y_pred_, average='macro', multi_class='ovr')

    print(classification_report(y_test, y_pred))
    return {'f1':f1,'acc':acc,'auc':auc,'kap':kap,'kap2':kap2}
