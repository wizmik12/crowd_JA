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

figure_of_merit = 'kap2'

def create_setup(X, lengthscale, variance, lr, num_inducing, K):
    model = gpflow.models.SVGP(kernel=gpflow.kernels.SquaredExponential(lengthscales=lengthscale, variance=variance),
                            likelihood=gpflow.likelihoods.MultiClass(K),
                            inducing_variable=kmeans2(X,num_inducing,minit='points')[0],
                            num_latent_gps=K,
                            num_data=len(X))
    optimizer = tf.optimizers.Adam(learning_rate=lr)
    return model, optimizer

def optimization_step(
    model: gpflow.models.SVGP, batch: Tuple[tf.Tensor, tf.Tensor], optimizer: tf.optimizers.Adam
):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.trainable_variables)
        loss = model.training_loss(batch)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def optimization_step(model,Y_cr_mb,Y_mask_mb,idxs_mb):
    scale = N/tf.cast(X_tr_nor_mb.shape[0],float_type) # scale to take into account the minibatch size
    with tf.GradientTape() as tape:   # tf.GradientTape() records the gradients. All the computations for which gradients are required must go in such a block
        q_mb = tf.nn.softmax(tf.gather(q_raw,idxs_mb)) # N_mb,K (constraint: positive and adds to 1 by rows). Gather selects the minibatch
        alpha_tilde = tf.exp(alpha_tilde_raw)  # A,K,K  (cosntraint: positive)
        # Annotations term (term 1 in eq.15 TPAMI)
        expect_log = tf.math.digamma(alpha_tilde)-tf.math.digamma(tf.reduce_sum(alpha_tilde,1,keepdims=True)) # A,K,K
        tnsr_expCrow = tf.gather_nd(expect_log,Y_cr_mb)*tf.cast(Y_mask_mb[:,:,None],float_type) # (N_mb,S,K) = (N_mb,S,K)*(N_mb,S,1)
        annot_term = tf.reduce_sum(tnsr_expCrow*q_mb[:,None,:])*scale # scalar
        # SVGP likelihood term (terms 2 in eq.15 TPAMI)
        f_mean, f_var = model.predict_f(X_tr_nor_mb, full_cov=False, full_output_cov=False) # N,K ; N,K
        liks = [model.likelihood.variational_expectations(f_mean,f_var,
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
    return -negELBO

def crowd_optimization_step(
    model: gpflow.models.SVGP, batch: Tuple[tf.Tensor, tf.Tensor], optimizer: tf.optimizers.Adam
):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.trainable_variables)
        loss = model.training_loss(batch)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def run_adam(train_iter, model, dataset, optimizer, n_epochs, iter_per_epochs, X_tr=None, y_tr=None, X_vl=None, y_vl=None, save_path=None):
    """
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    training_loss = model.training_loss_closure(train_iter, compile=True)
    tf_optimization_step = tf.function(optimization_step)

    train_metrics = {"f1": [], "acc": [], "auc": [], "kap": [], "kap2": []}
    val_metrics = {"f1": [], "acc": [], "auc": [], "kap": [], "kap2": []}

    best_val_metric = 0

    for epoch in range(n_epochs):
        print('    Epoch: {:2d} / {}'.format(epoch+1, n_epochs))
        for step in range(iter_per_epochs):
            tf_optimization_step(model, next(train_iter), optimizer)
            if step % 20 == 0:
                train_step_metrics = evaluate(model, X_tr, y_tr)
                val_step_metrics = evaluate(model, X_vl, y_vl)
                for key in train_metrics.keys():
                    train_metrics[key].append(train_step_metrics[key])
                    val_metrics[key].append(val_step_metrics[key])
                print(step,'/', iter_per_epochs)
                print('ELBO: ', -training_loss().numpy())

                # Save metric
                if val_metrics[figure_of_merit][-1] > best_val_metric:
                    best_val_metric = val_metrics[figure_of_merit][-1]
                    best_params = gpflow.utilities.parameter_dict(model)
                    with open(save_path + 'best_svgp.pickle', 'wb') as handle:
                        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Load best model
    gpflow.utilities.multiple_assign(model, best_params)

    # Save logs
    logs = {'train': train_metrics, 'val': val_metrics}
    with open(save_path + 'logs.pickle', 'wb') as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
