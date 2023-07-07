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
