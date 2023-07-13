import data
import train_cr as train
import pickle


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tensorflow as tf
#gpu = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(device=gpu[0], enable=True)
import numpy as np
import pandas as pd

# Hyperparameters
FLOAT_TYPE = tf.float64
n_epochs = 20
n_inducings = [32, 64, 128, 256, 512, 1024]
lr = 1e-3
batch_size = 128
K = 4
A = 7

# Experiment
train_names = ['grxcr'] #, 'grxcr']
val_names = ['vlc', 'grxmv', 'grxem']

for n_inducing in n_inducings:
    for val_name in val_names:

        # Save_path
        save_path = 'experiments/{}_{}/{}/'.format(train_names[0],val_name,n_inducing)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Load and select data
        data_loaded = data.load_data()
        data_loaded.select('grxcr', val_name)

        # Train dataset
        X_tr = tf.convert_to_tensor(data_loaded.X_train,dtype=FLOAT_TYPE)
        y_tr = tf.convert_to_tensor(data_loaded.y_train,dtype=FLOAT_TYPE)
        Y_mask = tf.convert_to_tensor(data_loaded.grx.train_mask) #TODO: reconvertir datos e implementar la mask

        # Validation dataset
        X_vl = tf.convert_to_tensor(data_loaded.X_val,dtype=FLOAT_TYPE)
        y_vl= tf.convert_to_tensor(data_loaded.y_val,dtype=FLOAT_TYPE)

        # Model definition and train
        model, optimizer = train.create_setup(X_tr, 2.0, 2.0, lr, n_inducing, K)
        best_model, best_val \
            = train.run_adam_cr(model, optimizer, n_epochs, batch_size, X_tr,
                                y_tr, Y_mask, X_vl, y_vl, save_path)

        print('The best model in val obtained\n ' + best_val[0] + ': ' + str(best_val[1]))

        #############################################
        ##############       TEST       #############
        #############################################
        results_dict = {"data": [], "f1": [], "acc": [], "auc": [], "kap": [], "kap2": []}

        # SICAP test
        X_ts = tf.convert_to_tensor(data_loaded.sicap.X_test,dtype=FLOAT_TYPE)
        y_ts = data_loaded.sicap.y_test
        y_ts= tf.convert_to_tensor(y_ts,dtype=FLOAT_TYPE)
        results = train.evaluate(best_model,X_ts,y_ts)
        print("SICAP:\n", results)
        for key in results.keys():
            results_dict[key].append(results[key])
        results_dict['data'].append('vlc')

        X_ts = tf.convert_to_tensor(data_loaded.grx.X_test,dtype=FLOAT_TYPE)
        y_ts = data_loaded.grx.test_EM
        y_ts= tf.convert_to_tensor(y_ts,dtype=FLOAT_TYPE)
        results = train.evaluate(best_model,X_ts,y_ts)
        print("GRX-EM:\n",results)
        for key in results.keys():
            results_dict[key].append(results[key])
        results_dict['data'].append('GRX-EM')

        y_ts = data_loaded.grx.test_MV
        y_ts= tf.convert_to_tensor(y_ts,dtype=FLOAT_TYPE)
        results = train.evaluate(best_model,X_ts,y_ts)
        print("GRX-MV:\n",results)
        for key in results.keys():
            results_dict[key].append(results[key])
        results_dict['data'].append('GRX-MV')

        y_ts = data_loaded.grx.test_GT
        y_ts= tf.convert_to_tensor(y_ts,dtype=FLOAT_TYPE)
        results = train.evaluate(best_model,X_ts,y_ts)
        print("GRX-GT:\n",results)
        for key in results.keys():
            results_dict[key].append(results[key])
        results_dict['data'].append('GRX-GT')

        for i in range(7):
            y_ts = data_loaded.grx.y_test[:,i]
            y_ts= tf.convert_to_tensor(y_ts,dtype=FLOAT_TYPE)
            results = train.evaluate(best_model,X_ts,y_ts)
            print("Marker" + str(i+1) + ":\n",results)
            for key in results.keys():
                results_dict[key].append(results[key])
            results_dict['data'].append('Marker '+ str(i+1))

        results_df = pd.DataFrame(results_dict)
        results_df.to_csv(save_path + "results.csv")
