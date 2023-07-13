import pickle
import numpy as np

def unpack_data(data):
    return data['features'], data['label_list'], data['names']

def unpack_data_grx_train(data):
    return data['features'], np.array(data['label_list']), data['names'], data['MV'], data['EM']

def unpack_data_grx_test(data):
    return data['features'], np.array(data['label_list']), data['names'], data['MV'], data['EM'], data['GT']

def process_labels_cr(labels):
    labels_cr = np.zeros((labels.shape[0], labels.shape[1], 2))
    labels_mask = np.zeros((labels.shape[0], labels.shape[1]))
    for ix, x in enumerate(labels):
        for iy, y in enumerate(x):
            if y != -1:
                labels_cr[ix, iy, 0] = iy   
                labels_cr[ix, iy, 1] = y
                labels_mask[ix, iy] = 1
            else:
                labels_cr[ix, iy, 0] = -1   
                labels_cr[ix, iy, 1] = -1
                labels_mask[ix, iy] = 0
    return labels_cr, labels_mask

class load_data:
    def __init__(self):
        print("Load data")

    def select(self, train, val):
        self.sicap = SICAP_data()
        self.grx = grx_data()

        if train == 'grxem':
            self.X_train = self.grx.X_train
            self.y_train = self.grx.train_EM
        elif train == 'grxmv':
            self.X_train = self.grx.X_train
            self.y_train = self.grx.train_MV
        elif train == 'grxcr':
            self.X_train = self.grx.X_train
            self.y_train = self.grx.y_train
        elif train == 'vlc':
            self.X_train = self.sicap.X_train
            self.y_train = self.sicap.y_train
        else:
            raise ("Set not found")

        # Validation
        if val == 'grxem':
            self.X_val = self.grx.X_val
            self.y_val = self.grx.val_EM
        elif val == 'grxmv':
            self.X_val = self.grx.X_val
            self.y_val = self.grx.val_MV
        elif val == 'grxcr':
            raise NotImplementedError
        elif val == 'vlc':
            self.X_val = self.sicap.X_val
            self.y_val = self.sicap.y_val
        
        self.m, self.std = self.X_train.mean(0), self.X_train.std(0)

        print("Train: ", train, self.X_train.shape, "\n Val: ", val, self.X_val.shape)

        self.X_train = self._norm(self.X_train)
        self.X_val = self._norm(self.X_val)

        self.sicap.X_test = self._norm(self.sicap.X_test)
        self.grx.X_test = self._norm(self.grx.X_test)

    def _norm(self, data):
        return (data-self.m)/self.std
        
class SICAP_data:
    def __init__(self):

        # Normalize

        # load data
        with open('../feat_extraction/features/sicap.pickle', 'rb') as fp:
            sicap_data = pickle.load(fp)

        # training
        print(sicap_data.keys())
        train = sicap_data['train']
        self.X_train, self.y_train, self.train_names = unpack_data(train)

        # validation
        val = sicap_data['val']
        self.X_val, self.y_val, self.val_names = unpack_data(val)

        # test
        test = sicap_data['test']
        self.X_test, self.y_test, self.test_names = unpack_data(test)

        print('Train: ', self.X_train.shape)
        print('Val: ', self.X_val.shape)
        print('Test: ', self.X_test.shape)


class grx_data:
    def __init__(self):

        # load data
        with open('../feat_extraction/features/grx.pickle', 'rb') as fp:
            grx_data = pickle.load(fp)

        # training
        print(grx_data.keys())
        train = grx_data['train']
        self.X_train, self.y_train, self.train_names, \
            self.train_MV, self.train_EM  = unpack_data_grx_train(train)
        
        self.y_train, self.train_mask = process_labels_cr(self.y_train)

        # validation
        val = grx_data['val']
        self.X_val, self.y_val, self.val_names, \
            self.val_MV, self.val_EM = unpack_data_grx_train(val)

        # test
        test = grx_data['test']
        self.X_test, self.y_test, self.test_names, \
        self.test_MV, self.test_EM, self.test_GT = unpack_data_grx_test(test)



        # self.y_test_crowd = [x[:,i] for i in range(7) ]

        print('Train: ', self.X_train.shape)
        print('Val: ', self.X_val.shape)
        print('Test: ', self.X_test.shape)