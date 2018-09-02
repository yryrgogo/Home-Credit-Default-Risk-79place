epoch_size = 10
import os
import datetime
import sys

sys.path.append('../../../github/module/')
from preprocessing import set_validation, split_dataset, factorize_categoricals, get_dummies, data_regulize, inf_replace
from load_data import pararell_load_data, x_y_split
from convinience_function import get_categorical_features, get_numeric_features
from logger import logger_func
from make_file import make_feature_set, make_npy, make_raw_feature
import time
import warnings
from contextlib import contextmanager
import pandas as pd
import numpy as np
from scipy import sparse as ssp
import pylab as plt
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, NMF, PCA, FactorAnalysis
from sklearn.feature_selection import SelectFromModel, SelectPercentile, f_classif
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.cross_validation import StratifiedKFold, KFold
import tensorflow as tf
from keras.layers import Input, BatchNormalization, LeakyReLU
from keras.layers import Layer, Dense, Activation, AlphaDropout, GaussianNoise
from keras.models import Model
from keras.optimizers import SGD
from keras import backend as K
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Embedding, LSTM, Dense, Flatten, merge, Convolution1D, MaxPooling1D, Lambda, AveragePooling1D
from keras.layers import Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Concatenate
from sklearn.preprocessing import MinMaxScaler


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s\n".format(title, time.time() - t0))


def as_keras_metric(method):
        import functools
        from keras import backend as K
        import tensorflow as tf
        @functools.wraps(method)
        def wrapper(self, args, **kwargs):
            """ Wrapper for turning tensorflow metrics into keras metrics """
            value, update_op = method(self, args, **kwargs)
            K.get_session().run(tf.local_variables_initializer())
            with tf.control_dependencies([update_op]):
                value = tf.identity(value)
            return value
        return wrapper
auc_roc = as_keras_metric(tf.metrics.auc)


start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
logger = logger_func()

start_time = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
' データセットからそのまま使用する特徴量 '
unique_id = 'SK_ID_CURR'
target = 'TARGET'
ignore_features = [unique_id, target, 'valid_no', 'valid_no_2',
                   'valid_no_3', 'valid_no_4', 'is_train', 'is_test']
val_col = 'valid_no_4'

# Parameter
seed = 1208
np.random.seed(seed)
#  hidden = 64
hidden = 100

' DR P '
W_regularizer = None
activation = 'relu'
batch_norm = True
batch_size = 128
dropout_proba = 0.75
early_stopping_patience = 10
early_stopping_window = 5
embedding_hidden_units = 100
embedding_regularizer = None
embedding_size = 10
#  epoch_size = 10**9
learning_rate = 'adaptive'
learning_rate_init = 0.001
momentum = 0.9
n_hidden_units = 128
n_iter = 2000
n_layers = 2
optimizer = 'adam'
power_t = 0.5


' dropout_probaはどう使うんだ？→flatten? '
' flatten '
#  activation = 'relu'
#  model.add(Flatten())
#  # dropout probability: tensorflow: percent=proba_keep, keras=p1-proba_keep
#  params = {'activation': activation, 'batchnorm': True, 'dropout_proba': 0.2}
#  model = create_dense(model, layer_name='dense_1', num_neurons=100, **params)
#  model = create_dense(model, layer_name='dense_2', num_neurons=50,  **params)
#  model = create_dense(model, layer_name='dense_3', num_neurons=10,  **params)
#  # output layer: number of neurons=1; no softmax layer is used in last FC layer
#  model.add(Dense(1, name='dense_ouput'))
#  # compile model: commai used defaults
#  opt = Adam(lr=learning_rate)
#  model.compile(optimizer="adam", loss=loss_metric, metrics=['accuracy'])


def x_y_split(data, target):
    x = data.drop(target, axis=1)
    y = data[target].values
    return x, y


class AucCallback(Callback):  # inherits from Callback

    def __init__(self, validation_data=(), patience=20, is_regression=True, best_model_name='best_keras.mdl', feval='roc_auc_score', batch_size=1024*8):
        super(Callback, self).__init__()

        self.patience = patience
        self.X_val, self.y_val = validation_data  # tuple of validation X and y
        self.best = -np.inf
        self.wait = 0  # counter for patience
        self.best_model = None
        self.best_model_name = best_model_name
        self.is_regression = is_regression
        self.y_val = self.y_val  # .astype(np.int)
        self.feval = feval
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs={}):
        p = self.model.predict(
            self.X_val, batch_size=self.batch_size, verbose=0)  # .ravel()
        if self.feval == 'roc_auc_score':
            current = roc_auc_score(self.y_val, p)

        if current > self.best:
            self.best = current
            self.wait = 0
            self.model.save_weights(self.best_model_name, overwrite=True)

        else:
            if self.wait >= self.patience:
                self.model.stop_training = True
                print('Epoch %05d: early stopping' % (epoch))

            self.wait += 1  # incremental the number of times without improvement
        print('Epoch %d Auc: %f | Best Auc: %f \n' %
              (epoch, current, self.best))


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]


def keras_classifier(data):

    data = data.query('is_train==1')
    #  data.drop('is_train', axis=1, inplace=True)
    columns = data.columns.tolist()
    columns = [col for col in columns if not(
        col.count('is_train')) and not(col.count('is_test'))]
    data = data[columns]
    flatten_layers = []
    inputs = []
    use_cols = []
    for c in columns:
        if c in ignore_features:
            continue
        use_cols.append(c)

        #  inputs_c = Input(shape=(1,), dtype='int32')
        inputs_c = Input(shape=(1,), dtype='float16')

        num_c = len(np.unique(data[c].values))

        embed_c = Embedding(
            num_c,
            dim,
            dropout=0.2,
            input_length=1,
            embeddings_regularizer=None
        )(inputs_c)
        flatten_c = Flatten()(embed_c)

        inputs.append(inputs_c)
        flatten_layers.append(flatten_c)

    #  flatten = merge(flatten_layers, mode='concat')
    flatten = Concatenate()(flatten_layers)

    fc1 = Dense(hidden, activation='relu')(flatten)
    dp1 = Dropout(0.5)(fc1)

    outputs = Dense(1, activation='sigmoid')(dp1)

    model = Model(input=inputs, output=outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
    )
    #  del data

    logger.info(f'create dataset start.')
#     train, valid = split_dataset(data, val_no, val_col=val_col)
    train = data.query("valid_no_4 != 2")
    valid = data.query("valid_no_4 == 2")
    x_train, y_train = x_y_split(train, target)
    x_val, y_val = x_y_split(valid, target)

#     use_cols = [col for col in x_train.columns if col not in ignore_features]

    x_train = x_train[use_cols].values
    x_train = [x_train[:, i] for i in range(x_train.shape[1])]
    x_val = x_val[use_cols].values
    x_val = [x_val[:, i] for i in range(x_val.shape[1])]

    logger.info(f'create dataset end.')

    model_name = 'mlp_residual_%s_%s.hdf5' % (dim, hidden)
    model_checkpoint = ModelCheckpoint(
        model_name, monitor='val_loss', save_best_only=True)
    auc_callback = AucCallback(validation_data=(x_val, y_val), patience=2,
                               is_regression=True, best_model_name=path+'best_keras.mdl', feval='roc_auc_score')

    nb_epoch = 10
    #  nb_epoch = 10**9

    #  batch_size = 1024*8
    batch_size = 128
    load_model = False

    if load_model:
        print('Load Model')
        model.load_weights(path+model_name)
        # model.load_weights(path+'best_keras.mdl')

    # Early-stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    logger.info(f'model fitting start.')
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        nb_epoch=nb_epoch,
        verbose=1,
        shuffle=True,
        validation_data=[x_val, y_val],
        callbacks=[early_stopping]
        # callbacks = [
        # model_checkpoint,
        # auc_callback,
        # ],
    )

    # model.load_weights(model_name)
    # model.load_weights(path+'best_keras.mdl')

    logger.info(f'prediction start')
    #  y_preds = model.predict(X_test, batch_size=1024*8)
    y_preds = model.predict(x_val, batch_size=batch_size)
    sc_score = roc_auc_score(y_val, y_preds)
    logger.info(f'auc: {sc_score}')

    return y_preds
    #  sys.exit()

    # print('Make submission')
    #  X_t = [X_t[:, i] for i in range(X_t.shape[1])]
    #  outcome = model.predict(X_t, batch_size=1024*8)
    #  outcome = model.predict(X_t, batch_size=128)
    #  submission = pd.DataFrame()
    #  submission['activity_id'] = activity_id
    #  submission['outcome'] = outcome
    #  submission.to_csv('submission_residual_%s_%s.csv' %
    #                    (dim, hidden), index=False)


def redhat_keras(x_train, y_train, x_val, y_val, x_test, dim):

    # モデル定義
    model = Sequential()

    Embedding
    model.add(Embedding(
        output_dim = embedding_size,
        input_dim = dim,
        input_length=dim,
        dropout=0.2,
        embeddings_regularizer=None
    ))
    model.add(Dropout(0.1))

    model.add(Dense(128, activation='relu', input_shape=(dim,)))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[auc_roc])

    model.fit(x_train,
              y_train,
              epochs=epoch_size,
              batch_size=batch_size,
              validation_data=(x_val, y_val)
              )
    y_pred = model.predict(x_val, batch_size=batch_size)
    sc_score = roc_auc_score(y_val, y_preds)
    logger.info(f'auc: {sc_score}')
    test_pred = model.predict(x_test, batch_size=2048)

    return y_pred, sc_score, test_pred


def mlp_clf(x_train, y_train):
    ' sklearn '
    from sklearn.neural_network import MLPClassifier
    params = {
        'hidden_layer_sizes': 128,
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001,
        'batch_size': 128,
        'learning_rate': 'adaptive',
        'learning_rate_init': 0.001,
        'power_t': 0.5,
        'max_iter': 2000,
        'shuffle': True,
        'random_state': 1208,
        'tol': 0.0001,
        'verbose': False,
        'warm_start': False,
        'momentum': 0.9,
        'nesterovs_momentum': True,
        'early_stopping': True,
        'validation_fraction': 0.2,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'epsilon': 1e-08
    }

    clf = MLPClassifier(**params)
    clf.fit(x_train, y_train)
    logger.info(f'auc: {clf.score(x_val, y_val)}')


def kernel_keras(x_train, y_train, x_val, y_val, x_test, dim):

    print('Setting up neural network...')
    nn = Sequential()
    nn.add(Dense(units=400, kernel_initializer='normal', input_dim=dim))
    nn.add(PReLU())
    nn.add(Dropout(.3))
    nn.add(Dense(units=160, kernel_initializer='normal'))
    nn.add(PReLU())
    nn.add(BatchNormalization())
    nn.add(Dropout(.3))
    nn.add(Dense(units=64, kernel_initializer='normal'))
    nn.add(PReLU())
    nn.add(BatchNormalization())
    nn.add(Dropout(.3))
    nn.add(Dense(units=26, kernel_initializer='normal'))
    nn.add(PReLU())
    nn.add(BatchNormalization())
    nn.add(Dropout(.3))
    nn.add(Dense(units=12, kernel_initializer='normal'))
    nn.add(PReLU())
    nn.add(BatchNormalization())
    nn.add(Dropout(.3))
    nn.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    #  nn.compile(loss='binary_crossentropy', optimizer='adam')
    nn.compile(loss='binary_crossentropy', optimizer='adam' , metrics=[auc_roc])

    # Early-stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    print('Fitting neural network...')
    nn.fit(x_train,
           y_train,
           epochs=epoch_size,
           batch_size=batch_size,
           verbose=2,
           callbacks=[early_stopping],
           validation_data=(x_val, y_val)
           )

    print('Predicting...')
    #  y_pred = nn.predict(x_val).flatten().clip(0, 1)
    y_pred = nn.predict(x_val, batch_size=1024)
    #  test_pred = nn.predict(x_test).flatten().clip(0, 1)
    #  test_pred = nn.predict(x_test)

    sc_score = roc_auc_score(y_val, y_pred)
    logger.info(f'auc: {sc_score}')

    #  return y_pred, sc_score, test_pred
    return y_pred, sc_score, 0



def dense_bn_block(x, size, act='elu'):  # LeakyReLU(alpha=0.01)):
    x = Dense(size, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = AlphaDropout(0.3)(x)
    if isinstance(act, str):
        z = Activation(act)(x)
    else:
        z = act(x)
    return z


def build_classifier(input_shape):
    model_input = Input(shape=input_shape)

    model = GaussianNoise(stddev=0.001)(model_input)

    model = dense_bn_block(model, 1024)
    model = dense_bn_block(model, 1024)
    model = dense_bn_block(model, 1024)
    model = dense_bn_block(model, 256)
    model = dense_bn_block(model, 32)

    model_output = Dense(1, activation='sigmoid')(model)
    return Model(inputs=[model_input], outputs=[model_output])


def keras_ANN(x_train, y_train, x_val, y_val, x_test, dim):

    with timer("calculating class weights"):
        from sklearn.utils import class_weight

        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
        print('sklearn class weights: {}'.format(class_weights))


    with timer("creating classifier"):
        input_shape = (x_train.shape[-1],)
        model = build_classifier(input_shape)

    # saved model variables
    best_model_path = './best_model.h5'

    # auc variables
    best_auc = 0

    min_delta = 0.0003

    # early stop variables
    max_es_rounds = 10
    current_es_round = 0

    # reduce lr variables
    max_lr_rounds = 4
    current_lr_round = 0
    reduce_lr_rate = (1 / 3)
    current_lr = 1.0

    # increase batch size
    max_bs_rounds = 6
    current_bs_round = 0
    increase_bs_rate = 2
    current_bs = 256
    max_bs = 2048

    current_epoch = 1
    while True:
        with timer("epoch #{}".format(current_epoch)):
            # epochs are random stratified minibatch
            sgd = SGD(lr=current_lr, momentum=0.85, decay=0.0, nesterov=True)
            auc_roc = as_keras_metric(tf.metrics.auc)
            model.compile(
                optimizer=sgd, loss='binary_crossentropy', metrics=[auc_roc])
            print('current lr: {}'.format(current_lr))

            # train model
            model.fit(x_train, y_train, batch_size=current_bs,
                      epochs=1, class_weight=class_weights)

            # get validation score
            #  y_pred = model.predict(x_val, batch_size=2048)
            ' 2次元配列になってるのでflattenする '
            y_pred = model.predict(x_val, batch_size=2048).flatten().clip(0, 1)

            try:
                auc = roc_auc_score(y_val, y_pred)
            except ValueError:
                logger.info(f'y_val: {y_val}')
                logger.info(f'y_pred: {y_pred}')
                logger.info(f'{y_pred[y_pred==np.nan]}')
                logger.info('ValueError')
            logger.info(f'auc: {auc}')

            # save model and check early stop
            if (auc - best_auc) > min_delta:
                print('validation ROC_AUC improved from {} to {}!'.format(
                    best_auc, auc))
                model.save(best_model_path)
                print('model_saved to {}'.format(best_model_path))

                # deal with es variables
                current_es_round = 0

                # deal with lr variables
                current_lr_round = 0

                # deal with bs variables
                current_bs_round = 0

                best_auc = auc
            else:
                current_es_round += 1
                current_lr_round += 1
                current_bs_round += 1
                print('validation ROC_AUC did not improve from {} for {} rounds'.format(
                    best_auc, current_es_round))

            # stop if potato
            if current_es_round >= max_es_rounds:
                print('Stopping due to validation AUC_ROC not increasing for {} rounds'.format(
                    max_es_rounds))
                break

            # reduce lr if potato
            if current_lr_round >= max_lr_rounds:
                print('reducing learning rate from {} to {}'.format(
                    current_lr, current_lr * reduce_lr_rate))
                current_lr *= reduce_lr_rate
                current_lr_round = 0

            # increase batch size if potato
            if current_bs_round >= max_bs_rounds:
                if int(current_bs * increase_bs_rate) <= max_bs:
                    print('increasing batch size from {} to {}'.format(
                        current_bs, int(current_bs * increase_bs_rate)))
                    current_bs = int(current_bs * increase_bs_rate)
                current_bs_round = 0

        current_epoch += 1

    ' 2次元になるのでflatten '
    test_pred = model.predict(x_test, batch_size=2048).flatten().clip(0,1)

    return y_pred, auc, test_pred


def main(path):
    #  a = np.load('../output/keras_valid1.0.npy')
    #  print(a)
    #  sys.exit()

    #  path = '../features/3_winner/*.npy'
    #  path = '../features/regularize_select_target_feature/*.npy'
    #  path = '../features/regularize_first_feature/*.npy'
    #  path = '../features/regularize_before_add_kernel_select_target_CV0.7992_avg/*.npy'
    #  base = pd.read_csv('../data/base.csv')
    #  tmp_data = make_feature_set(base, path)
    #  tmp_data = tmp_data.astype('float16')

    #  data = pd.read_csv('../data/0819_keras_target_feature.csv')

    #  data = data_regulize(df=data, mm_flg=1, na_flg=1, inf_flg=1, float16_flg=1, ignore_feature_list=ignore_features, logger=logger)
    #  make_raw_feature(data=data, path='../features/regularize_before_add_kernel_select_target_CV0.7992_avg/', ignore_list=ignore_features, logger=logger)
    #  sys.exit()
    tmp_data = pd.read_csv('../data/regular_no_app_2.csv')
    base = pd.read_csv('../data/base.csv')
    tmp_data['is_train'] = base['is_train'].values
    tmp_data['TARGET'] = base['TARGET'].values
    tmp_data['valid_no_4'] = base['valid_no_4'].values

    logger.info(f'shape: {tmp_data.shape}')
    data = tmp_data.query('is_train == 1')
    test = tmp_data.query('is_train != 1')

    tmp_pred = np.zeros(len(test))
    valid_list = data[val_col].drop_duplicates().values
    score_list = []

    ' MLP '
    #  x_train, y_train = x_y_split(data, target)
    #  use_cols = [col for col in x_train.columns if col not in ignore_features and not(col.count('Unname'))]
    #  x_train = x_train[use_cols].values
    #  mlp_clf(x_train, y_train)
    #  sys.exit()

    result = pd.DataFrame([])
    for val_no in valid_list:

        train = data.query(f"valid_no_4 != {val_no}")
        valid = data.query(f"valid_no_4 == {val_no}")
        x_train, y_train = x_y_split(train, target)
        x_val, y_val = x_y_split(valid, target)

        ' テストのため、絞る '
        use_cols = [ col for col in x_train.columns if col not in ignore_features and not(col.count('Unname'))]
        dim = len(use_cols)

        #  tmp = x_val[unique_id].to_frame()
        tmp = x_val[unique_id].to_frame()
        x_train = x_train[use_cols].values
        x_val = x_val[use_cols].values
        x_test = test[use_cols].values

        logger.info(f'VALID_NO: {val_no} START!!')
        #  y_pred, sc_score, test_pred = redhat_keras(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, x_test=x_test, dim=dim)
        y_pred, sc_score, test_pred = keras_ANN(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, x_test=x_test, dim=dim)
        #  y_pred, sc_score, test_pred = kernel_keras(
        #      x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, x_test=test, dim=dim)
        logger.info(f'valid_no: {val_no} | auc: {sc_score}')

        #  np.save(file=f'../output/{start_time[:11]}keras_valid{val_no}', arr=y_pred)
        score_list.append(sc_score)
        tmp['prediction'] = y_pred

        tmp_pred += test_pred / len(valid_list)
        #  tmp_pred += (np.array(test_pred) / len(valid_list))

        if len(result) == 0:
            result = tmp.copy()
        else:
            result = pd.concat([result, tmp], axis=0)

    score = np.mean(score_list)
    logger.info(f'RESULT AUC: {score}')

    submit = pd.read_csv('../data/sample_submission.csv')[unique_id].to_frame()
    submit[target] = tmp_pred
    #  submit = submit.merge(test[[unique_id, target]], on=unique_id, how='inner')
    submit.to_csv(f'../submit/{start_time[:11]}_keras_submission_{len(use_cols)}features_auc{str(score)[:7]}.csv', index=False)

    result = pd.concat([result, submit], axis=0)
    result.to_csv(f'../output/{start_time}_{len(use_cols)}features_auc{str(score)[:7]}_keras_prediction.csv', index=False)

    #  keras_classifier(data)


if __name__ == '__main__':
    path_list = [
        '../features/regularize_select_target_feature/*.npy'
        #  ,'../features/regularize_first_feature/*.npy'
        #  ,'../features/regularize_before_add_kernel_select_target_CV0.7992_avg/*.npy'
    ]
    for path in path_list:
        main(path)
