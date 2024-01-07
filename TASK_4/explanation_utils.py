import numpy as np
import pandas as pd
import tensorflow as tf
from enum import Enum
import pickle
from aix360.metrics import faithfulness_metric, monotonicity_metric
from keras.models import load_model
from scikeras.wrappers import KerasClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

def evaluate_explanation(model, instance, feature_importances, feature_defaults):
    metrics = {}
    metrics['faithfulness'] = faithfulness_metric(model, instance, feature_importances, feature_defaults)
    metrics['monotonicity'] = monotonicity_metric(model, instance, feature_importances, feature_defaults)
    return metrics

class Classifiers(Enum):
    # TODO: aggiungere gli altri
    DT = 'DecisionTreeClassifier' # tree
    KNN = 'KNearestNeighborsClassifier' #kernel explainer
    NC = 'NearestCentroidClassifier' #kernel explainer
    NN = 'NeuralNetworkClassifier' #deep explainer / kernel
    RF = 'RandomForestClassifier' # tree
    #AB = 'AdaBoostClassifier'
    NBM = 'NaiveBayesMixedClassifier'
    #RIPPER = 'RipperClassifier'
    SVM = 'SupportVectorMachineClassifier' #kernel
    TN = 'TabNetClassifier'
    XGB = 'ExtremeGradientBoostingClassifier' # tree

def get_classifiers_objects(load_path, delete_feature_names=True): # TODO: verificarne il funzionamento una volta aggiunti gli altri
    METRICS = [
        tf.keras.metrics.BinaryCrossentropy(name='cross entropy'),  # same as model's loss
        tf.keras.metrics.MeanSquaredError(name='Brier score'),
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
    def get_clf(meta, hidden_layer_sizes, dropout, final_activation_fun, output_bias):
        output_bias = tf.keras.initializers.Constant(output_bias)
        n_features_in_ = meta["n_features_in_"]

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(n_features_in_,)))
        for hidden_layer_size in hidden_layer_sizes:
            model.add(tf.keras.layers.Dense(hidden_layer_size, activation="relu"))
            model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.Dense(1, activation=final_activation_fun, bias_initializer=output_bias))
        return model
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=6,
        restore_best_weights=True)

    clf_names = [clf.value for clf in Classifiers]
    classifiers = {}
    for clf_name in clf_names:
        if clf_name == Classifiers.NN.value:
            nn = load_model(load_path+clf_name+'.h5')
            def dummy_f(X):
                return np.array([[1-x[0], x[0]] for x in nn.predict(X)])
            nn.predict_proba = dummy_f
            
            '''KerasClassifier(
                model=get_clf,
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), # because there is a sigmoid layer
                metrics=METRICS,
                callbacks=[early_stopping],
                validation_split=0.2
            )
            nn.model = load_model(load_path+clf_name+'.h5')'''
            classifiers[clf_name] = nn

        elif clf_name == Classifiers.TN.value:
            tn = TabNetClassifier()
            tn.load_model(load_path+Classifiers.TN.value+'.pkl.zip')
            classifiers[clf_name] = tn
        else:
            with open(load_path+clf_name+'.pkl', 'rb') as file:
                print(clf_name)
                classifiers[clf_name] = pickle.load(file)
        if delete_feature_names:
            if clf_name != Classifiers.XGB.value:
                classifiers[clf_name].feature_names_in_ = None
    return classifiers

def get_classifiers_predictions(load_path): # TODO: verificarne il funzionamento una volta aggiunti gli altri
    clf_names = [clf.value for clf in Classifiers]
    preds = {}
    for clf_name in clf_names:
        preds[clf_name] = {}
        clf_preds = pd.read_csv(load_path+clf_name+'_preds.csv')
        preds[clf_name]['labels'] = clf_preds['labels']
        if clf_name != Classifiers.NC.value and clf_name != Classifiers.KNN.value:
            preds[clf_name]['probs'] = clf_preds['probs']
    return preds