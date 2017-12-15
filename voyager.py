import pandas as pd
import numpy as np
from sklearn.svm import SVC, OneClassSVM, NuSVC
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score, make_scorer, recall_score, accuracy_score, confusion_matrix, precision_recall_curve, precision_score, roc_curve, auc, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.base import clone
from sklearn.neighbors import LocalOutlierFactor
import time
import matplotlib.pyplot as plt
import itertools
from sklearn.cross_validation import train_test_split


class MultiUnderSamplerEstimator:
    def __init__(self, decomposor, estimator, n_estimators = 10, verbose = True):
        self.clfs = []
        self.score_weights = []
        self.imbs = []
        self.decomposor = decomposor
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.verbose = verbose
        self.roc_aucs = []
        self.fpr_tprs = []
        self.y_tests = []
        self.y_scores = []
    
    def fit(self, df_X, df_y):

        # Deal with imbalanced data
        for i in range(self.n_estimators):
            imb = RandomUnderSampler()
            X_resampled, y_resampled = imb.fit_sample(df_X, df_y)            
            X_res_vis = self.decomposor.transform(X_resampled)
        
            clf = clone(self.estimator)
            score = cross_val_score(clf, X_res_vis, y_resampled, cv = 10,
                                    scoring = make_scorer(roc_auc_score)).mean()
            if self.verbose:
                print("Estimator %d:" % (i + 1))
                print("AUC: %.2f" % score)
            self.score_weights.append(score)
            
            clf.fit(X_res_vis, y_resampled)
            self.clfs.append(clf)
            self.imbs.append(imb)

    def fit_divided_hab(self, df_X, df_y):
        for i in range(self.n_estimators):
            imb = RandomUnderSampler()
            X_resampled, y_resampled = imb.fit_sample(df_X, df_y)
            X_res_vis = self.decomposor.transform(X_resampled)
            X_train, X_test, y_train, y_test = train_test_split(X_res_vis, y_resampled, test_size=0.3, random_state=0)

            clf = clone(self.estimator)
            score = cross_val_score(clf, X_train, y_train, cv = 10,
                                    scoring = make_scorer(roc_auc_score)).mean()
            if self.verbose:
                print("Estimator %d:" % (i + 1))
                print("AUC: %.2f" % score)
            self.score_weights.append(score)
            
            # clf.fit(X_train, y_train)
            y_score = clf.fit(X_train, y_train).decision_function(X_test)
            self.clfs.append(clf)
            self.imbs.append(imb)
            self.y_tests.append(y_test)
            self.y_scores.append(y_score)

    def get_roc_value(self):
        best_roc_aucs = 0
        fpr = 0
        tpr = 0
        for i in range(len(self.roc_aucs)):
            if self.roc_aucs[i] > best_roc_aucs:
                best_roc_aucs = self.roc_aucs[i]
                fpr, tpr = self.fpr_tprs[i]
        return fpr, tpr, best_roc_aucs


        
    def predict(self, df_X):
        if len(self.clfs) == 0:
            raise("Error: fit the model please.")
        
        res = []
        for clf in self.clfs:
            res.append(clf.predict(df_X))
        
        weights = np.array(pd.DataFrame(MinMaxScaler().fit_transform(np.array(self.score_weights).reshape(-1, 1))).T)[0]
        assert len(res) == len(weights)
        
        for i in range(len(res)):
            res[i] = np.array([x * weights[i] for x in res[i]])
            
        return np.array([1 if x > (sum(weights) * 0.5) else 0 for x in sum(res)])
    
    def predict_proba(self, df_X):
        if len(self.clfs) == 0:
            raise("Error: fit the model please.")
        
        res = []
        for clf in self.clfs:
            res.append(clf.predict_proba(df_X))
        
        weights = np.array(pd.DataFrame(MinMaxScaler().fit_transform(np.array(self.score_weights).reshape(-1, 1))).T)[0]
        weights = weights / weights.sum()
        assert len(res) == len(weights)
        
        for i in range(len(res)):
            res[i] = np.array([x[0] * weights[i] for x in res[i]])
            
        return np.array([[x, 1-x] for x in sum(res)])
        

def timeit(func):
    def wrapper(*arg, **kw):
        start = time.time()
        res = func(*arg, **kw)
        print(func.__name__, "execution time:", round(time.time() - start, 2), "s")
        return res
    return wrapper


# Delete columns that contains too many nans.
def remove_nan_columns(df):
    for col in df.columns:
        if df[col].isnull().sum() > len(df) * 0.3:
            df = df.drop([col], axis = 1)
    return df


def load_confirmed(shuffle_all = False):

    df = pd.read_csv("data/confirmed planets 2017-11-02.csv").drop(["rowid"], axis = 1)
    df = remove_nan_columns(df)
    if shuffle_all: df = shuffle(df)
    
    df_X = df.drop(["habitable"], axis = 1)
    df_y = df["habitable"]
    
    # Convert alphabet into integer
    df_X["pl_letter"] = [ord(x) - 96 for x in df_X["pl_letter"]]
    
    return df_X, df_y


def load_kepler(candidate = False, habitable = False, shuffle_all = False):
    
    df = pd.read_csv("data/kepler candidates 2017-08-31.csv").drop(["rowid"], axis = 1)
    df = remove_nan_columns(df)
    if habitable:
        df = df[df['koi_disposition'] == "CONFIRMED"].reset_index(drop = True)
        if shuffle_all: df = shuffle(df)
        df_X = df.drop(['habitable'], axis = 1)
        df_y = df['habitable']
        
        return df_X, df_y
    else:        
        if not candidate:
            df = df[df['koi_disposition'] != "CANDIDATE"].reset_index(drop = True)
            if shuffle_all: df = shuffle(df)
            df_X = df.drop(['koi_disposition'], axis = 1)
            
            def to_binary(label):
                return 1 if label == "CONFIRMED" else 0
            
            df_y = df['koi_disposition'].map(to_binary)
            
            return df_X, df_y
        else:
            return df[df['koi_disposition'] == "CANDIDATE"].reset_index(drop = True)


# Delete columns if they are strings.
def remove_not_numeric(df):
    for col in df.columns:
        if df[col].dtype == object:
            df = df.drop([col], axis = 1)
    
    return df


def train_habitable(df_X, df_y, ensemble = False):
    
    df_X = remove_not_numeric(df_X)
    
    # Missing value
    # Standardization
    pipe = Pipeline([("imputer", Imputer()), 
                     ("minmaxscaler", MinMaxScaler())])
    df_X = pipe.fit_transform(df_X)
    
    # PCA process
    pca = PCA(n_components = 100)
    pca.fit_transform(df_X)
    #est = RandomForestClassifier(random_state = 0, n_estimators = 100, n_jobs = -1)
    est = NuSVC(probability = True)
    if ensemble:
        clf = MultiUnderSamplerEstimator(pca, est, n_estimators = 100, verbose = False)
        clf.fit(df_X, df_y)
    else:
        # Deal with imbalanced data
        imb = RandomUnderSampler()
        X_resampled, y_resampled = imb.fit_sample(df_X, df_y)
        X_res_vis = pca.transform(X_resampled)
    
        clf = clone(est)
        score = cross_val_score(clf, X_res_vis, y_resampled, cv = 10,
                                scoring = make_scorer(recall_score)).mean()
        print("Recall: %.2f" % score)
        
        clf.fit(X_res_vis, y_resampled)
    
    return clf, pipe, pca
    
    
def train_candidate(df_X, df_y):
    
    df_X = remove_not_numeric(df_X)
    
    # Missing value
    # Standardization
    # PCA process
    pipe = Pipeline([("imputer", Imputer()), 
                     ("scaler", MinMaxScaler()),
                     ("decomposition", PCA(n_components = 100))])
    df_X = pipe.fit_transform(df_X)
    
    # clf = RandomForestClassifier(random_state = 0, n_estimators = 10, n_jobs = -1)
    clf = SVC()
    score = cross_val_score(clf, df_X, df_y, cv = 10).mean()
    print("Accuracy on whether exoplanet or not: %.2f" % score)
    
    clf.fit(df_X, df_y)
    return clf, pipe

   
@timeit
def kepler_candidate_dataset(probability = 0.99, ensemble = True):
    df_X, df_y = load_kepler()
    clf, pipe = train_candidate(df_X, df_y)
    
    df_candidate = load_kepler(candidate = True)
    df_candidate = remove_not_numeric(df_candidate)
    df = pipe.transform(df_candidate)
    res = clf.predict(df)
    df_candidate['pre_confirmed'] = pd.Series(res)
    print("{:.2%}".format(res.sum() / len(res)) + " of the candidates are comfirmed.")
    df_pre_confirmed = df_candidate[df_candidate['pre_confirmed'] == 1].reset_index(drop = True)
    
    # train habitable
    df_hab_X, df_hab_y = load_kepler(habitable = True)
    clf, pipe, pca = train_habitable(df_hab_X, df_hab_y, ensemble = ensemble)
    df = pipe.transform(df_pre_confirmed.drop(["habitable", "pre_confirmed"], axis = 1))
    df = pca.transform(df)
    res = clf.predict(df)
    df_pre_confirmed["habitable"] = pd.Series(res)
    print("{:.2%}".format(res.sum() / len(res)) + " of the confirmed planets are habitable.")
    df_pre_habitable = df_candidate[df_candidate['habitable'] == 1].reset_index(drop = True)
    res = np.array([1 if x[0] > probability else 0 for x in clf.predict_proba(df)])
    print("{:.2%}".format(res.sum() / len(res)) + " of the confirmed planets have " + 
          "{:.2%}".format(probability) + " probability habitable.")
    


@timeit
def kepler_train_test():
    df_X, df_y = load_kepler()
    
    
    div = int(len(df_X) * 0.7)
    X_train = df_X[:div]
    X_test = df_X[div:]
    y_train = df_y[:div]
    y_test = df_y[div:]
    clf, pipe = train_candidate(X_train, y_train)
    
    X_test = remove_not_numeric(X_test)
    df = pipe.transform(X_test)
    print("Accuracy on test: %.2f" % accuracy_score(y_test, clf.predict(df)))

@timeit
def kepler_train_test_new():
    df_X, df_y = load_kepler()
    
    X_hab = df_X[df_y == 1]
    X_not_hab = df_X[df_y == 0]
    y_hab = df_y[df_y == 1]
    y_not_hab = df_y[df_y == 0]

    div_hab = int(len(X_hab) * 0.7)
    div_not_hab = int(len(X_not_hab) * 0.7)
    X_train = X_not_hab[:div_not_hab]
    X_test = X_not_hab[div_not_hab:]
    X_train = X_train.append(X_hab[:div_hab], ignore_index=True)
    # print(X_train)
    X_test = X_test.append(X_hab[div_hab:], ignore_index=True)
    y_train = y_not_hab[:div_not_hab]
    y_test = y_not_hab[div_not_hab:]
    y_train = y_train.append(y_hab[:div_hab], ignore_index=True)
    # print(y_train)
    y_test = y_test.append(y_hab[div_hab:], ignore_index=True)
    clf, pipe = train_candidate(X_train, y_train)
    
    X_test = remove_not_numeric(X_test)
    df = pipe.transform(X_test)

    imb = RandomUnderSampler()
    X_test_resample, y_test_resample = imb.fit_sample(df, y_test)

    # print("Accuracy on test: %.2f" % accuracy_score(y_test, clf.predict(df)))
    # print("Accuracy on test: %.2f" % accuracy_score(y_test_resample, clf.predict(X_test_resample)))
    y_pred = clf.predict(df)
    # y_pred = clf.predict(X_test_resample)
    # print("precision score: " + str(precision_score(y_test, y_pred)))
    # print("precision score: " + str(precision_score(y_test_resample, y_pred)))
    print("precision_score: " + str(precision_score(y_test, y_pred)))
    print("recall_score: " + str(recall_score(y_test, y_pred)))
    print("f1_score: " + str(f1_score(y_test, y_pred)))
    print("auc score: " + str(roc_auc_score(y_test, y_pred)))

    # print("precision_score: " + str(precision_score(y_test_resample, y_pred)))
    # print("recall_score: " + str(recall_score(y_test_resample, y_pred)))
    # print("f1_score: " + str(f1_score(y_test_resample, y_pred)))
    # print("auc score: " + str(roc_auc_score(y_test_resample, y_pred)))

    return y_test, y_pred
    # return y_test_resample, y_pred

@timeit
def confirmed_exoplanets_dataset_onebyone(ensemble = True):
    df_X, df_y = load_confirmed()
    index = df_X[df_y == 1].index
    cnt = 0
    for each in index:
        df = df_X[each:each+1]
        # print(df_y[each:each+1])
        df_X_ = df_X.drop([each])
        df_y_ = df_y.drop([each])
        
        clf, pipe, pca = train_habitable(df_X_, df_y_, ensemble = ensemble)
        
        df = remove_not_numeric(df)
        df = pipe.transform(df)
        df = pca.transform(df)
        res = clf.predict(df)
        # print("{:.2%}".format(res.sum() / len(res)) + " of the confirmed planets are habitable.")    
        if res.sum() / len(res) > 0.5:
            cnt += 1
    
    print("{:.2%}".format(cnt / len(index)) + " of the habitable planets are classified correctly.")    

@timeit 
def confirmed_exoplanets_dataset(ensemble = True):
    df_X, df_y = load_confirmed()
    df_X = remove_not_numeric(df_X)
    X_hab = df_X[df_y == 1]
    X_not_hab = df_X[df_y == 0]
    y_hab = df_y[df_y == 1]
    y_not_hab = df_y[df_y == 0]

    div_hab = int(len(X_hab) * 0.7)
    div_not_hab = int(len(X_not_hab) * 0.7)
    # X_train = df_X[:div]
    # X_test = df_X[div:]
    # y_train = df_y[:div]
    # y_test = df_y[div:]
    # print(type(X_train))
    X_train = X_not_hab[:div_not_hab]
    X_test = X_not_hab[div_not_hab:]
    X_train = X_train.append(X_hab[:div_hab], ignore_index=True)
    # print(X_train)
    X_test = X_test.append(X_hab[div_hab:], ignore_index=True)
    y_train = y_not_hab[:div_not_hab]
    y_test = y_not_hab[div_not_hab:]
    y_train = y_train.append(y_hab[:div_hab], ignore_index=True)
    # print(y_train)
    y_test = y_test.append(y_hab[div_hab:], ignore_index=True)
    print(len(y_test))
    # print(y_test)
    # print(X_train)
    # print(y_train)
    clf, pipe, pca = train_habitable(X_train, y_train, ensemble = ensemble)
    X_test = pipe.transform(X_test)
    X_test = pca.transform(X_test)

    imb = RandomUnderSampler()
    X_test_resample, y_test_resample = imb.fit_sample(X_test, y_test)
    # imb = SMOTE()
    # X_test_resample, y_test_resample = imb.fit_sample(X_test, y_test)
    
    y_pred = clf.predict_proba(X_test_resample)
    preds = y_pred[:,1]
    fpr, tpr, threshold = roc_curve(y_test_resample, preds)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    # y_pred = clf.predict(X_test)
    # print("precision_score: " + str(precision_score(y_test_resample, y_pred)))
    # print("recall_score: " + str(recall_score(y_test_resample, y_pred)))
    # print("f1_score: " + str(f1_score(y_test_resample, y_pred)))
    # print("auc score: " + str(roc_auc_score(y_test_resample, y_pred)))
    print("precision_score: " + str(precision_score(y_test, y_pred)))
    print("recall_score: " + str(recall_score(y_test, y_pred)))
    print("f1_score: " + str(f1_score(y_test, y_pred)))
    print("auc score: " + str(roc_auc_score(y_test, y_pred)))

    # return y_test_resample, y_pred
    return y_test, y_pred
    
    
@timeit
def test_false_positive(ensemble = True):
    df_X, df_y = load_confirmed()
    index = df_X[df_y == 0].index
    cnt = 0
    for each in index:
        df = df_X[each:each+1]
        print(df_y[each:each+1])
        df_X_ = df_X.drop([each])
        df_y_ = df_y.drop([each])
        
        clf, pipe, pca = train_habitable(df_X_, df_y_, ensemble = ensemble)
        
        df = remove_not_numeric(df)
        df = pipe.transform(df)
        df = pca.transform(df)
        res = clf.predict(df)
        # print("{:.2%}".format(res.sum() / len(res)) + " of the confirmed planets are habitable.")    
        if res.sum() / len(res) > 0.5:
            cnt += 1
    
    print("{:.2%}".format(cnt / len(index)) + " of the unhabitable planets are mis-classified.")    


@timeit
def outlier_detection_SVM():
    df_X, df_y = load_confirmed()
    df_X = remove_not_numeric(df_X)
    
    index = df_X[df_y == 1].index
    
    X = df_X.drop(index)
    y = df_y.drop(index)
    print(X)
    print(len(index), index)
    div = int(len(X) * 0.7)
    X_train = X[:div]
    X_test = X[div:]
    X_outliers = df_X.ix[index]
    print(X_outliers)
    
    pipe = Pipeline([("imputer", Imputer()), 
                     ("scaler", MinMaxScaler()),
                     ("decomposition", PCA(n_components = 100))])
    X_train = pipe.fit_transform(X_train)
    clf = OneClassSVM(gamma = (1 / len(X_train)), nu = 0.37)
    clf.fit(X_train)
    
    X_test = pipe.transform(X_test)    
    X_outliers = pipe.transform(X_outliers)
    print(X_outliers)
    
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    print("**********  " + str(len(X_test)) + "   " + str(len(y_pred_test)))
    y_pred_outliers = clf.predict(X_outliers)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
    
    print("train error:", n_error_train / len(y_pred_train))
    print("test error:", n_error_test / len(y_pred_test))
    print("outliers error:", n_error_outliers / len(y_pred_outliers))
    return y_pred_train, y_pred_test, y_pred_outliers, y[div:]


@timeit
def outlier_detection_local():
    df_X, df_y = load_confirmed()
    df_X = remove_not_numeric(df_X)
    
    pipe = Pipeline([("imputer", Imputer()), 
                     ("scaler", MinMaxScaler()),
                     ("decomposition", PCA(n_components = 100))])
    df_X = pipe.fit_transform(df_X)
    clf = LocalOutlierFactor()
    res = clf.fit_predict(df_X)
    
    return res, df_y, 0

def draw_count_figure(df, class_name):
    count_classes = pd.value_counts(df[class_name], sort = True).sort_index()
    count_classes.plot(kind = 'bar')
    plt.title(class_name + " class histogram")
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def draw_confusion_figure(y_test, y_pred):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
    print(precision_score(y_test, y_pred))

    # Plot non-normalized confusion matrix
    class_names = [0,1]
    plt.figure()
    plot_confusion_matrix(cnf_matrix
                          , classes=class_names
                          , title='Confusion matrix')
    plt.show()

@timeit 
def confirmed_exoplanets_dataset_divided(ensemble = True):
    df_X, df_y = load_confirmed()
    df_X = remove_not_numeric(df_X)
    print(df_y.shape)
    n_classes = 2

    clf, pipe, pca = train_habitable(df_X, df_y, ensemble = ensemble)
    y_test = clf.y_tests[0]
    y_score = clf.y_score[0]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    print(fpr)
    print(tpr)
    print(roc_auc)
    return fpr, tpr, roc_auc


def plot_roc():
    import matplotlib.pyplot as plt
    df_X, df_y = load_confirmed()
    clf, pipe, pca = train_habitable(df_X, df_y, ensemble = True)    
    df = df_X
    y_test = df_y
    df = remove_not_numeric(df)
    df = pipe.transform(df)
    df = pca.transform(df)
    res = clf.predict_proba(df)
    preds = res[:,1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


if __name__ == '__main__':
    #x, y, z = outlier_detection_local()
    # kepler_train_test()
    #kepler_candidate_dataset(probability = 0.5)
    #confirmed_exoplanets_dataset_onebyone()
    #test_false_positive()
    
    # df_x, df_y = load_confirmed()
    # df_x, df_y = load_kepler()
    # df_x['habitable'] = df_y
    # df_x['exoplanet'] = df_y
    # draw_count_figure(df_x, 'habitable')

    # y_test, y_pred = confirmed_exoplanets_dataset()
    # print("**********  " + str(len(y_test)) + "   " + str(len(y_pred)))
    # draw_confusion_figure(y_test, y_pred)

    # fpr, tpr, roc_auc = confirmed_exoplanets_dataset_divided()

    # y_test, y_pred = kepler_train_test_new()
    # print("**********  " + str(len(y_test)) + "   " + str(len(y_pred)))
    # draw_confusion_figure(y_test, y_pred)
    confirmed_exoplanets_dataset()

