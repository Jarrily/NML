# pip install openpyxl
# pip install XlsxWriter
import zipfile
import streamlit as st
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_curve, classification_report, auc
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from skopt import BayesSearchCV
from xgboost import XGBClassifier
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import joblib

# st.info("Predictive models for the malignancy of NML based on multiple machine learning methods")
st.markdown("<h3 style='font-size: 24px; font-weight: bold;'>Predictive models for the malignancy of NML based on multiple machine learning methods</h3>", unsafe_allow_html=True)
st.sidebar.image("./PIC.png", caption="UltraVisionAI", use_column_width=True)
modelfusionall_words = st.sidebar.selectbox("Select a model",
                                        ["Random Forest", "Decision Tree", "Extra Trees", "SVM", "Logistic Regression",
                                         "SGD", "KNN", "XGBoost", "Adaboost", "GBDT", "CatBoost", "LightGBM", "Bayes"])
chaocanshumoxing = st.sidebar.selectbox("Please select a hyperparameter model",[ "Random search", "Grid search","Bayesian optimization"])

RFhunxuwenjian = st.file_uploader("Please upload the file", key="RFhunxuwenjian")
st.info("Model settings......")
leftLogistic2,rightLogistic2,DSAJFINASGNR = st.columns(3)
with leftLogistic2:
    tukuan = int(st.number_input("Wide", value=12, key="Logisticwide"))
with rightLogistic2:
    tugao = int(st.number_input("High", value=10, key="Logistichigh"))
with DSAJFINASGNR:
    DPI = int(st.number_input("DPI", value=600, key="LogisticDPI"))
sdfghsgasdsafgjnuj752782_7327,sdfghsgasdsafgjnuj752782_73271,sdfghsgasdsafgjnuj752782_732713 = st.columns(3)
with sdfghsgasdsafgjnuj752782_7327:
    test_size11nei = st.number_input("Validation group ratio", value=0.3,key="RFndssddsdfgfdsgsdfgsdfgfafdafaggargsafasf_jobs")
with sdfghsgasdsafgjnuj752782_73271:
    bilishu = float(st.number_input("External Authentication Group Ratio", value=0.3, key="RFyanzhevDSAFADSFADGFGTHGFDHzfangzubili"))
with sdfghsgasdsafgjnuj752782_732713:
    moxingderandomstate = int(st.number_input("Number of Random Seeds", value=42,
                                         key="LogsdfadfadisdsfafsafadsfasfasdftidsadhyhhtsvdfvdfvbsraregassghrfdsafafcDPI"))
st.info("Hyperparameter settings......")
if chaocanshumoxing == "Grid search":
    dsafniagafdalm, sdfinafiniasdfianogiaravn = st.columns(2)
    with dsafniagafdalm:
        allcv = int(st.number_input("Cross-validation", value=10, key="LogistidsafdsafafcDPI"))
    with sdfinafiniasdfianogiaravn:
        allverbose = int(st.number_input("Level of Detail", value=2, key="LogisdsfasfasdftidsafdsafafcDPI"))
if chaocanshumoxing == "Random search":
    dsafniagafsdafsadalm, sdfinafiniasdfisadfsadfaanogiaravn,dfgadfgasdgshytryshthsr,dsfafujfujd5afa5sf16safda = st.columns(4)
    with dsafniagafsdafsadalm:
        allcv = int(st.number_input("Cross-validation", value=10, key="LogistidsxzcvzxvafdsafafcDPI"))
    with sdfinafiniasdfisadfsadfaanogiaravn:
        allverbose = int(st.number_input("Level of Detail", value=2, key="LogisdsfasfasdftidsadhyhhtsghrfdsafafcDPI"))
    with dfgadfgasdgshytryshthsr:
        tworandomstate = int(st.number_input("Number of Random Seeds", value=42, key="LogisdsfafsafadsfasfasdftidsadhyhhtsghrfdsafafcDPI"))
    with dsfafujfujd5afa5sf16safda:
        twoniters = int(st.number_input("Number of iterations", value=1000, key="sdfassdafsaddssdafdsafasgdfsgfdgfaghj5fasdfa"))
if chaocanshumoxing == "Bayesian optimization":
    dsafniagafsdafdsfasdafsadalm, sdfinafiniasdfiagdfgsthyhhsadfsadfaanogiaravn,dfgadfgasyhstfhdfgdfasdgshytryshthsrdsfadsfsd,ds1f13sadafs5fe1f16ea1f85aefaew = st.columns(4)
    with dsafniagafsdafdsfasdafsadalm:
        allcv = int(st.number_input("Cross-validation", value=10, key="LogistsdfaasdfasdfidsxzcvzxvafdsafafcDPI"))
    with sdfinafiniasdfiagdfgsthyhhsadfsadfaanogiaravn:
        allverbose = int(st.number_input("Level of Detail", value=2, key="LogisdsfasfasdftidsadhyhhtsghrfdsafdsasdfdasdasdafcDPI"))
    with dfgadfgasyhstfhdfgdfasdgshytryshthsrdsfadsfsd:
        tworandomstate = int(st.number_input("Number of Random Seeds", value=42, key="LogisdsfafsafadsfasfasdftidsadhyhhtsvdfvdfvbsraregassghrfdsafafcDPI"))
    with ds1f13sadafs5fe1f16ea1f85aefaew:
        twoniters = int(st.number_input("Number of iterations", value=1000, key="sdfassdafsadddsfasdfsadfsfaggdfsgsdhj5fasdfa"))


def ACCAUEDENGDENGtrain(y_truetrain,y_protrain,y_predtrain,X_train,mingshishenme):
    global acc_citrain_up, auc_citrain_up, sensitivity_citrain_up, specificity_citrain_up, ppv_citrain_up, npv_citrain_up, f1_citrain_up,dataTRAINPONGJIAZHIBIAO,moxing_train
    global acc_citrain_down, auc_citrain_down, sensitivity_citrain_down, specificity_citrain_down, ppv_citrain_down, npv_citrain_down, f1_citrain_down
    global accuracytrain, ppvtrain, sensitivitytrain, specificitytrain, sensitivitytrain, ppvtrain, npvtrain, f1train, roc_auctrain, cutoff_ptrain, point_roctrain, best_Youden_indextrain
    y_truetrain = np.array(y_truetrain)
    fpr, tpr, thresholds = roc_curve(y_truetrain, y_protrain[:, 1])
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred = (y_protrain[:, 1] >= optimal_threshold).astype(int)
    conf_matrix = confusion_matrix(y_truetrain, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    accuracytrain = (tp + tn) / (tn + fp + fn + tp)
    sensitivitytrain = tp / (tp + fn)
    recalltrain = sensitivitytrain
    specificitytrain = tn / (tn + fp)
    ppvtrain = tp / (tp + fp)
    precisiontrain = ppvtrain
    npvtrain = tn / (tn + fn)
    f1train = (2 * precisiontrain * recalltrain) / (precisiontrain + recalltrain)
    roc_auctrain = auc(fpr, tpr)
    n = tn + fp + fn + tp
    auc_citrain_down = roc_auctrain - 1.96 * np.sqrt((roc_auctrain * (1 - roc_auctrain) + 0.5 / n) / (n - 1))
    auc_citrain_up = roc_auctrain + 1.96 * np.sqrt((roc_auctrain * (1 - roc_auctrain) + 0.5 / n) / (n - 1))
    ppv_citrain_down = ppvtrain - 1.96 * np.sqrt((ppvtrain * (1 - ppvtrain) + 0.5 / n) / (n - 1))
    ppv_citrain_up = ppvtrain + 1.96 * np.sqrt((ppvtrain * (1 - ppvtrain) + 0.5 / n) / (n - 1))
    sensitivity_citrain_down = sensitivitytrain - 1.96 * np.sqrt(
        (sensitivitytrain * (1 - sensitivitytrain) + 0.5 / n) / (n - 1))
    sensitivity_citrain_up = sensitivitytrain + 1.96 * np.sqrt(
        (sensitivitytrain * (1 - sensitivitytrain) + 0.5 / n) / (n - 1))
    specificity_citrain_down = specificitytrain - 1.96 * np.sqrt(
        (specificitytrain * (1 - specificitytrain) + 0.5 / n) / (n - 1))
    specificity_citrain_up = specificitytrain + 1.96 * np.sqrt(
        (specificitytrain * (1 - specificitytrain) + 0.5 / n) / (n - 1))
    acc_citrain_down = accuracytrain - 1.96 * np.sqrt((accuracytrain * (1 - accuracytrain) + 0.5 / n) / (n - 1))
    acc_citrain_up = accuracytrain + 1.96 * np.sqrt((accuracytrain * (1 - accuracytrain) + 0.5 / n) / (n - 1))
    npv_citrain_down = npvtrain - 1.96 * np.sqrt((npvtrain * (1 - npvtrain) + 0.5 / n) / (n - 1))
    npv_citrain_up = npvtrain + 1.96 * np.sqrt((npvtrain * (1 - npvtrain) + 0.5 / n) / (n - 1))
    f1_citrain_down = f1train - 1.96 * np.sqrt((f1train * (1 - f1train) + 0.5 / n) / (n - 1))
    f1_citrain_up = f1train + 1.96 * np.sqrt((f1train * (1 - f1train) + 0.5 / n) / (n - 1))
    Youden_indextrain = np.argmax(tpr - fpr)
    cutoff_ptrain = optimal_threshold
    best_Youden_indextrain = 0.0
    for cutoff in np.arange(0.001, 1.00, 0.001):
        y_pred_cutoff = (y_protrain[:, 1] >= cutoff).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_truetrain, y_pred_cutoff).ravel()
        tpr_cutoff = tp / (tp + fn)
        fpr_cutoff = fp / (fp + tn)
        Youden_indextrain_cutoff = tpr_cutoff - fpr_cutoff
        if Youden_indextrain_cutoff > best_Youden_indextrain:
            best_Youden_indextrain = Youden_indextrain_cutoff
    point_roctrain = [fpr[Youden_indextrain], tpr[Youden_indextrain]]
    dataTRAINPONGJIAZHI = {
                    'P-value': [cutoff_ptrain,roc_auctrain,accuracytrain,sensitivitytrain,specificitytrain, ppvtrain,npvtrain,precisiontrain, recalltrain,f1train,best_Youden_indextrain],
                    '95%CI_down': ['NA',auc_citrain_down,acc_citrain_down,sensitivity_citrain_down, specificity_citrain_down, ppv_citrain_down,npv_citrain_down,ppv_citrain_down,sensitivity_citrain_down,f1_citrain_down,'NA'],
                    '95%CI_up': ['NA',auc_citrain_up,acc_citrain_up,sensitivity_citrain_up, specificity_citrain_up, ppv_citrain_up, npv_citrain_up,ppv_citrain_up, sensitivity_citrain_up,f1_citrain_up,'NA']}
    dataTRAINPONGJIAZHIBIAO = pd.DataFrame(dataTRAINPONGJIAZHI, index=['Cut-off','AUC','ACC', 'SEN','SPE','PPV','NPV','Precison', 'Recall', 'F1-score','Youden'])
    moxing_train = pd.read_excel(RF).iloc[X_train.index, :].copy()
    moxing_train['predict'] = y_predtrain
    for i in range(y_protrain.shape[1]):
        moxing_train[f'predict_proba_{i}'] = y_protrain[:, i]


def ACCAUEDENGDENGTEST(y_truetest,y_protest,y_predtest,X_test,modelfusionall_words,mingshishenme):
    global acc_ciTEST_up, auc_ciTEST_up, sensitivity_ciTEST_up, specificity_ciTEST_up, ppv_ciTEST_up, npv_ciTEST_up, f1_ciTEST_up,dataTESTPONGJIAZHIBIAO,moxing_test
    global acc_ciTEST_down, auc_ciTEST_down, sensitivity_ciTEST_down, specificity_ciTEST_down, ppv_ciTEST_down, npv_ciTEST_down, f1_ciTEST_down
    global accuracyTEST, ppvTEST, sensitivityTEST, specificityTEST, sensitivityTEST, ppvTEST, npvTEST, f1TEST, roc_aucTEST, best_Youden_indexTEST, point_rocTEST, cutoff_pTEST
    y_truetest = np.array(y_truetest)
    fpr, tpr, thresholds = roc_curve(y_truetest, y_protest[:, 1])
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred = (y_protest[:, 1] >= optimal_threshold).astype(int)
    conf_matrix = confusion_matrix(y_truetest, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    accuracyTEST = (tp + tn) / (tn + fp + fn + tp)
    sensitivityTEST = tp / (tp + fn)
    recallTEST = sensitivityTEST
    specificityTEST = tn / (tn + fp)
    ppvTEST = tp / (tp + fp)
    precisionTEST = ppvTEST
    npvTEST = tn / (tn + fn)
    f1TEST = (2 * precisionTEST * recallTEST) / (precisionTEST + recallTEST)
    roc_aucTEST = auc(fpr, tpr)
    n = tn + fp + fn + tp
    auc_ciTEST_down = roc_aucTEST - 1.96 * np.sqrt((roc_aucTEST * (1 - roc_aucTEST) + 0.5 / n) / (n - 1))
    auc_ciTEST_up = roc_aucTEST + 1.96 * np.sqrt((roc_aucTEST * (1 - roc_aucTEST) + 0.5 / n) / (n - 1))
    ppv_ciTEST_down = ppvTEST - 1.96 * np.sqrt((ppvTEST * (1 - ppvTEST) + 0.5 / n) / (n - 1))
    ppv_ciTEST_up = ppvTEST + 1.96 * np.sqrt((ppvTEST * (1 - ppvTEST) + 0.5 / n) / (n - 1))
    sensitivity_ciTEST_down = sensitivityTEST - 1.96 * np.sqrt(
        (sensitivityTEST * (1 - sensitivityTEST) + 0.5 / n) / (n - 1))
    sensitivity_ciTEST_up = sensitivityTEST + 1.96 * np.sqrt(
        (sensitivityTEST * (1 - sensitivityTEST) + 0.5 / n) / (n - 1))
    specificity_ciTEST_down = specificityTEST - 1.96 * np.sqrt(
        (specificityTEST * (1 - specificityTEST) + 0.5 / n) / (n - 1))
    specificity_ciTEST_up = specificityTEST + 1.96 * np.sqrt(
        (specificityTEST * (1 - specificityTEST) + 0.5 / n) / (n - 1))
    acc_ciTEST_down = accuracyTEST - 1.96 * np.sqrt((accuracyTEST * (1 - accuracyTEST) + 0.5 / n) / (n - 1))
    acc_ciTEST_up = accuracyTEST + 1.96 * np.sqrt((accuracyTEST * (1 - accuracyTEST) + 0.5 / n) / (n - 1))
    npv_ciTEST_down = npvTEST - 1.96 * np.sqrt((npvTEST * (1 - npvTEST) + 0.5 / n) / (n - 1))
    npv_ciTEST_up = npvTEST + 1.96 * np.sqrt((npvTEST * (1 - npvTEST) + 0.5 / n) / (n - 1))
    f1_ciTEST_down = f1TEST - 1.96 * np.sqrt((f1TEST * (1 - f1TEST) + 0.5 / n) / (n - 1))
    f1_ciTEST_up = f1TEST + 1.96 * np.sqrt((f1TEST * (1 - f1TEST) + 0.5 / n) / (n - 1))
    Youden_indexTEST = np.argmax(tpr - fpr)
    cutoff_pTEST = optimal_threshold
    best_Youden_indexTEST = 0.0
    for cutoff in np.arange(0.001, 1.00, 0.001):
        y_pred_cutoff = (y_protest[:, 1] >= cutoff).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_truetest, y_pred_cutoff).ravel()
        tpr_cutoff = tp / (tp + fn)
        fpr_cutoff = fp / (fp + tn)
        Youden_indexTEST_cutoff = tpr_cutoff - fpr_cutoff
        if Youden_indexTEST_cutoff > best_Youden_indexTEST:
            best_Youden_indexTEST = Youden_indexTEST_cutoff
    point_rocTEST = [fpr[Youden_indexTEST], tpr[Youden_indexTEST]]
    dataTESTPONGJIAZHI = {
            'P-value': [cutoff_pTEST,roc_aucTEST,accuracyTEST,sensitivityTEST, specificityTEST, ppvTEST,npvTEST,ppvTEST, sensitivityTEST,f1TEST,best_Youden_indexTEST],
            '95%CI_down': ['NA',auc_ciTEST_down,acc_ciTEST_down,sensitivity_ciTEST_down, specificity_ciTEST_down, ppv_ciTEST_down,npv_ciTEST_down,ppv_ciTEST_down, sensitivity_ciTEST_down,f1_ciTEST_down,'NA'],
            '95%CI_up': ['NA',auc_ciTEST_up,acc_ciTEST_up, sensitivity_ciTEST_up, specificity_ciTEST_up, ppv_ciTEST_up,npv_ciTEST_up, ppv_ciTEST_up,  sensitivity_ciTEST_up,f1_ciTEST_up,'NA']}
    dataTESTPONGJIAZHIBIAO = pd.DataFrame(dataTESTPONGJIAZHI, index=['Cut-off','AUC','ACC', 'SEN','SPE','PPV','NPV','Precison', 'Recall', 'F1-score','Youden'])
    moxing_test = pd.read_excel(RF).iloc[X_test.index, :].copy()
    moxing_test['predict'] = y_predtest
    for i in range(y_protest.shape[1]):
        moxing_test[f'predict_proba_{i}'] = y_protest[:, i]
    fprtest, tprtest, thresholds = roc_curve(y_truetest, y_protest[:, 1], pos_label=1)
    plt.figure(figsize=(tukuan, tugao), dpi=DPI)
    plt.rc('font', family='Arial')
    plt.plot(fprtest, tprtest,
             label=str(round(roc_aucTEST, 3)) + ',' + '95%CI:' + str(round(auc_ciTEST_down, 3)) + '-' + str(
                 round(auc_ciTEST_up, 3)))
    plt.plot(point_rocTEST[0], point_rocTEST[1], 'ro', markersize=8)
    plt.text(point_rocTEST[0], point_rocTEST[1], f'Threshold:{cutoff_pTEST:.3f}', fontsize=18)
    plt.legend(loc='lower right', fontsize=18)
    plt.plot([0, 1], [0, 1], 'r--', color='grey')
    plt.tick_params(labelsize=18)
    plt.ylabel('1 - Specificity', fontsize=20)
    plt.xlabel('Sensitivity', fontsize=20)
    plt.savefig('./DataStatistics/' + modelfusionall_words + '/' + modelfusionall_words + '_ROC' + mingshishenme + '.png', dpi=DPI, bbox_inches='tight')
    plt.savefig('./DataStatistics/' + modelfusionall_words + '/' + modelfusionall_words + '_ROC' + mingshishenme + '.pdf', dpi=DPI, bbox_inches='tight')
    st.markdown("#### AUC：" + str(roc_aucTEST))
    st.pyplot(plt)

def quanzhognhuizhi(hebingQUANZHONG, modelfusionall_words):
    hebingQUANZHONG.set_index("Features", inplace=True)
    sorted_data = hebingQUANZHONG.sort_values(by="Coefficients", ascending=False)
    quanzhongtugao = int(5 * tukuan / 6)
    plt.figure(figsize=(tukuan, quanzhongtugao), dpi=DPI)
    plt.bar(sorted_data.index, sorted_data["Coefficients"], color='#6699CC', width=0.5)
    plt.rc('font', family='Arial')
    plt.xticks(rotation=45, ha='right', va='top', fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Features', size=20)
    plt.ylabel('Features coefficients', size=20)
    bwith = 2
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.savefig('./DataStatistics/' + modelfusionall_words + '/' + modelfusionall_words + '_QZ.png', dpi=DPI,
                bbox_inches='tight')
    plt.savefig('./DataStatistics/' + modelfusionall_words + '/' + modelfusionall_words + '_QZ.pdf', dpi=DPI,
                bbox_inches='tight')

def moxinggoujianerfenlei(test_size,train_size):
    if modelfusionall_words == "Random Forest":
            max_depth1 = 5
            max_depth2 = 10
            max_depth3 = 20
            max_depth4 = 50
            n_estimators1 = 50
            n_estimators2 = 100
            n_estimators3 = 200
            n_estimators4 = 300
            min_samples_split1 = 2
            min_samples_split2 = 5
            min_samples_split3 = 10
            min_samples_split4 = 20
            min_samples_leaf1 = 1
            min_samples_leaf2 = 2
            min_samples_leaf3 = 4
            min_samples_leaf4 = 8
            params = {
                'n_estimators': [n_estimators1, n_estimators2, n_estimators3, n_estimators4],
                'max_depth': [max_depth1, max_depth2, max_depth3, max_depth4],
                'min_samples_split': [min_samples_split1, min_samples_split2, min_samples_split3,min_samples_split4],
                'min_samples_leaf': [min_samples_leaf1, min_samples_leaf2, min_samples_leaf3,min_samples_leaf4],
                'max_features': ['sqrt', 'log2', None],
                'criterion': ['gini', 'entropy'],
                'bootstrap': [True, False],
                'class_weight': [None, 'balanced', 'balanced_subsample'],
            }
    if modelfusionall_words == "SVM":
            c1 = 0.001
            c2 = 0.01
            c3 = 0.1
            c4 = 1
            c5 = 10
            c6 = 100
            degree1 = 2
            degree2 = 3
            degree3 = 4
            degree4 = 5
            tol1 = 1e-3
            tol2 = 1e-4
            tol3 = 1e-5
            tol4 = 1e-6
            max_iter1 = -1
            max_iter2 = 1000
            max_iter3 = 5000
            max_iter4 = 10000
            params = {
                'C': [c1, c2, c3, c4, c5, c6],
                'degree': [degree1, degree2, degree3, degree4],
                'tol': [tol1, tol2, tol3, tol4],
                'max_iter': [max_iter1, max_iter2, max_iter3, max_iter4],
                'probability': [True],
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'coef0': [0, 1],
                'shrinking': [True, False],
                'class_weight': [None, 'balanced'],
            }
    if modelfusionall_words == "Logistic Regression":
            c1 = 0.001
            c2 = 0.01
            c3 = 0.1
            c4 = 1
            c5 = 10
            c6 = 100
            max_iter1 = 100
            max_iter2 = 1000
            max_iter3 = 5000
            max_iter4 = 10000
            params = {
                'max_iter': [max_iter1, max_iter2, max_iter3, max_iter4],
                'C': [c1, c2, c3, c4, c5, c6],
                'penalty': ['l2'],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'fit_intercept': [True, False],
                'class_weight': [None, 'balanced'],
                'multi_class': ['ovr', 'multinomial'],
                'random_state': [42]
            }
    if modelfusionall_words == "SGD":
            alpha1 = 0.0001
            alpha2 = 0.001
            alpha3 = 0.01
            alpha4 = 0.1
            l1_ratio1 = 0.15
            l1_ratio2 = 0.25
            l1_ratio3 = 0.5
            l1_ratio4 = 0.75
            tol1 = 0.00001
            tol2 = 0.0001
            tol3 = 1e-3
            tol4 = 1e-2
            max_iter1 = -1
            max_iter2 = 1000
            max_iter3 = 5000
            max_iter4 = 10000
            eta01 = 0.001
            eta02 = 0.01
            eta03 = 0.1
            eta04 = 1
            random_state = 42
            params = {
                'alpha': [alpha1, alpha2, alpha3, alpha4],
                'l1_ratio': [l1_ratio1, l1_ratio2, l1_ratio3, l1_ratio4],
                'max_iter': [max_iter1, max_iter2, max_iter3, max_iter4],
                'tol': [tol1, tol2, tol3, tol4],
                'eta0': [eta01, eta02, eta03, eta04],
                'loss': ['log', 'modified_huber'],
                'penalty': ['none', 'l2', 'l1', 'elasticnet'],
                'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
                'random_state': [int(random_state)]
            }
    if modelfusionall_words == "KNN":
            n_neighbors1 = 3
            n_neighbors2 = 5
            n_neighbors3 = 7
            n_neighbors4 = 9
            n_neighbors5 = 11
            n_neighbors6 = 13
            n_neighbors7 = 15
            leaf_size1 = 10
            leaf_size2 = 30
            leaf_size3 = 50
            leaf_size4 = 100
            p1 = 1
            p2 = 2
            p3 = 3
            p4 = 4
            params = {
                'n_neighbors': [n_neighbors1, n_neighbors2, n_neighbors3, n_neighbors4, n_neighbors5, n_neighbors6,
                                n_neighbors7],
                'leaf_size': [leaf_size1, leaf_size2, leaf_size3, leaf_size4],
                'p': [p1, p2, p3, p4],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev'],
            }
    if modelfusionall_words == "XGBoost":
            learning_rate1 = 0.1
            learning_rate2 = 0.01
            learning_rate3 = 0.001
            max_depth1 = 3
            max_depth2 = 5
            max_depth3 = 7
            n_estimators1 = 50
            n_estimators2 = 100
            n_estimators3 = 200
            subsample1 = 0.8
            subsample2 = 0.9
            subsample3 = 1.0
            colsample_bytree1 = 0.8
            colsample_bytree2 = 0.9
            colsample_bytree3 = 1.0
            gamma1 = 0
            gamma2 = 0.1
            gamma3 = 0.2
            reg_alpha1 = 0
            reg_alpha2 = 0.1
            reg_alpha3 = 0.2
            reg_lambda1 = 0
            reg_lambda2 = 0.1
            reg_lambda3 = 0.2
            params = {
                'learning_rate': [learning_rate1, learning_rate2, learning_rate3],
                'max_depth': [max_depth1, max_depth2, max_depth3],
                'n_estimators': [n_estimators1, n_estimators2, n_estimators3],
                'subsample': [subsample1, subsample2, subsample3],
                'colsample_bytree': [colsample_bytree1, colsample_bytree2, colsample_bytree3],
                'gamma': [gamma1, gamma2, gamma3],
                'reg_alpha': [reg_alpha1, reg_alpha2, reg_alpha3],
                'reg_lambda': [reg_lambda1, reg_lambda2, reg_lambda3]
            }
    if modelfusionall_words == "LightGBM":
            learning_rate1 = 0.1
            learning_rate2 = 0.01
            learning_rate3 = 0.001
            max_depth1 = 1
            max_depth2 = 5
            max_depth3 = 9
            n_estimators1 = 50
            n_estimators2 = 100
            n_estimators3 = 200
            subsample1 = 0.8
            subsample2 = 0.9
            subsample3 = 1.0
            colsample_bytree1 = 0.8
            colsample_bytree2 = 0.9
            colsample_bytree3 = 1.0
            num_leaves1 = 31
            num_leaves2 = 63
            num_leaves3 = 127
            reg_alpha1 = 0
            reg_alpha2 = 0.1
            reg_alpha3 = 0.2
            reg_lambda1 = 0
            reg_lambda2 = 0.1
            reg_lambda3 = 0.2
            min_child_samples1 = 20
            min_child_samples2 = 50
            min_child_samples3 = 100
            params = {
                'num_leaves': [num_leaves1, num_leaves2, num_leaves3],
                'learning_rate': [learning_rate1, learning_rate2, learning_rate3],
                'n_estimators': [n_estimators1, n_estimators2, n_estimators3],
                'max_depth': [max_depth1, max_depth2, max_depth3],
                'min_child_samples': [min_child_samples1, min_child_samples2, min_child_samples3],
                'subsample': [subsample1, subsample2, subsample3],
                'colsample_bytree': [colsample_bytree1, colsample_bytree2, colsample_bytree3],
                'reg_alpha': [reg_alpha1, reg_alpha2, reg_alpha3],
                'reg_lambda': [reg_lambda1, reg_lambda2, reg_lambda3],
                'boosting_type': ['gbdt', 'dart']
            }
    if modelfusionall_words == "Bayes":
            params = {
            }
    if modelfusionall_words == "Decision Tree":
            max_depth1 = 2
            max_depth2 = 10
            max_depth3 = 20
            max_depth4 = 50
            min_samples_split1 = 2
            min_samples_split2 = 5
            min_samples_split3 = 10
            min_samples_split4 = 20
            min_samples_leaf1 = 1
            min_samples_leaf2 = 2
            min_samples_leaf3 = 4
            min_samples_leaf4 = 8
            params = {
                'max_depth': [None, max_depth1, max_depth2, max_depth3, max_depth4],
                'min_samples_split': [min_samples_split1, min_samples_split2, min_samples_split3,
                                      min_samples_split4],
                'min_samples_leaf': [min_samples_leaf1, min_samples_leaf2, min_samples_leaf3, min_samples_leaf4],
                'criterion': ['gini', 'entropy']
            }
    if modelfusionall_words == "Adaboost":
            learning_rate1 = 0.1
            learning_rate2 = 0.01
            learning_rate3 = 0.001
            n_estimators1 = 50
            n_estimators2 = 100
            n_estimators3 = 200
            params = {
                'learning_rate': [learning_rate1, learning_rate2, learning_rate3],
                'n_estimators': [n_estimators1, n_estimators2, n_estimators3],
                'algorithm': ['SAMME', 'SAMME.R']
            }
    if modelfusionall_words == "GBDT":
                learning_rate1 = 0.1
                learning_rate2 = 0.01
                learning_rate3 = 0.001
                n_estimators1 = 50
                n_estimators2 = 100
                n_estimators3 = 200
                max_depth1 = 2
                max_depth2 = 10
                max_depth3 = 20
                max_depth4 = 50
                min_samples_split1 = 2
                min_samples_split2 = 5
                min_samples_split3 = 10
                min_samples_split4 = 20
                params = {
                    'learning_rate': [learning_rate1, learning_rate2, learning_rate3],
                    'n_estimators': [n_estimators1, n_estimators2, n_estimators3],
                    'max_depth': [None, max_depth1, max_depth2, max_depth3, max_depth4],
                    'min_samples_split': [min_samples_split1, min_samples_split2, min_samples_split3,
                                          min_samples_split4],
                }
    if modelfusionall_words == "CatBoost":
            params = {
                'iterations': [-1, 1000, 5000, 10000],
                'learning_rate': [0.1, 0.01, 0.001],
                'depth': [3, 5, 7],
            }
    if modelfusionall_words == "Extra Trees":
            params = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [2, 10, 20, 50],
                'min_samples_split': [2, 5, 10, 20],
            }

    if modelfusionall_words == "Random Forest":
        os.makedirs('./DataStatistics/Random Forest/', exist_ok=True)
        def rfhanshu(XX,yy,mingshishenme,shenmebestmodel,testortrain_size):
            X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size= test_size11nei, random_state=42)
            bestmodel = RandomForestClassifier(random_state= moxingderandomstate )
            if mingshishenme != "test":
                if chaocanshumoxing == "Grid search":
                    best_model = GridSearchCV(bestmodel, param_grid=params, cv=allcv, n_jobs=-1,verbose=allverbose)
                elif  chaocanshumoxing == "Random search":
                    best_model = RandomizedSearchCV(estimator=bestmodel,param_distributions=params,n_iter=twoniters,cv=allcv,verbose=allverbose,random_state=tworandomstate,n_jobs=22)
                elif  chaocanshumoxing == "Bayesian optimization":
                    best_model = BayesSearchCV(bestmodel,params,n_iter=twoniters,cv=allcv,n_jobs=-1,random_state = tworandomstate,verbose=allverbose)
                best_model.fit(X_train, y_train)
                best_moxing_model = best_model.best_estimator_
                best_params = best_model.best_params_
                joblib.dump(best_model, './DataStatistics/' + modelfusionall_words + '/' +  modelfusionall_words +'.pkl')
                st.markdown(f"The optimal parameters of the model are:{best_params}")
                with open('./DataStatistics/' + modelfusionall_words + '/' +  modelfusionall_words +'_best_params.txt', "w") as best_paramsb:
                    best_paramsb.write("The optimal parameters of the model are:\n" + str(best_params))
            else:
                best_model = RandomForestClassifier(**shenmebestmodel,random_state=moxingderandomstate)
                best_model.fit(X_train, y_train)
                best_params = "None"
            y_truetrain = y_train
            y_truetest = y_test
            y_predtrain = best_model.predict(X_train)
            y_predtest = best_model.predict(X_test)
            y_protrain = best_model.predict_proba(X_train)
            y_protest = best_model.predict_proba(X_test)
            st.markdown("#### "+mingshishenme+"Here are the results:")
            ACCAUEDENGDENGtrain(y_truetrain,y_protrain,y_predtrain,X_train,mingshishenme)
            ACCAUEDENGDENGTEST(y_truetest, y_protest, y_predtest, X_test, modelfusionall_words,mingshishenme)
            reporttrain = classification_report(y_truetrain, y_predtrain, output_dict=True)
            reporttest = classification_report(y_truetest, y_predtest, output_dict=True)
            dfroctrain = pd.DataFrame(reporttrain).transpose()
            dfroctest = pd.DataFrame(reporttest).transpose()
            hebingcccc = pd.concat([moxing_test,moxing_train])
            if mingshishenme != "test":
                np.savetxt("./DataStatistics/Random Forest/RF_coefficient.csv", best_moxing_model.feature_importances_, delimiter=',')
                coef1 = pd.read_csv(r"./DataStatistics/Random Forest/RF_coefficient.csv", header=None)
                coef1_array = np.array(coef1.stack())
                coef1_list = coef1_array.tolist()
                df1 = pd.DataFrame(coef1_list)
                feature2 = list(X_train.columns)
                df2 = pd.DataFrame(feature2)
                df2.to_csv("./DataStatistics/Random Forest/RF_coefficient.csv", index=False)
                hebingQUANZHONG = pd.concat([df2, df1], axis=1)
                hebingQUANZHONG.columns = ['Features', 'Coefficients']
                quanzhognhuizhi(hebingQUANZHONG,modelfusionall_words)
            with pd.ExcelWriter("./DataStatistics/Random Forest/RF_" + mingshishenme + "_Report.xlsx", engine='xlsxwriter') as writer:
                hebingcccc.to_excel(writer, index=False, sheet_name="RF" + "_" + mingshishenme)
                moxing_train.to_excel(writer, index=False, sheet_name="RF_train" + "_" + mingshishenme)
                moxing_test.to_excel(writer, index=False, sheet_name="RF_test" + "_" + mingshishenme)
                if mingshishenme != "test":
                    hebingQUANZHONG.to_excel(writer, index=True, sheet_name="RF_coefficient" + "_" + mingshishenme)
                dataTRAINPONGJIAZHIBIAO.to_excel(writer,index=True, sheet_name="RF_ROCtrain" + "_" + mingshishenme + "_report_1")
                dataTESTPONGJIAZHIBIAO.to_excel(writer,index=True, sheet_name="RF_ROCtest" + "_" + mingshishenme + "_report_1")
                dfroctrain.to_excel(writer,index=True, sheet_name="RF_ROCtrain" + "_" + mingshishenme + "_report_2")
                dfroctest.to_excel(writer,index=True, sheet_name="RF_ROCtest" + "_" + mingshishenme + "_report_2")
            try:
                os.remove("./DataStatistics/Random Forest/RF_coefficient.csv")
            except:
                print("no")

            return best_params
        st.markdown("---")
        XUNLIANYANZHENGDEYANZHENGMINGtrain = "train"
        shenmebestmodel1 = "none"
        shenmebestmodel = rfhanshu(X_train_TRAIN, y_train_TRAIN, XUNLIANYANZHENGDEYANZHENGMINGtrain,
                                   shenmebestmodel1, train_size)
        st.markdown("---")
        XUNLIANYANZHENGDEYANZHENGMINGtest = "test"
        rfhanshu(X_test_TEST, y_test_TEST,XUNLIANYANZHENGDEYANZHENGMINGtest,shenmebestmodel,test_size)
        st.markdown("---")

    if modelfusionall_words == "Decision Tree":
        os.makedirs('./DataStatistics/Decision Tree/', exist_ok=True)
        def DThanshu(XX,yy,mingshishenme,shenmebestmodel,testortrain_size):
            X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size= test_size11nei, random_state=42)
            bestmodel = DecisionTreeClassifier(random_state= moxingderandomstate )
            if mingshishenme != "test":
                if chaocanshumoxing == "Grid search":
                    best_model = GridSearchCV(bestmodel, param_grid=params, cv=allcv, n_jobs=-1,verbose=allverbose)
                elif  chaocanshumoxing == "Random search":
                    best_model = RandomizedSearchCV(estimator=bestmodel,param_distributions=params,n_iter=twoniters,cv=allcv,verbose=allverbose,random_state=tworandomstate,n_jobs=22)
                elif  chaocanshumoxing == "Bayesian optimization":
                    best_model = BayesSearchCV(bestmodel,params,n_iter=twoniters,cv=allcv,n_jobs=-1,random_state = tworandomstate,verbose=allverbose)
                best_model.fit(X_train, y_train)
                best_moxing_model = best_model.best_estimator_
                best_params = best_model.best_params_
                st.markdown(f"The optimal parameters of the model are:{best_params}")
                with open('./DataStatistics/' + modelfusionall_words + '/' +  modelfusionall_words +'_best_params.txt', "w") as best_paramsb:
                    best_paramsb.write("The optimal parameters of the model are:\n" + str(best_params))
                joblib.dump(best_model, './DataStatistics/' + modelfusionall_words + '/' +  modelfusionall_words +'.pkl')
            else:
                best_model = DecisionTreeClassifier(**shenmebestmodel, random_state=moxingderandomstate)
                best_model.fit(X_train, y_train)
                best_params = "None"
            y_truetrain = y_train
            y_truetest = y_test
            y_predtrain = best_model.predict(X_train)
            y_predtest = best_model.predict(X_test)
            y_protrain = best_model.predict_proba(X_train)
            y_protest = best_model.predict_proba(X_test)
            st.markdown("#### "+mingshishenme+"Here are the results:")
            ACCAUEDENGDENGtrain(y_truetrain,y_protrain,y_predtrain,X_train,mingshishenme)
            ACCAUEDENGDENGTEST(y_truetest, y_protest, y_predtest, X_test, modelfusionall_words,mingshishenme)
            reporttrain = classification_report(y_truetrain, y_predtrain, output_dict=True)
            reporttest = classification_report(y_truetest, y_predtest, output_dict=True)
            dfroctrain = pd.DataFrame(reporttrain).transpose()
            dfroctest = pd.DataFrame(reporttest).transpose()
            hebingcccc = pd.concat([moxing_test,moxing_train])
            if mingshishenme != "test":
                np.savetxt("./DataStatistics/Decision Tree/DT_coefficient.csv", best_moxing_model.feature_importances_, delimiter=',')
                coef1 = pd.read_csv(r"./DataStatistics/Decision Tree/DT_coefficient.csv", header=None)
                coef1_array = np.array(coef1.stack())
                coef1_list = coef1_array.tolist()
                df1 = pd.DataFrame(coef1_list)
                feature2 = list(X_train.columns)
                df2 = pd.DataFrame(feature2)
                df2.to_csv("./DataStatistics/Decision Tree/DT_coefficient.csv", index=False)
                hebingQUANZHONG = pd.concat([df2, df1], axis=1)
                hebingQUANZHONG.columns = ['Features', 'Coefficients']
                quanzhognhuizhi(hebingQUANZHONG,modelfusionall_words)
            with pd.ExcelWriter("./DataStatistics/Decision Tree/DT_" + mingshishenme + "_Report.xlsx", engine='xlsxwriter') as writer:
                hebingcccc.to_excel(writer, index=False, sheet_name="DT" + "_" + mingshishenme)
                moxing_train.to_excel(writer, index=False, sheet_name="DT_train" + "_" + mingshishenme)
                moxing_test.to_excel(writer, index=False, sheet_name="DT_test" + "_" + mingshishenme)
                if mingshishenme != "test":
                    hebingQUANZHONG.to_excel(writer, index=True, sheet_name="DT_coefficient" + "_" + mingshishenme)
                dataTRAINPONGJIAZHIBIAO.to_excel(writer,index=True, sheet_name="DT_ROCtrain" + "_" + mingshishenme + "_report_1")
                dataTESTPONGJIAZHIBIAO.to_excel(writer,index=True, sheet_name="DT_ROCtest" + "_" + mingshishenme + "_report_1")
                dfroctrain.to_excel(writer,index=True, sheet_name="DT_ROCtrain" + "_" + mingshishenme + "_report_2")
                dfroctest.to_excel(writer,index=True, sheet_name="DT_ROCtest" + "_" + mingshishenme + "_report_2")
            try:
                os.remove("./DataStatistics/Decision Tree/DT_coefficient.csv")
            except:
                print("no")
            return best_params
        st.markdown("---")
        XUNLIANYANZHENGDEYANZHENGMINGtrain = "train"
        shenmebestmodel1 = "none"
        shenmebestmodel = DThanshu(X_train_TRAIN, y_train_TRAIN, XUNLIANYANZHENGDEYANZHENGMINGtrain,
                                   shenmebestmodel1, train_size)
        st.markdown("---")
        XUNLIANYANZHENGDEYANZHENGMINGtest = "test"
        DThanshu(X_test_TEST, y_test_TEST,XUNLIANYANZHENGDEYANZHENGMINGtest,shenmebestmodel,test_size)
        st.markdown("---")

    if modelfusionall_words == "Extra Trees":
        os.makedirs('./DataStatistics/Extra Trees/', exist_ok=True)
        def EThanshu(XX,yy,mingshishenme,shenmebestmodel,testortrain_size):
            X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size= test_size11nei, random_state=42)
            bestmodel = ExtraTreesClassifier(random_state= moxingderandomstate)
            if mingshishenme != "test":
                if chaocanshumoxing == "Grid search":
                    best_model = GridSearchCV(bestmodel, param_grid=params, cv=allcv, n_jobs=-1,verbose=allverbose)
                elif  chaocanshumoxing == "Random search":
                    best_model = RandomizedSearchCV(estimator=bestmodel,param_distributions=params,n_iter=twoniters,cv=allcv,verbose=allverbose,random_state=tworandomstate,n_jobs=22)
                elif  chaocanshumoxing == "Bayesian optimization":
                    best_model = BayesSearchCV(bestmodel,params,n_iter=twoniters,cv=allcv,n_jobs=-1,random_state = tworandomstate,verbose=allverbose)
                best_model.fit(X_train, y_train)
                best_moxing_model = best_model.best_estimator_
                best_params = best_model.best_params_
                st.markdown(f"The optimal parameters of the model are:{best_params}")
                joblib.dump(best_model, './DataStatistics/' + modelfusionall_words + '/' +  modelfusionall_words +'.pkl')
                with open('./DataStatistics/' + modelfusionall_words + '/' +  modelfusionall_words +'_best_params.txt', "w") as best_paramsb:
                    best_paramsb.write("The optimal parameters of the model are:\n" + str(best_params))
            else:
                best_model = ExtraTreesClassifier(**shenmebestmodel,random_state=moxingderandomstate)
                best_model.fit(X_train, y_train)
                best_params = "None"
            y_truetrain = y_train
            y_truetest = y_test
            y_predtrain = best_model.predict(X_train)
            y_predtest = best_model.predict(X_test)
            y_protrain = best_model.predict_proba(X_train)
            y_protest = best_model.predict_proba(X_test)
            st.markdown("#### "+mingshishenme+"Here are the results:")
            ACCAUEDENGDENGtrain(y_truetrain,y_protrain,y_predtrain,X_train,mingshishenme)
            ACCAUEDENGDENGTEST(y_truetest, y_protest, y_predtest, X_test, modelfusionall_words,mingshishenme)
            reporttrain = classification_report(y_truetrain, y_predtrain, output_dict=True)
            reporttest = classification_report(y_truetest, y_predtest, output_dict=True)
            dfroctrain = pd.DataFrame(reporttrain).transpose()
            dfroctest = pd.DataFrame(reporttest).transpose()
            hebingcccc = pd.concat([moxing_test,moxing_train])
            if mingshishenme != "test":
                np.savetxt("./DataStatistics/Extra Trees/ET_coefficient.csv", best_moxing_model.feature_importances_, delimiter=',')
                coef1 = pd.read_csv(r"./DataStatistics/Extra Trees/ET_coefficient.csv", header=None)
                coef1_array = np.array(coef1.stack())
                coef1_list = coef1_array.tolist()
                df1 = pd.DataFrame(coef1_list)
                feature2 = list(X_train.columns)
                df2 = pd.DataFrame(feature2)
                df2.to_csv("./DataStatistics/Extra Trees/ET_coefficient.csv", index=False)
                hebingQUANZHONG = pd.concat([df2, df1], axis=1)
                hebingQUANZHONG.columns = ['Features', 'Coefficients']
                quanzhognhuizhi(hebingQUANZHONG,modelfusionall_words)
            with pd.ExcelWriter("./DataStatistics/Extra Trees/ET_" + mingshishenme + "_Report.xlsx", engine='xlsxwriter') as writer:
                hebingcccc.to_excel(writer, index=False, sheet_name="ET" + "_" + mingshishenme)
                moxing_train.to_excel(writer, index=False, sheet_name="ET_train" + "_" + mingshishenme)
                moxing_test.to_excel(writer, index=False, sheet_name="ET_test" + "_" + mingshishenme)
                if mingshishenme != "test":
                    hebingQUANZHONG.to_excel(writer, index=True, sheet_name="ET_coefficient" + "_" + mingshishenme)
                dataTRAINPONGJIAZHIBIAO.to_excel(writer,index=True, sheet_name="ET_ROCtrain" + "_" + mingshishenme + "_report_1")
                dataTESTPONGJIAZHIBIAO.to_excel(writer,index=True, sheet_name="ET_ROCtest" + "_" + mingshishenme + "_report_1")
                dfroctrain.to_excel(writer,index=True, sheet_name="ET_ROCtrain" + "_" + mingshishenme + "_report_2")
                dfroctest.to_excel(writer,index=True, sheet_name="ET_ROCtest" + "_" + mingshishenme + "_report_2")
            try:
                os.remove("./DataStatistics/Extra Trees/ET_coefficient.csv")
            except:
                print("no")
            return best_params
        st.markdown("---")
        XUNLIANYANZHENGDEYANZHENGMINGtrain = "train"
        shenmebestmodel1 = "none"
        shenmebestmodel = EThanshu(X_train_TRAIN, y_train_TRAIN, XUNLIANYANZHENGDEYANZHENGMINGtrain,
                                   shenmebestmodel1, train_size)
        st.markdown("---")
        XUNLIANYANZHENGDEYANZHENGMINGtest = "test"
        EThanshu(X_test_TEST, y_test_TEST,XUNLIANYANZHENGDEYANZHENGMINGtest,shenmebestmodel,test_size)
        st.markdown("---")

    if modelfusionall_words == "SVM":
        os.makedirs('./DataStatistics/SVM/', exist_ok=True)
        bestmodel = svm.SVC(probability=True)
        def svmhanshu(XX,yy,mingshishenme,shenmebestmodel,testortrain_size):
            X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size= test_size11nei, random_state=42)
            if mingshishenme != "test":
                if chaocanshumoxing == "Grid search":
                    best_model = GridSearchCV(bestmodel, param_grid=params, cv=allcv, n_jobs=-1,verbose=allverbose)
                elif  chaocanshumoxing == "Random search":
                    best_model = RandomizedSearchCV(estimator=bestmodel,param_distributions=params,n_iter=twoniters,cv=allcv,verbose=allverbose,random_state=tworandomstate,n_jobs=22)
                elif  chaocanshumoxing == "Bayesian optimization":
                    best_model = BayesSearchCV(bestmodel,params,n_iter=twoniters,cv=allcv,n_jobs=-1,random_state = tworandomstate,verbose=allverbose)
                best_model.fit(X_train, y_train)
                best_moxing_model = best_model.best_estimator_
                best_params = best_model.best_params_
                st.markdown(f"The optimal parameters of the model are:{best_params}")
                joblib.dump(best_model, './DataStatistics/' + modelfusionall_words + '/' +  modelfusionall_words +'.pkl')
                with open('./DataStatistics/' + modelfusionall_words + '/' +  modelfusionall_words +'_best_params.txt', "w") as best_paramsb:
                    best_paramsb.write("The optimal parameters of the model are:\n" + str(best_params))
            else:
                best_model = svm.SVC(**shenmebestmodel)
                best_model.fit(X_train, y_train)
                best_params = "None"
            y_truetrain = y_train
            y_truetest = y_test
            y_predtrain = best_model.predict(X_train)
            y_predtest = best_model.predict(X_test)
            y_protrain = best_model.predict_proba(X_train)
            y_protest = best_model.predict_proba(X_test)
            st.markdown("#### "+mingshishenme+"Here are the results:")
            ACCAUEDENGDENGtrain(y_truetrain,y_protrain,y_predtrain,X_train,mingshishenme)
            ACCAUEDENGDENGTEST(y_truetest, y_protest, y_predtest, X_test, modelfusionall_words,mingshishenme)
            reporttrain = classification_report(y_truetrain, y_predtrain, output_dict=True)
            reporttest = classification_report(y_truetest, y_predtest, output_dict=True)
            dfroctrain = pd.DataFrame(reporttrain).transpose()
            dfroctest = pd.DataFrame(reporttest).transpose()
            hebingcccc = pd.concat([moxing_test,moxing_train])
            if mingshishenme != "test":
                if hasattr(best_moxing_model, 'coef_'):
                    np.savetxt("./DataStatistics/SVM/SVM_coefficient.csv", best_moxing_model.coef_, delimiter=',')
                    coef1 = pd.read_csv(r"./DataStatistics/SVM/SVM_coefficient.csv", header=None)
                    coef1_array = np.array(coef1.stack())
                    coef1_list = coef1_array.tolist()
                    df1 = pd.DataFrame(coef1_list)
                    feature2 = list(X_train.columns)
                    df2 = pd.DataFrame(feature2)
                    df2.to_csv("./DataStatistics/SVM/SVM_coefficient.csv", index=False)
                    hebingQUANZHONG = pd.concat([df2, df1], axis=1)
                    hebingQUANZHONG.columns = ['Features', 'Coefficients']
                    quanzhognhuizhi(hebingQUANZHONG,modelfusionall_words)
                    with pd.ExcelWriter("./DataStatistics/SVM/SVM" + "_" + mingshishenme + "_Report.xlsx", engine='xlsxwriter') as writer:
                        hebingcccc.to_excel(writer, index=False, sheet_name="SVM" + "_" + mingshishenme)
                        moxing_train.to_excel(writer, index=False, sheet_name="SVM_train" + "_" + mingshishenme)
                        moxing_test.to_excel(writer, index=False, sheet_name="SVM_test" + "_" + mingshishenme)
                        hebingQUANZHONG.to_excel(writer, index=True, sheet_name="SVM_coefficient" + "_" + mingshishenme)
                        dataTRAINPONGJIAZHIBIAO.to_excel(writer,index=True, sheet_name="SVM_ROCtrain" + "_" + mingshishenme + "_report_1")
                        dataTESTPONGJIAZHIBIAO.to_excel(writer,index=True, sheet_name="SVM_ROCtest" + "_" + mingshishenme + "_report_1")
                        dfroctrain.to_excel(writer,index=True, sheet_name="SVM_ROCtrain" + "_" + mingshishenme + "_report_2")
                        dfroctest.to_excel(writer,index=True, sheet_name="SVM_ROCtest" + "_" + mingshishenme + "_report_2")
                    try:
                        os.remove("./DataStatistics/SVM/SVM_coefficient.csv")
                    except:
                        print("no")
                else:
                    st.warning("The best models have no weight attributes!")
                    with pd.ExcelWriter("./DataStatistics/SVM/SVM" + "_" + mingshishenme + "_Report.xlsx", engine='xlsxwriter') as writer:
                        hebingcccc.to_excel(writer, index=False, sheet_name="SVM" + "_" + mingshishenme)
                        moxing_train.to_excel(writer, index=False, sheet_name="SVM_train" + "_" + mingshishenme)
                        moxing_test.to_excel(writer, index=False, sheet_name="SVM_test" + "_" + mingshishenme)
                        dataTRAINPONGJIAZHIBIAO.to_excel(writer,index=True, sheet_name="SVM_ROCtrain" + "_" + mingshishenme + "_report_1")
                        dataTESTPONGJIAZHIBIAO.to_excel(writer,index=True, sheet_name="SVM_ROCtest" + "_" + mingshishenme + "_report_1")
                        dfroctrain.to_excel(writer,index=True, sheet_name="SVM_ROCtrain" + "_" + mingshishenme + "_report_2")
                        dfroctest.to_excel(writer,index=True, sheet_name="SVM_ROCtest" + "_" + mingshishenme + "_report_2")
            else:
                with pd.ExcelWriter("./DataStatistics/SVM/SVM" + "_" + mingshishenme + "_Report.xlsx",engine='xlsxwriter') as writer:
                    hebingcccc.to_excel(writer, index=False, sheet_name="SVM" + "_" + mingshishenme)
                    moxing_train.to_excel(writer, index=False, sheet_name="SVM_train" + "_" + mingshishenme)
                    moxing_test.to_excel(writer, index=False, sheet_name="SVM_test" + "_" + mingshishenme)
                    dataTRAINPONGJIAZHIBIAO.to_excel(writer, index=True,sheet_name="SVM_ROCtrain" + "_" + mingshishenme + "_report_1")
                    dataTESTPONGJIAZHIBIAO.to_excel(writer, index=True,sheet_name="SVM_ROCtest" + "_" + mingshishenme + "_report_1")
                    dfroctrain.to_excel(writer, index=True,sheet_name="SVM_ROCtrain" + "_" + mingshishenme + "_report_2")
                    dfroctest.to_excel(writer, index=True,sheet_name="SVM_ROCtest" + "_" + mingshishenme + "_report_2")
            return best_params
        st.markdown("---")
        XUNLIANYANZHENGDEYANZHENGMINGtrain = "train"
        shenmebestmodel1 = "none"
        shenmebestmodel = svmhanshu(X_train_TRAIN, y_train_TRAIN, XUNLIANYANZHENGDEYANZHENGMINGtrain,
                                   shenmebestmodel1, train_size)
        st.markdown("---")
        XUNLIANYANZHENGDEYANZHENGMINGtest = "test"
        svmhanshu(X_test_TEST, y_test_TEST,XUNLIANYANZHENGDEYANZHENGMINGtest,shenmebestmodel,test_size)
        st.markdown("---")

    if modelfusionall_words == "Logistic Regression":
        os.makedirs('./DataStatistics/Logistic Regression/', exist_ok=True)
        bestmodel = LogisticRegression()
        def lrhanshu(XX,yy,mingshishenme,shenmebestmodel,testortrain_size):
            X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size= test_size11nei, random_state=42)
            if mingshishenme != "test":
                if chaocanshumoxing == "Grid search":
                    best_model = GridSearchCV(bestmodel, param_grid=params, cv=allcv, n_jobs=-1,verbose=allverbose)
                elif  chaocanshumoxing == "Random search":
                    best_model = RandomizedSearchCV(estimator=bestmodel,param_distributions=params,n_iter=twoniters,cv=allcv,verbose=allverbose,random_state=tworandomstate,n_jobs=22)
                elif  chaocanshumoxing == "Bayesian optimization":
                    best_model = BayesSearchCV(bestmodel,params,n_iter=twoniters,cv=allcv,n_jobs=-1,random_state = tworandomstate,verbose=allverbose)
                best_model.fit(X_train, y_train)
                best_moxing_model = best_model.best_estimator_
                best_params = best_model.best_params_
                st.markdown(f"The optimal parameters of the model are:{best_params}")
                joblib.dump(best_model, './DataStatistics/' + modelfusionall_words + '/' +  modelfusionall_words +'.pkl')
                joblib.dump(best_model, './DataStatistics/' + modelfusionall_words + '/' +  modelfusionall_words +'.pkl')
                with open('./DataStatistics/' + modelfusionall_words + '/' +  modelfusionall_words +'_best_params.txt', "w") as best_paramsb:
                    best_paramsb.write("The optimal parameters of the model are:\n" + str(best_params))
            else:
                best_model = LogisticRegression(**shenmebestmodel)
                best_model.fit(X_train, y_train)
                best_params = "None"
            y_truetrain = y_train
            y_truetest = y_test
            y_predtrain = best_model.predict(X_train)
            y_predtest = best_model.predict(X_test)
            y_protrain = best_model.predict_proba(X_train)
            y_protest = best_model.predict_proba(X_test)
            st.markdown("#### "+mingshishenme+"Here are the results:")
            ACCAUEDENGDENGtrain(y_truetrain,y_protrain,y_predtrain,X_train,mingshishenme)
            ACCAUEDENGDENGTEST(y_truetest, y_protest, y_predtest, X_test, modelfusionall_words,mingshishenme)
            reporttrain = classification_report(y_truetrain, y_predtrain, output_dict=True)
            reporttest = classification_report(y_truetest, y_predtest, output_dict=True)
            dfroctrain = pd.DataFrame(reporttrain).transpose()
            dfroctest = pd.DataFrame(reporttest).transpose()
            hebingcccc = pd.concat([moxing_test, moxing_train])
            if mingshishenme != "test":
                if hasattr(best_moxing_model, 'coef_'):
                    np.savetxt("./DataStatistics/Logistic Regression/LR_coefficient.csv",
                               best_moxing_model.coef_, delimiter=',')
                    coef1 = pd.read_csv(r"./DataStatistics/Logistic Regression/LR_coefficient.csv", header=None)
                    coef1_array = np.array(coef1.stack())
                    coef1_list = coef1_array.tolist()
                    df1 = pd.DataFrame(coef1_list)
                    feature2 = list(X_train.columns)
                    df2 = pd.DataFrame(feature2)
                    df2.to_csv("./DataStatistics/Logistic Regression/LR_coefficient.csv", index=False)
                    hebingQUANZHONG = pd.concat([df2, df1], axis=1)
                    hebingQUANZHONG.columns = ['Features', 'Coefficients']
                    quanzhognhuizhi(hebingQUANZHONG, modelfusionall_words)
                    with pd.ExcelWriter("./DataStatistics/Logistic Regression/LR" + "_" + mingshishenme + "_Report.xlsx",
                                        engine='xlsxwriter') as writer:
                        hebingcccc.to_excel(writer, index=False, sheet_name="LR" + "_" + mingshishenme)
                        moxing_train.to_excel(writer, index=False, sheet_name="LR_train" + "_" + mingshishenme)
                        moxing_test.to_excel(writer, index=False, sheet_name="LR_test" + "_" + mingshishenme)
                        hebingQUANZHONG.to_excel(writer, index=True, sheet_name="LR_coefficient" + "_" + mingshishenme)
                        dataTRAINPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="LR_ROCtrain" + "_" + mingshishenme + "_report_1")
                        dataTESTPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="LR_ROCtest" + "_" + mingshishenme + "_report_1")
                        dfroctrain.to_excel(writer, index=True, sheet_name="LR_ROCtrain" + "_" + mingshishenme + "_report_2")
                        dfroctest.to_excel(writer, index=True, sheet_name="LR_ROCtest" + "_" + mingshishenme + "_report_2")
                    try:
                        os.remove("./DataStatistics/Logistic Regression/LR_coefficient.csv")
                    except:
                        print("no")
                else:
                    st.warning("The best models have no weight attributes!")
                    with pd.ExcelWriter("./DataStatistics/Logistic Regression/LR" + "_" + mingshishenme + "_Report.xlsx",
                                        engine='xlsxwriter') as writer:
                        hebingcccc.to_excel(writer, index=False, sheet_name="LR" + "_" + mingshishenme)
                        moxing_train.to_excel(writer, index=False, sheet_name="LR_train" + "_" + mingshishenme)
                        moxing_test.to_excel(writer, index=False, sheet_name="LR_test" + "_" + mingshishenme)
                        dataTRAINPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="LR_ROCtrain" + "_" + mingshishenme + "_report_1")
                        dataTESTPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="LR_ROCtest" + "_" + mingshishenme + "_report_1")
                        dfroctrain.to_excel(writer, index=True, sheet_name="LR_ROCtrain" + "_" + mingshishenme + "_report_2")
                        dfroctest.to_excel(writer, index=True, sheet_name="LR_ROCtest" + "_" + mingshishenme + "_report_2")
            else:
                with pd.ExcelWriter("./DataStatistics/Logistic Regression/LR" + "_" + mingshishenme + "_Report.xlsx",engine='xlsxwriter') as writer:
                    hebingcccc.to_excel(writer, index=False, sheet_name="LR" + "_" + mingshishenme)
                    moxing_train.to_excel(writer, index=False, sheet_name="LR_train" + "_" + mingshishenme)
                    moxing_test.to_excel(writer, index=False, sheet_name="LR_test" + "_" + mingshishenme)
                    dataTRAINPONGJIAZHIBIAO.to_excel(writer, index=True,sheet_name="LR_ROCtrain" + "_" + mingshishenme + "_report_1")
                    dataTESTPONGJIAZHIBIAO.to_excel(writer, index=True,sheet_name="LR_ROCtest" + "_" + mingshishenme + "_report_1")
                    dfroctrain.to_excel(writer, index=True,sheet_name="LR_ROCtrain" + "_" + mingshishenme + "_report_2")
                    dfroctest.to_excel(writer, index=True,sheet_name="LR_ROCtest" + "_" + mingshishenme + "_report_2")
            return best_params
        st.markdown("---")
        XUNLIANYANZHENGDEYANZHENGMINGtrain = "train"
        shenmebestmodel1 = "none"
        shenmebestmodel = lrhanshu(X_train_TRAIN, y_train_TRAIN, XUNLIANYANZHENGDEYANZHENGMINGtrain,
                                   shenmebestmodel1, train_size)
        st.markdown("---")
        XUNLIANYANZHENGDEYANZHENGMINGtest = "test"
        lrhanshu(X_test_TEST, y_test_TEST,XUNLIANYANZHENGDEYANZHENGMINGtest,shenmebestmodel,test_size)
        st.markdown("---")

    if modelfusionall_words == "SGD":
        os.makedirs('./DataStatistics/SGD/', exist_ok=True)
        bestmodel = SGDClassifier()
        def sgdhanshu(XX,yy,mingshishenme,shenmebestmodel,testortrain_size):
            X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size= test_size11nei, random_state=42)
            if mingshishenme != "test":
                if chaocanshumoxing == "Grid search":
                    best_model = GridSearchCV(bestmodel, param_grid=params, cv=allcv, n_jobs=-1,verbose=allverbose)
                elif  chaocanshumoxing == "Random search":
                    best_model = RandomizedSearchCV(estimator=bestmodel,param_distributions=params,n_iter=twoniters,cv=allcv,verbose=allverbose,random_state=tworandomstate,n_jobs=22)
                elif  chaocanshumoxing == "Bayesian optimization":
                    best_model = BayesSearchCV(bestmodel,params,n_iter=twoniters,cv=allcv,n_jobs=-1,random_state = tworandomstate,verbose=allverbose)
                best_model.fit(X_train, y_train)
                best_moxing_model = best_model.best_estimator_
                best_params = best_model.best_params_
                st.markdown(f"The optimal parameters of the model are:{best_params}")
                joblib.dump(best_model,
                            './DataStatistics/' + modelfusionall_words + '/' + modelfusionall_words + '.pkl')
                with open('./DataStatistics/' + modelfusionall_words + '/' +  modelfusionall_words +'_best_params.txt', "w") as best_paramsb:
                    best_paramsb.write("The optimal parameters of the model are:\n" + str(best_params))
            else:
                best_model = SGDClassifier(**shenmebestmodel)
                best_model.fit(X_train, y_train)
                best_params = "None"
            y_truetrain = y_train
            y_truetest = y_test
            y_predtrain = best_model.predict(X_train)
            y_predtest = best_model.predict(X_test)
            y_protrain = best_model.predict_proba(X_train)
            y_protest = best_model.predict_proba(X_test)
            st.markdown("#### "+mingshishenme+"Here are the results:")
            ACCAUEDENGDENGtrain(y_truetrain,y_protrain,y_predtrain,X_train,mingshishenme)
            ACCAUEDENGDENGTEST(y_truetest, y_protest, y_predtest, X_test, modelfusionall_words,mingshishenme)
            reporttrain = classification_report(y_truetrain, y_predtrain, output_dict=True)
            reporttest = classification_report(y_truetest, y_predtest, output_dict=True)
            dfroctrain = pd.DataFrame(reporttrain).transpose()
            dfroctest = pd.DataFrame(reporttest).transpose()
            hebingcccc = pd.concat([moxing_test, moxing_train])
            if mingshishenme != "test":
                if hasattr(best_moxing_model, 'coef_'):
                    np.savetxt("./DataStatistics/SGD/SGD_coefficient.csv",
                               best_moxing_model.coef_, delimiter=',')
                    coef1 = pd.read_csv(r"./DataStatistics/SGD/SGD_coefficient.csv", header=None)
                    coef1_array = np.array(coef1.stack())
                    coef1_list = coef1_array.tolist()
                    df1 = pd.DataFrame(coef1_list)
                    feature2 = list(X_train.columns)
                    df2 = pd.DataFrame(feature2)
                    df2.to_csv("./DataStatistics/SGD/SGD_coefficient.csv", index=False)
                    hebingQUANZHONG = pd.concat([df2, df1], axis=1)
                    hebingQUANZHONG.columns = ['Features', 'Coefficients']
                    quanzhognhuizhi(hebingQUANZHONG, modelfusionall_words)
                    with pd.ExcelWriter("./DataStatistics/SGD/SGD" + "_" + mingshishenme + "_Report.xlsx",
                                        engine='xlsxwriter') as writer:
                        hebingcccc.to_excel(writer, index=False, sheet_name="SGD" + "_" + mingshishenme)
                        moxing_train.to_excel(writer, index=False, sheet_name="SGD_train" + "_" + mingshishenme)
                        moxing_test.to_excel(writer, index=False, sheet_name="SGD_test" + "_" + mingshishenme)
                        hebingQUANZHONG.to_excel(writer, index=True, sheet_name="SGD_coefficient" + "_" + mingshishenme)
                        dataTRAINPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="SGD_ROCtrain" + "_" + mingshishenme + "_report_1")
                        dataTESTPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="SGD_ROCtest" + "_" + mingshishenme + "_report_1")
                        dfroctrain.to_excel(writer, index=True, sheet_name="SGD_ROCtrain" + "_" + mingshishenme + "_report_2")
                        dfroctest.to_excel(writer, index=True, sheet_name="SGD_ROCtest" + "_" + mingshishenme + "_report_2")
                    try:
                        os.remove("./DataStatistics/SGD/SGD_coefficient.csv")
                    except:
                        print("no")
                else:
                    st.warning("The best models have no weight attributes!")
                    with pd.ExcelWriter("./DataStatistics/SGD/SGD" + "_" + mingshishenme + "_Report.xlsx",
                                        engine='xlsxwriter') as writer:
                        hebingcccc.to_excel(writer, index=False, sheet_name="SGD" + "_" + mingshishenme)
                        moxing_train.to_excel(writer, index=False, sheet_name="SGD_train" + "_" + mingshishenme)
                        moxing_test.to_excel(writer, index=False, sheet_name="SGD_test" + "_" + mingshishenme)
                        dataTRAINPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="SGD_ROCtrain" + "_" + mingshishenme + "_report_1")
                        dataTESTPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="SGD_ROCtest" + "_" + mingshishenme + "_report_1")
                        dfroctrain.to_excel(writer, index=True, sheet_name="SGD_ROCtrain" + "_" + mingshishenme + "_report_2")
                        dfroctest.to_excel(writer, index=True, sheet_name="SGD_ROCtest" + "_" + mingshishenme + "_report_2")
            else:
                with pd.ExcelWriter("./DataStatistics/SGD/SGD" + "_" + mingshishenme + "_Report.xlsx",engine='xlsxwriter') as writer:
                    hebingcccc.to_excel(writer, index=False, sheet_name="SGD" + "_" + mingshishenme)
                    moxing_train.to_excel(writer, index=False, sheet_name="SGD_train" + "_" + mingshishenme)
                    moxing_test.to_excel(writer, index=False, sheet_name="SGD_test" + "_" + mingshishenme)
                    dataTRAINPONGJIAZHIBIAO.to_excel(writer, index=True,sheet_name="SGD_ROCtrain" + "_" + mingshishenme + "_report_1")
                    dataTESTPONGJIAZHIBIAO.to_excel(writer, index=True,sheet_name="SGD_ROCtest" + "_" + mingshishenme + "_report_1")
                    dfroctrain.to_excel(writer, index=True,sheet_name="SGD_ROCtrain" + "_" + mingshishenme + "_report_2")
                    dfroctest.to_excel(writer, index=True,sheet_name="SGD_ROCtest" + "_" + mingshishenme + "_report_2")
            return best_params
        st.markdown("---")
        XUNLIANYANZHENGDEYANZHENGMINGtrain = "train"
        shenmebestmodel1 = "none"
        shenmebestmodel=sgdhanshu(X_train_TRAIN, y_train_TRAIN,XUNLIANYANZHENGDEYANZHENGMINGtrain,shenmebestmodel1,train_size)
        st.markdown("---")
        XUNLIANYANZHENGDEYANZHENGMINGtest = "test"
        sgdhanshu(X_test_TEST, y_test_TEST,XUNLIANYANZHENGDEYANZHENGMINGtest,shenmebestmodel,test_size)
        st.markdown("---")

    if modelfusionall_words == "KNN":
        os.makedirs('./DataStatistics/KNN/', exist_ok=True)
        bestmodel = KNeighborsClassifier()
        def KNNhanshu(XX,yy,mingshishenme,shenmebestmodel,testortrain_size):
            X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size= test_size11nei, random_state=42)
            if mingshishenme != "test":
                if chaocanshumoxing == "Grid search":
                    best_model = GridSearchCV(bestmodel, param_grid=params, cv=allcv, n_jobs=-1,verbose=allverbose)
                elif  chaocanshumoxing == "Random search":
                    best_model = RandomizedSearchCV(estimator=bestmodel,param_distributions=params,n_iter=twoniters,cv=allcv,verbose=allverbose,random_state=tworandomstate,n_jobs=22)
                elif  chaocanshumoxing == "Bayesian optimization":
                    best_model = BayesSearchCV(bestmodel,params,n_iter=twoniters,cv=allcv,n_jobs=-1,random_state = tworandomstate,verbose=allverbose)
                best_model.fit(X_train, y_train)
                best_moxing_model = best_model.best_estimator_
                best_params = best_model.best_params_
                st.markdown(f"The optimal parameters of the model are:{best_params}")
                joblib.dump(best_model,
                            './DataStatistics/' + modelfusionall_words + '/' + modelfusionall_words + '.pkl')
                with open('./DataStatistics/' + modelfusionall_words + '/' +  modelfusionall_words +'_best_params.txt', "w") as best_paramsb:
                    best_paramsb.write("The optimal parameters of the model are:\n" + str(best_params))
            else:
                best_model = KNeighborsClassifier(**shenmebestmodel)
                best_model.fit(X_train, y_train)
                best_params = "None"
            y_truetrain = y_train
            y_truetest = y_test
            y_predtrain = best_model.predict(X_train)
            y_predtest = best_model.predict(X_test)
            y_protrain = best_model.predict_proba(X_train)
            y_protest = best_model.predict_proba(X_test)
            st.markdown("#### "+mingshishenme+"Here are the results:")
            ACCAUEDENGDENGtrain(y_truetrain,y_protrain,y_predtrain,X_train,mingshishenme)
            ACCAUEDENGDENGTEST(y_truetest, y_protest, y_predtest, X_test, modelfusionall_words,mingshishenme)
            reporttrain = classification_report(y_truetrain, y_predtrain, output_dict=True)
            reporttest = classification_report(y_truetest, y_predtest, output_dict=True)
            dfroctrain = pd.DataFrame(reporttrain).transpose()
            dfroctest = pd.DataFrame(reporttest).transpose()
            hebingcccc = pd.concat([moxing_test, moxing_train])
            st.warning("The best models have no weight attributes!")
            with pd.ExcelWriter("./DataStatistics/KNN/KNN" + "_" + mingshishenme + "_Report.xlsx",engine='xlsxwriter') as writer:
                hebingcccc.to_excel(writer, index=False, sheet_name="KNN" + "_" + mingshishenme)
                moxing_train.to_excel(writer, index=False, sheet_name="KNN_train" + "_" + mingshishenme)
                moxing_test.to_excel(writer, index=False, sheet_name="KNN_test" + "_" + mingshishenme)
                dataTRAINPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="KNN_ROCtrain" + "_" + mingshishenme + "_report_1")
                dataTESTPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="KNN_ROCtest" + "_" + mingshishenme + "_report_1")
                dfroctrain.to_excel(writer, index=True, sheet_name="KNN_ROCtrain" + "_" + mingshishenme + "_report_2")
                dfroctest.to_excel(writer, index=True, sheet_name="KNN_ROCtest" + "_" + mingshishenme + "_report_2")
            return best_params
        st.markdown("---")
        XUNLIANYANZHENGDEYANZHENGMINGtrain = "train"
        shenmebestmodel1 = "none"
        shenmebestmodel=KNNhanshu(X_train_TRAIN, y_train_TRAIN,XUNLIANYANZHENGDEYANZHENGMINGtrain,shenmebestmodel1,train_size)
        st.markdown("---")
        XUNLIANYANZHENGDEYANZHENGMINGtest = "test"
        KNNhanshu(X_test_TEST, y_test_TEST,XUNLIANYANZHENGDEYANZHENGMINGtest,shenmebestmodel,test_size)
        st.markdown("---")

    if modelfusionall_words == "XGBoost":
        os.makedirs('./DataStatistics/XGBoost/', exist_ok=True)
        bestmodel = XGBClassifier(random_state= moxingderandomstate)
        def XGBhanshu(XX,yy,mingshishenme,shenmebestmodel,testortrain_size):
            X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size= test_size11nei, random_state=42)
            if mingshishenme != "test":
                if chaocanshumoxing == "Grid search":
                    best_model = GridSearchCV(bestmodel, param_grid=params, cv=allcv, n_jobs=-1,verbose=allverbose)
                elif  chaocanshumoxing == "Random search":
                    best_model = RandomizedSearchCV(estimator=bestmodel,param_distributions=params,n_iter=twoniters,cv=allcv,verbose=allverbose,random_state=tworandomstate,n_jobs=22)
                elif  chaocanshumoxing == "Bayesian optimization":
                    best_model = BayesSearchCV(bestmodel,params,n_iter=twoniters,cv=allcv,n_jobs=-1,random_state = tworandomstate,verbose=allverbose)
                best_model.fit(X_train, y_train)
                best_moxing_model = best_model.best_estimator_
                best_params = best_model.best_params_
                st.markdown(f"The optimal parameters of the model are:{best_params}")
                joblib.dump(best_model,
                            './DataStatistics/' + modelfusionall_words + '/' + modelfusionall_words + '.pkl')
                with open('./DataStatistics/' + modelfusionall_words + '/' +  modelfusionall_words +'_best_params.txt', "w") as best_paramsb:
                    best_paramsb.write("The optimal parameters of the model are:\n" + str(best_params))
            else:
                best_model = XGBClassifier(**shenmebestmodel,random_state=moxingderandomstate)
                best_model.fit(X_train, y_train)
                best_params = "None"
            y_truetrain = y_train
            y_truetest = y_test
            y_predtrain = best_model.predict(X_train)
            y_predtest = best_model.predict(X_test)
            y_protrain = best_model.predict_proba(X_train)
            y_protest = best_model.predict_proba(X_test)
            st.markdown("#### " + mingshishenme + "Here are the results:")
            ACCAUEDENGDENGtrain(y_truetrain,y_protrain,y_predtrain,X_train,mingshishenme)
            ACCAUEDENGDENGTEST(y_truetest, y_protest, y_predtest, X_test, modelfusionall_words,mingshishenme)
            reporttrain = classification_report(y_truetrain, y_predtrain, output_dict=True)
            reporttest = classification_report(y_truetest, y_predtest, output_dict=True)
            dfroctrain = pd.DataFrame(reporttrain).transpose()
            dfroctest = pd.DataFrame(reporttest).transpose()
            hebingcccc = pd.concat([moxing_test, moxing_train])
            if mingshishenme != "test":
                if hasattr(best_moxing_model, 'feature_importances_'):
                    np.savetxt("./DataStatistics/XGBoost/XGB_coefficient.csv",
                               best_moxing_model.feature_importances_, delimiter=',')
                    coef1 = pd.read_csv(r"./DataStatistics/XGBoost/XGB_coefficient.csv", header=None)
                    coef1_array = np.array(coef1.stack())
                    coef1_list = coef1_array.tolist()
                    df1 = pd.DataFrame(coef1_list)
                    feature2 = list(X_train.columns)
                    df2 = pd.DataFrame(feature2)
                    df2.to_csv("./DataStatistics/XGBoost/XGB_coefficient.csv", index=False)
                    hebingQUANZHONG = pd.concat([df2, df1], axis=1)
                    hebingQUANZHONG.columns = ['Features', 'Coefficients']
                    quanzhognhuizhi(hebingQUANZHONG, modelfusionall_words)
                    with pd.ExcelWriter("./DataStatistics/XGBoost/XGB" + "_" + mingshishenme + "_Report.xlsx",engine='xlsxwriter') as writer:
                        hebingcccc.to_excel(writer, index=False, sheet_name="XGB" + "_" + mingshishenme)
                        moxing_train.to_excel(writer, index=False, sheet_name="XGB_train" + "_" + mingshishenme)
                        moxing_test.to_excel(writer, index=False, sheet_name="XGB_test" + "_" + mingshishenme)
                        hebingQUANZHONG.to_excel(writer, index=True, sheet_name="XGB_coefficient" + "_" + mingshishenme)
                        dataTRAINPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="XGB_ROCtrain" + "_" + mingshishenme + "_report_1")
                        dataTESTPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="XGB_ROCtest" + "_" + mingshishenme + "_report_1")
                        dfroctrain.to_excel(writer, index=True, sheet_name="XGB_ROCtrain" + "_" + mingshishenme + "_report_2")
                        dfroctest.to_excel(writer, index=True, sheet_name="XGB_ROCtest" + "_" + mingshishenme + "_report_2")
                    try:
                        os.remove("./DataStatistics/XGBoost/XGB_coefficient.csv")
                    except:
                        print("no")
                else:
                    st.warning("The best models have no weight attributes!")
                    with pd.ExcelWriter("./DataStatistics/XGBoost/XGB" + "_" + mingshishenme + "_Report.xlsx",
                                        engine='xlsxwriter') as writer:
                        hebingcccc.to_excel(writer, index=False, sheet_name="XGB" + "_" + mingshishenme)
                        moxing_train.to_excel(writer, index=False, sheet_name="XGB_train" + "_" + mingshishenme)
                        moxing_test.to_excel(writer, index=False, sheet_name="XGB_test" + "_" + mingshishenme)
                        dataTRAINPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="XGB_ROCtrain" + "_" + mingshishenme + "_report_1")
                        dataTESTPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="XGB_ROCtest" + "_" + mingshishenme + "_report_1")
                        dfroctrain.to_excel(writer, index=True, sheet_name="XGB_ROCtrain" + "_" + mingshishenme + "_report_2")
                        dfroctest.to_excel(writer, index=True, sheet_name="XGB_ROCtest" + "_" + mingshishenme + "_report_2")
            else:
                with pd.ExcelWriter("./DataStatistics/XGBoost/XGB" + "_" + mingshishenme + "_Report.xlsx",engine='xlsxwriter') as writer:
                    hebingcccc.to_excel(writer, index=False, sheet_name="XGB" + "_" + mingshishenme)
                    moxing_train.to_excel(writer, index=False, sheet_name="XGB_train" + "_" + mingshishenme)
                    moxing_test.to_excel(writer, index=False, sheet_name="XGB_test" + "_" + mingshishenme)
                    dataTRAINPONGJIAZHIBIAO.to_excel(writer, index=True,sheet_name="XGB_ROCtrain" + "_" + mingshishenme + "_report_1")
                    dataTESTPONGJIAZHIBIAO.to_excel(writer, index=True,sheet_name="XGB_ROCtest" + "_" + mingshishenme + "_report_1")
                    dfroctrain.to_excel(writer, index=True,sheet_name="XGB_ROCtrain" + "_" + mingshishenme + "_report_2")
                    dfroctest.to_excel(writer, index=True,sheet_name="XGB_ROCtest" + "_" + mingshishenme + "_report_2")
            return best_params
        st.markdown("---")
        XUNLIANYANZHENGDEYANZHENGMINGtrain = "train"
        shenmebestmodel1 = "none"
        shenmebestmodel = XGBhanshu(X_train_TRAIN, y_train_TRAIN,XUNLIANYANZHENGDEYANZHENGMINGtrain,shenmebestmodel1,train_size)
        st.markdown("---")
        XUNLIANYANZHENGDEYANZHENGMINGtest = "test"
        XGBhanshu(X_test_TEST, y_test_TEST,XUNLIANYANZHENGDEYANZHENGMINGtest,shenmebestmodel,test_size)
        st.markdown("---")

    if modelfusionall_words == "Adaboost":
        os.makedirs('./DataStatistics/Adaboost/', exist_ok=True)
        bestmodel = AdaBoostClassifier(random_state= moxingderandomstate)
        def adahanshu(XX,yy,mingshishenme,shenmebestmodel,testortrain_size):
            X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size= test_size11nei, random_state=42)
            if mingshishenme != "test":
                if chaocanshumoxing == "Grid search":
                    best_model = GridSearchCV(bestmodel, param_grid=params, cv=allcv, n_jobs=-1,verbose=allverbose)
                elif  chaocanshumoxing == "Random search":
                    best_model = RandomizedSearchCV(estimator=bestmodel,param_distributions=params,n_iter=twoniters,cv=allcv,verbose=allverbose,random_state=tworandomstate,n_jobs=22)
                elif  chaocanshumoxing == "Bayesian optimization":
                    best_model = BayesSearchCV(bestmodel,params,n_iter=twoniters,cv=allcv,n_jobs=-1,random_state = tworandomstate,verbose=allverbose)
                best_model.fit(X_train, y_train)
                best_moxing_model = best_model.best_estimator_
                best_params = best_model.best_params_
                st.markdown(f"The optimal parameters of the model are:{best_params}")
                joblib.dump(best_model,
                            './DataStatistics/' + modelfusionall_words + '/' + modelfusionall_words + '.pkl')
                with open('./DataStatistics/' + modelfusionall_words + '/' +  modelfusionall_words +'_best_params.txt', "w") as best_paramsb:
                    best_paramsb.write("The optimal parameters of the model are:\n" + str(best_params))
            else:
                best_model = AdaBoostClassifier(**shenmebestmodel,random_state=moxingderandomstate)
                best_model.fit(X_train, y_train)
                best_params = "None"
            y_truetrain = y_train
            y_truetest = y_test
            y_predtrain = best_model.predict(X_train)
            y_predtest = best_model.predict(X_test)
            y_protrain = best_model.predict_proba(X_train)
            y_protest = best_model.predict_proba(X_test)
            st.markdown("#### "+mingshishenme+"Here are the results:")
            ACCAUEDENGDENGtrain(y_truetrain,y_protrain,y_predtrain,X_train,mingshishenme)
            ACCAUEDENGDENGTEST(y_truetest, y_protest, y_predtest, X_test, modelfusionall_words,mingshishenme)
            reporttrain = classification_report(y_truetrain, y_predtrain, output_dict=True)
            reporttest = classification_report(y_truetest, y_predtest, output_dict=True)
            dfroctrain = pd.DataFrame(reporttrain).transpose()
            dfroctest = pd.DataFrame(reporttest).transpose()
            hebingcccc = pd.concat([moxing_test, moxing_train])
            if mingshishenme != "test":
                if hasattr(best_moxing_model, 'feature_importances_'):
                    np.savetxt("./DataStatistics/Adaboost/ADA_coefficient.csv",
                               best_moxing_model.feature_importances_, delimiter=',')
                    coef1 = pd.read_csv(r"./DataStatistics/Adaboost/ADA_coefficient.csv", header=None)
                    coef1_array = np.array(coef1.stack())
                    coef1_list = coef1_array.tolist()
                    df1 = pd.DataFrame(coef1_list)
                    feature2 = list(X_train.columns)
                    df2 = pd.DataFrame(feature2)
                    df2.to_csv("./DataStatistics/Adaboost/ADA_coefficient.csv", index=False)
                    hebingQUANZHONG = pd.concat([df2, df1], axis=1)
                    hebingQUANZHONG.columns = ['Features', 'Coefficients']
                    quanzhognhuizhi(hebingQUANZHONG, modelfusionall_words)
                    with pd.ExcelWriter("./DataStatistics/Adaboost/ADA" + "_" + mingshishenme + "_Report.xlsx",engine='xlsxwriter') as writer:
                        hebingcccc.to_excel(writer, index=False, sheet_name="ADA" + "_" + mingshishenme)
                        moxing_train.to_excel(writer, index=False, sheet_name="ADA_train" + "_" + mingshishenme)
                        moxing_test.to_excel(writer, index=False, sheet_name="ADA_test" + "_" + mingshishenme)
                        hebingQUANZHONG.to_excel(writer, index=True, sheet_name="ADA_coefficient" + "_" + mingshishenme)
                        dataTRAINPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="ADA_ROCtrain" + "_" + mingshishenme + "_report_1")
                        dataTESTPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="ADA_ROCtest" + "_" + mingshishenme + "_report_1")
                        dfroctrain.to_excel(writer, index=True, sheet_name="ADA_ROCtrain" + "_" + mingshishenme + "_report_2")
                        dfroctest.to_excel(writer, index=True, sheet_name="ADA_ROCtest" + "_" + mingshishenme + "_report_2")
                    try:
                        os.remove("./DataStatistics/Adaboost/ADA_coefficient.csv")
                    except:
                        print("no")
                else:
                    st.warning("The best models have no weight attributes!")
                    with pd.ExcelWriter("./DataStatistics/Adaboost/ADA" + "_" + mingshishenme + "_Report.xlsx",
                                        engine='xlsxwriter') as writer:
                        hebingcccc.to_excel(writer, index=False, sheet_name="ADA" + "_" + mingshishenme)
                        moxing_train.to_excel(writer, index=False, sheet_name="ADA_train" + "_" + mingshishenme)
                        moxing_test.to_excel(writer, index=False, sheet_name="ADA_test" + "_" + mingshishenme)
                        dataTRAINPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="ADA_ROCtrain" + "_" + mingshishenme + "_report_1")
                        dataTESTPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="ADA_ROCtest" + "_" + mingshishenme + "_report_1")
                        dfroctrain.to_excel(writer, index=True, sheet_name="ADA_ROCtrain" + "_" + mingshishenme + "_report_2")
                        dfroctest.to_excel(writer, index=True, sheet_name="ADA_ROCtest" + "_" + mingshishenme + "_report_2")
            else:
                with pd.ExcelWriter("./DataStatistics/Adaboost/ADA" + "_" + mingshishenme + "_Report.xlsx",
                                    engine='xlsxwriter') as writer:
                    hebingcccc.to_excel(writer, index=False, sheet_name="ADA" + "_" + mingshishenme)
                    moxing_train.to_excel(writer, index=False, sheet_name="ADA_train" + "_" + mingshishenme)
                    moxing_test.to_excel(writer, index=False, sheet_name="ADA_test" + "_" + mingshishenme)
                    dataTRAINPONGJIAZHIBIAO.to_excel(writer, index=True,
                                                     sheet_name="ADA_ROCtrain" + "_" + mingshishenme + "_report_1")
                    dataTESTPONGJIAZHIBIAO.to_excel(writer, index=True,
                                                    sheet_name="ADA_ROCtest" + "_" + mingshishenme + "_report_1")
                    dfroctrain.to_excel(writer, index=True,
                                        sheet_name="ADA_ROCtrain" + "_" + mingshishenme + "_report_2")
                    dfroctest.to_excel(writer, index=True,
                                       sheet_name="ADA_ROCtest" + "_" + mingshishenme + "_report_2")
            return best_params
        st.markdown("---")
        XUNLIANYANZHENGDEYANZHENGMINGtrain = "train"
        shenmebestmodel1 = "none"
        shenmebestmodel = adahanshu(X_train_TRAIN, y_train_TRAIN,XUNLIANYANZHENGDEYANZHENGMINGtrain,shenmebestmodel1,train_size)
        st.markdown("---")
        XUNLIANYANZHENGDEYANZHENGMINGtest = "test"
        adahanshu(X_test_TEST, y_test_TEST,XUNLIANYANZHENGDEYANZHENGMINGtest,shenmebestmodel,test_size)
        st.markdown("---")

    if modelfusionall_words == "GBDT":
        os.makedirs('./DataStatistics/GBDT/', exist_ok=True)
        bestmodel = GradientBoostingClassifier(random_state= moxingderandomstate)
        def GBDThanshu(XX,yy,mingshishenme,shenmebestmodel,testortrain_size):
            X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size= test_size11nei, random_state=42)
            if mingshishenme != "test":
                if chaocanshumoxing == "Grid search":
                    best_model = GridSearchCV(bestmodel, param_grid=params, cv=allcv, n_jobs=-1,verbose=allverbose)
                elif  chaocanshumoxing == "Random search":
                    best_model = RandomizedSearchCV(estimator=bestmodel,param_distributions=params,n_iter=twoniters,cv=allcv,verbose=allverbose,random_state=tworandomstate,n_jobs=22)
                elif  chaocanshumoxing == "Bayesian optimization":
                    best_model = BayesSearchCV(bestmodel,params,n_iter=twoniters,cv=allcv,n_jobs=-1,random_state = tworandomstate,verbose=allverbose)
                best_model.fit(X_train, y_train)
                best_moxing_model = best_model.best_estimator_
                best_params = best_model.best_params_
                st.markdown(f"The optimal parameters of the model are:{best_params}")
                joblib.dump(best_model,
                            './DataStatistics/' + modelfusionall_words + '/' + modelfusionall_words + '.pkl')
                with open('./DataStatistics/' + modelfusionall_words + '/' +  modelfusionall_words +'_best_params.txt', "w") as best_paramsb:
                    best_paramsb.write("The optimal parameters of the model are:\n" + str(best_params))
            else:
                best_model = GradientBoostingClassifier(**shenmebestmodel,random_state=moxingderandomstate)
                best_model.fit(X_train, y_train)
                best_params = "None"
            y_truetrain = y_train
            y_truetest = y_test
            y_predtrain = best_model.predict(X_train)
            y_predtest = best_model.predict(X_test)
            y_protrain = best_model.predict_proba(X_train)
            y_protest = best_model.predict_proba(X_test)
            st.markdown("#### " + mingshishenme + "Here are the results:")
            ACCAUEDENGDENGtrain(y_truetrain,y_protrain,y_predtrain,X_train,mingshishenme)
            ACCAUEDENGDENGTEST(y_truetest, y_protest, y_predtest, X_test, modelfusionall_words,mingshishenme)
            reporttrain = classification_report(y_truetrain, y_predtrain, output_dict=True)
            reporttest = classification_report(y_truetest, y_predtest, output_dict=True)
            dfroctrain = pd.DataFrame(reporttrain).transpose()
            dfroctest = pd.DataFrame(reporttest).transpose()
            hebingcccc = pd.concat([moxing_test, moxing_train])
            if mingshishenme != "test":
                if hasattr(best_moxing_model, 'feature_importances_'):
                    np.savetxt("./DataStatistics/GBDT/GBDT_coefficient.csv",
                               best_moxing_model.feature_importances_, delimiter=',')
                    coef1 = pd.read_csv(r"./DataStatistics/GBDT/GBDT_coefficient.csv", header=None)
                    coef1_array = np.array(coef1.stack())
                    coef1_list = coef1_array.tolist()
                    df1 = pd.DataFrame(coef1_list)
                    feature2 = list(X_train.columns)
                    df2 = pd.DataFrame(feature2)
                    df2.to_csv("./DataStatistics/GBDT/GBDT_coefficient.csv", index=False)
                    hebingQUANZHONG = pd.concat([df2, df1], axis=1)
                    hebingQUANZHONG.columns = ['Features', 'Coefficients']
                    quanzhognhuizhi(hebingQUANZHONG, modelfusionall_words)
                    with pd.ExcelWriter("./DataStatistics/GBDT/GBDT" + "_" + mingshishenme + "_Report.xlsx",engine='xlsxwriter') as writer:
                        hebingcccc.to_excel(writer, index=False, sheet_name="GBDT" + "_" + mingshishenme)
                        moxing_train.to_excel(writer, index=False, sheet_name="GBDT_train" + "_" + mingshishenme)
                        moxing_test.to_excel(writer, index=False, sheet_name="GBDT_test" + "_" + mingshishenme)
                        hebingQUANZHONG.to_excel(writer, index=True, sheet_name="GBDT_coefficient" + "_" + mingshishenme)
                        dataTRAINPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="GBDT_ROCtrain" + "_" + mingshishenme + "_report_1")
                        dataTESTPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="GBDT_ROCtest" + "_" + mingshishenme + "_report_1")
                        dfroctrain.to_excel(writer, index=True, sheet_name="GBDT_ROCtrain" + "_" + mingshishenme + "_report_2")
                        dfroctest.to_excel(writer, index=True, sheet_name="GBDT_ROCtest" + "_" + mingshishenme + "_report_2")
                    try:
                        os.remove("./DataStatistics/GBDT/GBDT_coefficient.csv")
                    except:
                        print("no")
                else:
                    st.warning("The best models have no weight attributes!")
                    with pd.ExcelWriter("./DataStatistics/GBDT/GBDT" + "_" + mingshishenme + "_Report.xlsx",
                                        engine='xlsxwriter') as writer:
                        hebingcccc.to_excel(writer, index=False, sheet_name="GBDT" + "_" + mingshishenme)
                        moxing_train.to_excel(writer, index=False, sheet_name="GBDT_train" + "_" + mingshishenme)
                        moxing_test.to_excel(writer, index=False, sheet_name="GBDT_test" + "_" + mingshishenme)
                        dataTRAINPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="GBDT_ROCtrain" + "_" + mingshishenme + "_report_1")
                        dataTESTPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="GBDT_ROCtest" + "_" + mingshishenme + "_report_1")
                        dfroctrain.to_excel(writer, index=True, sheet_name="GBDT_ROCtrain" + "_" + mingshishenme + "_report_2")
                        dfroctest.to_excel(writer, index=True, sheet_name="GBDT_ROCtest" + "_" + mingshishenme + "_report_2")
            else:
                with pd.ExcelWriter("./DataStatistics/GBDT/GBDT" + "_" + mingshishenme + "_Report.xlsx",
                                    engine='xlsxwriter') as writer:
                    hebingcccc.to_excel(writer, index=False, sheet_name="GBDT" + "_" + mingshishenme)
                    moxing_train.to_excel(writer, index=False, sheet_name="GBDT_train" + "_" + mingshishenme)
                    moxing_test.to_excel(writer, index=False, sheet_name="GBDT_test" + "_" + mingshishenme)
                    dataTRAINPONGJIAZHIBIAO.to_excel(writer, index=True,
                                                     sheet_name="GBDT_ROCtrain" + "_" + mingshishenme + "_report_1")
                    dataTESTPONGJIAZHIBIAO.to_excel(writer, index=True,
                                                    sheet_name="GBDT_ROCtest" + "_" + mingshishenme + "_report_1")
                    dfroctrain.to_excel(writer, index=True,
                                        sheet_name="GBDT_ROCtrain" + "_" + mingshishenme + "_report_2")
                    dfroctest.to_excel(writer, index=True,
                                       sheet_name="GBDT_ROCtest" + "_" + mingshishenme + "_report_2")
            return best_params
        st.markdown("---")
        XUNLIANYANZHENGDEYANZHENGMINGtrain = "train"
        shenmebestmodel1 = "none"
        shenmebestmodel = GBDThanshu(X_train_TRAIN, y_train_TRAIN,XUNLIANYANZHENGDEYANZHENGMINGtrain,shenmebestmodel1,train_size)
        st.markdown("---")
        XUNLIANYANZHENGDEYANZHENGMINGtest = "test"
        GBDThanshu(X_test_TEST, y_test_TEST,XUNLIANYANZHENGDEYANZHENGMINGtest,shenmebestmodel,test_size)
        st.markdown("---")

    if modelfusionall_words == "CatBoost":
        os.makedirs('./DataStatistics/CatBoost/', exist_ok=True)
        bestmodel = CatBoostClassifier(random_state= moxingderandomstate)
        def cabhanshu(XX,yy,mingshishenme,shenmebestmodel,testortrain_size):
            X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size= test_size11nei, random_state=42)
            if mingshishenme != "test":
                if chaocanshumoxing == "Grid search":
                    best_model = GridSearchCV(bestmodel, param_grid=params, cv=allcv, n_jobs=-1,verbose=allverbose)
                elif  chaocanshumoxing == "Random search":
                    best_model = RandomizedSearchCV(estimator=bestmodel,param_distributions=params,n_iter=twoniters,cv=allcv,verbose=allverbose,random_state=tworandomstate,n_jobs=22)
                elif  chaocanshumoxing == "Bayesian optimization":
                    best_model = BayesSearchCV(bestmodel,params,n_iter=twoniters,cv=allcv,n_jobs=-1,random_state = tworandomstate,verbose=allverbose)
                best_model.fit(X_train, y_train)
                best_moxing_model = best_model.best_estimator_
                best_params = best_model.best_params_
                st.markdown(f"The optimal parameters of the model are:{best_params}")
                joblib.dump(best_model,
                            './DataStatistics/' + modelfusionall_words + '/' + modelfusionall_words + '.pkl')
                with open('./DataStatistics/' + modelfusionall_words + '/' +  modelfusionall_words +'_best_params.txt', "w") as best_paramsb:
                    best_paramsb.write("The optimal parameters of the model are:\n" + str(best_params))
            else:
                best_model = CatBoostClassifier(**shenmebestmodel,random_state=moxingderandomstate)
                best_model.fit(X_train, y_train)
                best_params = "None"
            y_truetrain = y_train
            y_truetest = y_test
            y_predtrain = best_model.predict(X_train)
            y_predtest = best_model.predict(X_test)
            y_protrain = best_model.predict_proba(X_train)
            y_protest = best_model.predict_proba(X_test)
            st.markdown("#### "+mingshishenme+"Here are the results:")
            ACCAUEDENGDENGtrain(y_truetrain,y_protrain,y_predtrain,X_train,mingshishenme)
            ACCAUEDENGDENGTEST(y_truetest, y_protest, y_predtest, X_test, modelfusionall_words,mingshishenme)
            reporttrain = classification_report(y_truetrain, y_predtrain, output_dict=True)
            reporttest = classification_report(y_truetest, y_predtest, output_dict=True)
            dfroctrain = pd.DataFrame(reporttrain).transpose()
            dfroctest = pd.DataFrame(reporttest).transpose()
            hebingcccc = pd.concat([moxing_test, moxing_train])
            if mingshishenme != "test":
                if hasattr(best_moxing_model, 'feature_importances_'):
                    np.savetxt("./DataStatistics/CatBoost/CAB_coefficient.csv",
                               best_moxing_model.feature_importances_, delimiter=',')
                    coef1 = pd.read_csv(r"./DataStatistics/CatBoost/CAB_coefficient.csv", header=None)
                    coef1_array = np.array(coef1.stack())
                    coef1_list = coef1_array.tolist()
                    df1 = pd.DataFrame(coef1_list)
                    feature2 = list(X_train.columns)
                    df2 = pd.DataFrame(feature2)
                    df2.to_csv("./DataStatistics/CatBoost/CAB_coefficient.csv", index=False)
                    hebingQUANZHONG = pd.concat([df2, df1], axis=1)
                    hebingQUANZHONG.columns = ['Features', 'Coefficients']
                    quanzhognhuizhi(hebingQUANZHONG, modelfusionall_words)
                    with pd.ExcelWriter("./DataStatistics/CatBoost/CAB" + "_" + mingshishenme + "_Report.xlsx",engine='xlsxwriter') as writer:
                        hebingcccc.to_excel(writer, index=False, sheet_name="CAB" + "_" + mingshishenme)
                        moxing_train.to_excel(writer, index=False, sheet_name="CAB_train" + "_" + mingshishenme)
                        moxing_test.to_excel(writer, index=False, sheet_name="CAB_test" + "_" + mingshishenme)
                        hebingQUANZHONG.to_excel(writer, index=True, sheet_name="CAB_coefficient" + "_" + mingshishenme)
                        dataTRAINPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="CAB_ROCtrain" + "_" + mingshishenme + "_report_1")
                        dataTESTPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="CAB_ROCtest" + "_" + mingshishenme + "_report_1")
                        dfroctrain.to_excel(writer, index=True, sheet_name="CAB_ROCtrain" + "_" + mingshishenme + "_report_2")
                        dfroctest.to_excel(writer, index=True, sheet_name="CAB_ROCtest" + "_" + mingshishenme + "_report_2")
                    try:
                        os.remove("./DataStatistics/CatBoost/CAB_coefficient.csv")
                    except:
                        print("no")
                else:
                    st.warning("The best models have no weight attributes!")
                    with pd.ExcelWriter("./DataStatistics/CatBoost/CAB" + "_" + mingshishenme + "_Report.xlsx",
                                        engine='xlsxwriter') as writer:
                        hebingcccc.to_excel(writer, index=False, sheet_name="CAB" + "_" + mingshishenme)
                        moxing_train.to_excel(writer, index=False, sheet_name="CAB_train" + "_" + mingshishenme)
                        moxing_test.to_excel(writer, index=False, sheet_name="CAB_test" + "_" + mingshishenme)
                        dataTRAINPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="CAB_ROCtrain" + "_" + mingshishenme + "_report_1")
                        dataTESTPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="CAB_ROCtest" + "_" + mingshishenme + "_report_1")
                        dfroctrain.to_excel(writer, index=True, sheet_name="CAB_ROCtrain" + "_" + mingshishenme + "_report_2")
                        dfroctest.to_excel(writer, index=True, sheet_name="CAB_ROCtest" + "_" + mingshishenme + "_report_2")
            else:
                with pd.ExcelWriter("./DataStatistics/CatBoost/CAB" + "_" + mingshishenme + "_Report.xlsx",
                                    engine='xlsxwriter') as writer:
                    hebingcccc.to_excel(writer, index=False, sheet_name="CAB" + "_" + mingshishenme)
                    moxing_train.to_excel(writer, index=False, sheet_name="CAB_train" + "_" + mingshishenme)
                    moxing_test.to_excel(writer, index=False, sheet_name="CAB_test" + "_" + mingshishenme)
                    dataTRAINPONGJIAZHIBIAO.to_excel(writer, index=True,
                                                     sheet_name="CAB_ROCtrain" + "_" + mingshishenme + "_report_1")
                    dataTESTPONGJIAZHIBIAO.to_excel(writer, index=True,
                                                    sheet_name="CAB_ROCtest" + "_" + mingshishenme + "_report_1")
                    dfroctrain.to_excel(writer, index=True,
                                        sheet_name="CAB_ROCtrain" + "_" + mingshishenme + "_report_2")
                    dfroctest.to_excel(writer, index=True,
                                       sheet_name="CAB_ROCtest" + "_" + mingshishenme + "_report_2")
            return best_params
        st.markdown("---")
        XUNLIANYANZHENGDEYANZHENGMINGtrain = "train"
        shenmebestmodel1 = "none"
        shenmebestmodel = cabhanshu(X_train_TRAIN, y_train_TRAIN,XUNLIANYANZHENGDEYANZHENGMINGtrain,shenmebestmodel1,train_size)
        st.markdown("---")
        XUNLIANYANZHENGDEYANZHENGMINGtest = "test"
        cabhanshu(X_test_TEST, y_test_TEST,XUNLIANYANZHENGDEYANZHENGMINGtest,shenmebestmodel,test_size)
        st.markdown("---")

    if modelfusionall_words == "LightGBM":
        os.makedirs('./DataStatistics/LightGBM/', exist_ok=True)
        bestmodel = LGBMClassifier(random_state= moxingderandomstate)
        def LGBhanshu(XX,yy,mingshishenme,shenmebestmodel,testortrain_size):
            X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size= test_size11nei, random_state=42)
            if mingshishenme != "test":
                if chaocanshumoxing == "Grid search":
                    best_model = GridSearchCV(bestmodel, param_grid=params, cv=allcv, n_jobs=-1,verbose=allverbose)
                elif  chaocanshumoxing == "Random search":
                    best_model = RandomizedSearchCV(estimator=bestmodel,param_distributions=params,n_iter=twoniters,cv=allcv,verbose=allverbose,random_state=tworandomstate,n_jobs=22)
                elif  chaocanshumoxing == "Bayesian optimization":
                    best_model = BayesSearchCV(bestmodel,params,n_iter=twoniters,cv=allcv,n_jobs=-1,random_state = tworandomstate,verbose=allverbose)
                best_model.fit(X_train, y_train)
                best_moxing_model = best_model.best_estimator_
                best_params = best_model.best_params_
                st.markdown(f"The optimal parameters of the model are:{best_params}")
                joblib.dump(best_model,
                            './DataStatistics/' + modelfusionall_words + '/' + modelfusionall_words + '.pkl')
                with open('./DataStatistics/' + modelfusionall_words + '/' +  modelfusionall_words +'_best_params.txt', "w") as best_paramsb:
                    best_paramsb.write("The optimal parameters of the model are:\n" + str(best_params))
            else:
                best_model = LGBMClassifier(**shenmebestmodel,random_state=moxingderandomstate)
                best_model.fit(X_train, y_train)
                best_params = "None"
            y_truetrain = y_train
            y_truetest = y_test
            y_predtrain = best_model.predict(X_train)
            y_predtest = best_model.predict(X_test)
            y_protrain = best_model.predict_proba(X_train)
            y_protest = best_model.predict_proba(X_test)
            st.markdown("#### " + mingshishenme + "Here are the results:")
            ACCAUEDENGDENGtrain(y_truetrain,y_protrain,y_predtrain,X_train,mingshishenme)
            ACCAUEDENGDENGTEST(y_truetest, y_protest, y_predtest, X_test, modelfusionall_words,mingshishenme)
            reporttrain = classification_report(y_truetrain, y_predtrain, output_dict=True)
            reporttest = classification_report(y_truetest, y_predtest, output_dict=True)
            dfroctrain = pd.DataFrame(reporttrain).transpose()
            dfroctest = pd.DataFrame(reporttest).transpose()
            hebingcccc = pd.concat([moxing_test, moxing_train])
            if mingshishenme != "test":
                if hasattr(best_moxing_model, 'feature_importances_'):
                    np.savetxt("./DataStatistics/LightGBM/LGB_coefficient.csv",
                               best_moxing_model.feature_importances_, delimiter=',')
                    coef1 = pd.read_csv(r"./DataStatistics/LightGBM/LGB_coefficient.csv", header=None)
                    coef1_array = np.array(coef1.stack())
                    coef1_list = coef1_array.tolist()
                    df1 = pd.DataFrame(coef1_list)
                    feature2 = list(X_train.columns)
                    df2 = pd.DataFrame(feature2)
                    df2.to_csv("./DataStatistics/LightGBM/LGB_coefficient.csv", index=False)
                    hebingQUANZHONG = pd.concat([df2, df1], axis=1)
                    hebingQUANZHONG.columns = ['Features', 'Coefficients']
                    quanzhognhuizhi(hebingQUANZHONG, modelfusionall_words)
                    with pd.ExcelWriter("./DataStatistics/LightGBM/LGB" + "_" + mingshishenme + "_Report.xlsx",
                                        engine='xlsxwriter') as writer:
                        hebingcccc.to_excel(writer, index=False, sheet_name="LGB" + "_" + mingshishenme)
                        moxing_train.to_excel(writer, index=False, sheet_name="LGB_train" + "_" + mingshishenme)
                        moxing_test.to_excel(writer, index=False, sheet_name="LGB_test" + "_" + mingshishenme)
                        hebingQUANZHONG.to_excel(writer, index=True, sheet_name="LGB_coefficient" + "_" + mingshishenme)
                        dataTRAINPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="LGB_ROCtrain" + "_" + mingshishenme + "_report_1")
                        dataTESTPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="LGB_ROCtest" + "_" + mingshishenme + "_report_1")
                        dfroctrain.to_excel(writer, index=True, sheet_name="LGB_ROCtrain" + "_" + mingshishenme + "_report_2")
                        dfroctest.to_excel(writer, index=True, sheet_name="LGB_ROCtest" + "_" + mingshishenme + "_report_2")
                    try:
                        os.remove("./DataStatistics/LightGBM/LGB_coefficient.csv")
                    except:
                        print("no")
                else:
                    st.warning("The best models have no weight attributes!")
                    with pd.ExcelWriter("./DataStatistics/LightGBM/LGB" + "_" + mingshishenme + "_Report.xlsx",
                                        engine='xlsxwriter') as writer:
                        hebingcccc.to_excel(writer, index=False, sheet_name="LGB")
                        moxing_train.to_excel(writer, index=False, sheet_name="LGB_train")
                        moxing_test.to_excel(writer, index=False, sheet_name="LGB_test")
                        dataTRAINPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="LGB_ROCtrain" + "_" + mingshishenme + "report_1")
                        dataTESTPONGJIAZHIBIAO.to_excel(writer, index=True, sheet_name="LGB_ROCtest" + "_" + mingshishenme + "_report_1")
                        dfroctrain.to_excel(writer, index=True, sheet_name="LGB_ROCtrain" + "_" + mingshishenme + "_report_2")
                        dfroctest.to_excel(writer, index=True, sheet_name="LGB_ROCtest" + "_" + mingshishenme + "_report_2")
            else:
                with pd.ExcelWriter("./DataStatistics/LightGBM/LGB" + "_" + mingshishenme + "_Report.xlsx",
                                    engine='xlsxwriter') as writer:
                    hebingcccc.to_excel(writer, index=False, sheet_name="LGB")
                    moxing_train.to_excel(writer, index=False, sheet_name="LGB_train")
                    moxing_test.to_excel(writer, index=False, sheet_name="LGB_test")
                    dataTRAINPONGJIAZHIBIAO.to_excel(writer, index=True,
                                                     sheet_name="LGB_ROCtrain" + "_" + mingshishenme + "report_1")
                    dataTESTPONGJIAZHIBIAO.to_excel(writer, index=True,
                                                    sheet_name="LGB_ROCtest" + "_" + mingshishenme + "_report_1")
                    dfroctrain.to_excel(writer, index=True,
                                        sheet_name="LGB_ROCtrain" + "_" + mingshishenme + "_report_2")
                    dfroctest.to_excel(writer, index=True,
                                       sheet_name="LGB_ROCtest" + "_" + mingshishenme + "_report_2")
            return best_params
        st.markdown("---")
        XUNLIANYANZHENGDEYANZHENGMINGtrain = "train"
        shenmebestmodel1 = "none"
        shenmebestmodel = LGBhanshu(X_train_TRAIN, y_train_TRAIN,XUNLIANYANZHENGDEYANZHENGMINGtrain,shenmebestmodel1,train_size)
        st.markdown("---")
        XUNLIANYANZHENGDEYANZHENGMINGtest = "test"
        LGBhanshu(X_test_TEST, y_test_TEST,XUNLIANYANZHENGDEYANZHENGMINGtest,shenmebestmodel,test_size)
        st.markdown("---")

    if modelfusionall_words == "Bayes":
        os.makedirs('./DataStatistics/Bayes/', exist_ok=True)
        def Bayeshanshu(XX,yy,mingshishenme,shenmebestmodel,testortrain_size):
            X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size= test_size11nei, random_state=42)
            bestmodel = GaussianNB()
            if mingshishenme != "test":
                if chaocanshumoxing == "Grid search":
                    best_model = GridSearchCV(bestmodel, param_grid=params, cv=allcv, n_jobs=-1,verbose=allverbose)
                elif  chaocanshumoxing == "Random search":
                    best_model = RandomizedSearchCV(estimator=bestmodel,param_distributions=params,n_iter=twoniters,cv=allcv,verbose=allverbose,random_state=tworandomstate,n_jobs=22)
                elif  chaocanshumoxing == "Bayesian optimization":
                    best_model = BayesSearchCV(bestmodel,params,n_iter=twoniters,cv=allcv,n_jobs=-1,random_state = tworandomstate,verbose=allverbose)
                best_model.fit(X_train, y_train)
                best_moxing_model = best_model.best_estimator_
                best_params = best_model.best_params_
                st.markdown(f"The optimal parameters of the model are:{best_params}")
                joblib.dump(best_model,
                            './DataStatistics/' + modelfusionall_words + '/' + modelfusionall_words + '.pkl')
                with open('./DataStatistics/' + modelfusionall_words + '/' +  modelfusionall_words +'_best_params.txt', "w") as best_paramsb:
                    best_paramsb.write("The optimal parameters of the model are:\n" + str(best_params))
            else:
                best_model = GaussianNB(**shenmebestmodel)
                best_model.fit(X_train, y_train)
                best_params = "None"
            y_truetrain = y_train
            y_truetest = y_test
            y_predtrain = best_model.predict(X_train)
            y_predtest = best_model.predict(X_test)
            y_protrain = best_model.predict_proba(X_train)
            y_protest = best_model.predict_proba(X_test)
            st.markdown("#### "+mingshishenme+"Here are the results:")
            ACCAUEDENGDENGtrain(y_truetrain,y_protrain,y_predtrain,X_train,mingshishenme)
            ACCAUEDENGDENGTEST(y_truetest, y_protest, y_predtest, X_test, modelfusionall_words,mingshishenme)
            reporttrain = classification_report(y_truetrain, y_predtrain, output_dict=True)
            reporttest = classification_report(y_truetest, y_predtest, output_dict=True)
            dfroctrain = pd.DataFrame(reporttrain).transpose()
            dfroctest = pd.DataFrame(reporttest).transpose()
            hebingcccc = pd.concat([moxing_test,moxing_train])
            if mingshishenme != "test":
                if hasattr(best_moxing_model, 'coef_'):
                    np.savetxt("./DataStatistics/Bayes/Bayes_coefficient.csv", best_moxing_model.coef_, delimiter=',')
                    coef1 = pd.read_csv(r"./DataStatistics/Bayes/Bayes_coefficient.csv", header=None)
                    coef1_array = np.array(coef1.stack())
                    coef1_list = coef1_array.tolist()
                    df1 = pd.DataFrame(coef1_list)
                    feature2 = list(X_train.columns)
                    df2 = pd.DataFrame(feature2)
                    df2.to_csv("./DataStatistics/Bayes/Bayes_coefficient.csv", index=False)
                    hebingQUANZHONG = pd.concat([df2, df1], axis=1)
                    hebingQUANZHONG.columns = ['Features', 'Coefficients']
                    quanzhognhuizhi(hebingQUANZHONG, modelfusionall_words)
                    with pd.ExcelWriter("./DataStatistics/Bayes/Bayes" + "_" + mingshishenme + "_Report.xlsx",
                                        engine='xlsxwriter') as writer:
                        hebingcccc.to_excel(writer, index=False, sheet_name="Bayes" + "_" + mingshishenme)
                        moxing_train.to_excel(writer, index=False, sheet_name="Bayes_train" + "_" + mingshishenme)
                        moxing_test.to_excel(writer, index=False, sheet_name="Bayes_test" + "_" + mingshishenme)
                        hebingQUANZHONG.to_excel(writer, index=True,
                                                 sheet_name="Bayes_coefficient" + "_" + mingshishenme)
                        dataTRAINPONGJIAZHIBIAO.to_excel(writer, index=True,
                                                         sheet_name="Bayes_ROCtrain" + "_" + mingshishenme + "_report_1")
                        dataTESTPONGJIAZHIBIAO.to_excel(writer, index=True,
                                                        sheet_name="Bayes_ROCtest" + "_" + mingshishenme + "_report_1")
                        dfroctrain.to_excel(writer, index=True,
                                            sheet_name="Bayes_ROCtrain" + "_" + mingshishenme + "_report_2")
                        dfroctest.to_excel(writer, index=True,
                                           sheet_name="Bayes_ROCtest" + "_" + mingshishenme + "_report_2")
                    try:
                        os.remove("./DataStatistics/Bayes/Bayes_coefficient.csv")
                    except:
                        print("no")
                else:
                    st.warning("The best models have no weight attributes!")
                    with pd.ExcelWriter("./DataStatistics/Bayes/Bayes" + "_" + mingshishenme + "_Report.xlsx",
                                        engine='xlsxwriter') as writer:
                        hebingcccc.to_excel(writer, index=False, sheet_name="Bayes" + "_" + mingshishenme)
                        moxing_train.to_excel(writer, index=False, sheet_name="Bayes_train" + "_" + mingshishenme)
                        moxing_test.to_excel(writer, index=False, sheet_name="Bayes_test" + "_" + mingshishenme)
                        dataTRAINPONGJIAZHIBIAO.to_excel(writer, index=True,
                                                         sheet_name="Bayes_ROCtrain" + "_" + mingshishenme + "_report_1")
                        dataTESTPONGJIAZHIBIAO.to_excel(writer, index=True,
                                                        sheet_name="Bayes_ROCtest" + "_" + mingshishenme + "_report_1")
                        dfroctrain.to_excel(writer, index=True,
                                            sheet_name="Bayes_ROCtrain" + "_" + mingshishenme + "_report_2")
                        dfroctest.to_excel(writer, index=True,
                                           sheet_name="Bayes_ROCtest" + "_" + mingshishenme + "_report_2")
            else:
                with pd.ExcelWriter("./DataStatistics/Bayes/Bayes" + "_" + mingshishenme + "_Report.xlsx",
                                    engine='xlsxwriter') as writer:
                    hebingcccc.to_excel(writer, index=False, sheet_name="Bayes" + "_" + mingshishenme)
                    moxing_train.to_excel(writer, index=False, sheet_name="Bayes_train" + "_" + mingshishenme)
                    moxing_test.to_excel(writer, index=False, sheet_name="Bayes_test" + "_" + mingshishenme)
                    dataTRAINPONGJIAZHIBIAO.to_excel(writer, index=True,
                                                     sheet_name="Bayes_ROCtrain" + "_" + mingshishenme + "_report_1")
                    dataTESTPONGJIAZHIBIAO.to_excel(writer, index=True,
                                                    sheet_name="Bayes_ROCtest" + "_" + mingshishenme + "_report_1")
                    dfroctrain.to_excel(writer, index=True,
                                        sheet_name="Bayes_ROCtrain" + "_" + mingshishenme + "_report_2")
                    dfroctest.to_excel(writer, index=True,
                                       sheet_name="Bayes_ROCtest" + "_" + mingshishenme + "_report_2")
            return best_params
        st.markdown("---")
        XUNLIANYANZHENGDEYANZHENGMINGtrain = "train"
        shenmebestmodel1 = "none"
        shenmebestmodel = Bayeshanshu(X_train_TRAIN, y_train_TRAIN, XUNLIANYANZHENGDEYANZHENGMINGtrain,shenmebestmodel1,train_size)
        st.markdown("---")
        XUNLIANYANZHENGDEYANZHENGMINGtest = "test"
        Bayeshanshu(X_test_TEST, y_test_TEST,XUNLIANYANZHENGDEYANZHENGMINGtest,shenmebestmodel,test_size)
        st.markdown("---")

if st.button("Build predictive models",key="jisdadsafng"):
    os.makedirs('./DataStatistics/' + modelfusionall_words + '/' ,exist_ok=True)
    def rfshuffle():
        if RFhunxuwenjian is not None:
            rfhunxu1 = pd.read_excel(RFhunxuwenjian)
            rfhunxu = shuffle(rfhunxu1)
            rfhunxu.to_excel('./DataStatistics/' + modelfusionall_words + '/' + 'shffle.xlsx', index=False)
    rfshuffle()
    RF = './DataStatistics/' + modelfusionall_words + '/' + 'shffle.xlsx'
    datarad = pd.read_excel(RF)
    if 'Rad_score' in datarad.columns:
        data = datarad.drop(labels='Rad_score', axis=1)
    else:
        data = datarad
    test_size = int(len(data) * bilishu)
    train_size = len(data) - test_size
    datatest = data.iloc[:test_size]
    datatrain = data.iloc[test_size:]
    y_train_TRAIN = datatrain['label']
    X_train1_TRAIN = datatrain.drop(labels='label', axis=1)
    if 'Rad_score' in X_train1_TRAIN.columns:
        X_train_TRAIN = X_train1_TRAIN.drop(labels='Rad_score', axis=1)
    else:
        X_train_TRAIN = X_train1_TRAIN
    y_test_TEST = datatest['label']
    X_test1_TEST = datatest.drop(labels='label', axis=1)
    if 'Rad_score' in X_test1_TEST.columns:
        X_test_TEST = X_test1_TEST.drop(labels='Rad_score', axis=1)
    else:
        X_test_TEST = X_test1_TEST
    moxinggoujianerfenlei(test_size,train_size)
    st.success("Once the predictive model is built, save the data in time!")
    def zipdir(path, ziph):
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file))
    folder_pathmoxing = "./DataStatistics/" + modelfusionall_words + "/"
    zip_filename = modelfusionall_words + ".zip"
    zipf = zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED)
    zipdir(folder_pathmoxing, zipf)
    zipf.close()
    with open(zip_filename, 'rb') as f:
        bytes = f.read()
        st.download_button(
            label="Download the model file",
            data=bytes,
            file_name=zip_filename,
            mime="application/zip")
