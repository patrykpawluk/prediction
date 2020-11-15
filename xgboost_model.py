import pandas as pd 
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

#wczytanie danych do datasetu pandas
dataset = pd.read_csv('./datasets/credit_card_frauds.csv',delimiter=",")
y = dataset['Class']
X = dataset.iloc[:,:-1]

#train_test_split gotowa funkcja do podzielenia bazy do medelu na część trenującą i uczącą
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
X_train_D = xgb.DMatrix(X_train)
X_test_D = xgb.DMatrix(X_test)
D_train = xgb.DMatrix(X_train, label=y_train)
D_test = xgb.DMatrix(X_test, label=y_test)

#inicjujemy model i podajemy parametry modelu
param = {
    'booster': 'gbtree',
    'objective': 'binary:logistic', 
    'colsample_bytree': 0.5, 
    'learning_rate': 0.1, 
    'min_split_loss': 1, 
    'max_depth': 9, 
    'n_estimators': 80,
    'min_child_weight': 33,
    
    }
steps = 1000
#trenujemy model na danych trenujących
xg_reg = xgb.train(param, D_train, steps)
#wyciągamy dane z modelu dla
preds = xg_reg.predict(X_test_D)
preds2 = xg_reg.predict(X_train_D)
#liczymy średni błąd kwadratowy
rmse = np.sqrt(mean_squared_error(y_test, preds))
rmse2 = np.sqrt(mean_squared_error(y_train, preds2))
print("Test RMSE: %f", rmse)
print("Train RMSE: %f", rmse2)
#np.savetxt(r"N:\Users\Poland\Shared Documents\Debt Collection\COLLECTION_STRATEGY_TEAM\Patryk FF\predykcja.csv", preds, delimiter=",")
#np.savetxt(r"N:\Users\Poland\Shared Documents\Debt Collection\COLLECTION_STRATEGY_TEAM\Patryk FF\y_test.csv", y_test, delimiter=",")

from sklearn.metrics import confusion_matrix
preds_binary = np.array(preds)
preds_binary = preds_binary > 0.5
preds_binary = preds_binary.astype(int)

cm=confusion_matrix(y_test, preds_binary)

TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
print("False positive cases:", FP)
print("False negative cases:", FN)
print("True positive cases:", TP)
print("True negative cases:", TN)
print("//////////////////////////////////////////////")
print("True positive rate:", "{0:0f}%".format(TPR*100))
print("True negative rate:", "{0:0f}%".format(TNR*100))
print("Positive prediction value:", "{0:0f}%".format(PPV*100))
print("Negative predictive value:", "{0:0f}%".format(NPV*100))
print("False positive rate:", "{0:0f}%".format(FPR*100))
print("False negative rate:", "{0:0f}%".format(FNR*100))
print("False discovery rate:", "{0:0f}%".format(FDR*100))
print("//////////////////////////////////////////////")
print("Accuracy:", "{0:0f}%".format(ACC*100))
