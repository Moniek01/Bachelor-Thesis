import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import contingency_matrix

smote = SMOTE(random_state=0)
gb = GradientBoostingClassifier(random_state=0)
pipeline = Pipeline([('smote', smote), ('gb', gb)])

param_grid = {'gb__learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
              'gb__n_estimators': [50, 100, 200, 400, 500],
              'gb__max_depth': [None, 2, 4, 6, 8, 10, 12, 14, 16, 18],
              'gb__min_samples_leaf': [1, 2, 4],
              'gb__min_samples_split': [2, 5, 10],
              'gb__max_features': ['sqrt', 'log2', None],
              'smote__k_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

scorers = {'accuracy': 'accuracy',
           'precision': make_scorer(precision_score, average='weighted'),
           'recall': make_scorer(recall_score, average='weighted'),
           'f1_score': make_scorer(f1_score, average='weighted')}

results = {'dataset': [], 'best parameters': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'contingency matrix': []}

# Neck
Neck = pd.read_pickle('Neck.pkl')
X = Neck[['SubjectID', 'Acc_x_mean', 'Acc_y_mean', 'Acc_z_mean', 'Acc_x_sd', 'Acc_y_sd', 'Acc_z_sd', 'Acc_x_range', 'Acc_y_range', 
          'Acc_z_range', 'Gyr_x_mean', 'Gyr_y_mean', 'Gyr_z_mean', 'Gyr_x_sd', 'Gyr_y_sd', 'Gyr_z_sd', 'Gyr_x_range',
          'Gyr_y_range', 'Gyr_z_range']].values
y = Neck['Fall'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

groups = X_train[:, 0] # 'SubjectID' are the groups
logo = LeaveOneGroupOut()

X_train = np.delete(X_train, 0, axis = 1) # remove 'SubjectID'
X_test = np.delete(X_test, 0, axis = 1)

random_search_neck = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=10, cv=logo, scoring=scorers, refit='f1_score', random_state=0)
random_search_neck.fit(X_train, y_train, groups=groups)

y_pred = random_search_neck.predict(X_test)
results['dataset'].append('Neck')
results['best parameters'].append(random_search_neck.best_params_)
results['accuracy'].append(accuracy_score(y_test, y_pred))
results['precision'].append(precision_score(y_test, y_pred, average = 'weighted'))
results['recall'].append(recall_score(y_test, y_pred, average = 'weighted'))
results['f1'].append(f1_score(y_test, y_pred, average = 'weighted'))
results['contingency matrix'].append(contingency_matrix(y_test, y_pred))

# Waist
Waist = pd.read_pickle('Waist.pkl')
X = Waist[['Acc_x_mean', 'Acc_y_mean', 'Acc_z_mean', 'Acc_x_sd', 'Acc_y_sd', 'Acc_z_sd', 'Acc_x_range', 'Acc_y_range', 
          'Acc_z_range', 'Gyr_x_mean', 'Gyr_y_mean', 'Gyr_z_mean', 'Gyr_x_sd', 'Gyr_y_sd', 'Gyr_z_sd', 'Gyr_x_range',
          'Gyr_y_range', 'Gyr_z_range']].values
y = Waist['Fall'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

groups = X_train[:, 0] # 'SubjectID' are the groups
logo = LeaveOneGroupOut()

X_train = np.delete(X_train, 0, axis = 1) # remove 'SubjectID'
X_test = np.delete(X_test, 0, axis = 1)

random_search_waist = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=10, cv=logo, scoring=scorers, refit='f1_score', random_state=0)
random_search_waist.fit(X_train, y_train, groups=groups)

y_pred = random_search_waist.predict(X_test)
results['dataset'].append('Waist')
results['best parameters'].append(random_search_waist.best_params_)
results['accuracy'].append(accuracy_score(y_test, y_pred))
results['precision'].append(precision_score(y_test, y_pred, average = 'weighted'))
results['recall'].append(recall_score(y_test, y_pred, average = 'weighted'))
results['f1'].append(f1_score(y_test, y_pred, average = 'weighted'))
results['contingency matrix'].append(contingency_matrix(y_test, y_pred))

# Wrist
Wrist = pd.read_pickle('Wrist.pkl')
X = Wrist[['Acc_x_mean', 'Acc_y_mean', 'Acc_z_mean', 'Acc_x_sd', 'Acc_y_sd', 'Acc_z_sd', 'Acc_x_range', 'Acc_y_range', 
          'Acc_z_range', 'Gyr_x_mean', 'Gyr_y_mean', 'Gyr_z_mean', 'Gyr_x_sd', 'Gyr_y_sd', 'Gyr_z_sd', 'Gyr_x_range',
          'Gyr_y_range', 'Gyr_z_range']].values
y = Wrist['Fall'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

groups = X_train[:, 0] # 'SubjectID' are the groups
logo = LeaveOneGroupOut()

X_train = np.delete(X_train, 0, axis = 1) # remove 'SubjectID'
X_test = np.delete(X_test, 0, axis = 1)

random_search_wrist = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=10, cv=logo, scoring=scorers, refit='f1_score', random_state=0)
random_search_wrist.fit(X_train, y_train, groups=groups)

y_pred = random_search_wrist.predict(X_test)
results['dataset'].append('Wrist')
results['best parameters'].append(random_search_wrist.best_params_)
results['accuracy'].append(accuracy_score(y_test, y_pred))
results['precision'].append(precision_score(y_test, y_pred, average = 'weighted'))
results['recall'].append(recall_score(y_test, y_pred, average = 'weighted'))
results['f1'].append(f1_score(y_test, y_pred, average = 'weighted'))
results['contingency matrix'].append(contingency_matrix(y_test, y_pred))

# Get the results
df = pd.DataFrame(results)
df.to_csv('GB results.csv', index=False)

print(results)
