import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer

file_path = 'C:/Users/91630/Downloads/code_only.csv'
your_data = pd.read_csv(file_path, header=None, dtype=str)  


y = your_data.iloc[:, 0].astype(float)  


X = X.apply(pd.to_numeric, errors='coerce')


threshold = y.mean()  
y_binary = (y > threshold).astype(int)


X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

knn = KNeighborsClassifier()


param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15]}


grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')


grid_search.fit(X_train_imputed, y_train)

print("Best hyperparameters:", grid_search.best_params_)


best_knn = grid_search.best_estimator_


accuracy = best_knn.score(X_test_imputed, y_test)
print("Test set accuracy:", accuracy)
