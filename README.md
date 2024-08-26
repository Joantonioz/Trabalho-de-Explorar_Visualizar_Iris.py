# Trabalho-de-Explorar_Visualizar_Iris.py
# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Convert the data to a DataFrame for easier manipulation
data = pd.DataFrame(X, columns=feature_names)
data['target'] = y

# Exploratory Data Analysis
def plot_pairplot():
    """Plot pairplot to visualize the relationships between features."""
    sns.pairplot(data, hue='target', markers=["o", "s", "D"], palette='husl')
    plt.title("Pairplot of Iris Dataset")
    plt.show()
    
plot_pairplot()

# Feature Selection using SelectKBest
def feature_selection(X, y, k=2):
    """Select k best features using ANOVA F-statistic."""
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    return selector.transform(X), selector.get_support(indices=True)

# Select the best 2 features
X_selected, selected_indices = feature_selection(X, y, k=2)
print("Selected Features:", [feature_names[i] for i in selected_indices])

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Model Creation and Hyperparameter Tuning with GridSearchCV
def grid_search_rf(X_train, y_train):
    """Perform Grid Search to find the best parameters for Random Forest Classifier."""
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    print("Best Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

# Fit the model
best_rf_model = grid_search_rf(X_train, y_train)

# Making Predictions
y_pred = best_rf_model.predict(X_test)

# Evaluating the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=target_names))

# Visualizing Feature Importance
def plot_feature_importance(model, feature_names):
    """Plot feature importances from the model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure()
    plt.title("Feature Importances")
    plt.bar(range(X_selected.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_selected.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, X_selected.shape[1]])
    plt.show()

plot_feature_importance(best_rf_model, [feature_names[i] for i in selected_indices])
