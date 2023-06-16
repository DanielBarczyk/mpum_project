from data import Data
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd

distributions = ["full", "partial", "even"]

models = [
    LinearSVC(),
    RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0),
    ComplementNB(alpha=3/8),
    LogisticRegression(random_state=0),
    MLPClassifier(hidden_layer_sizes=(300, 16)),
]

named_labels = ["enfj", "enfp", "entj", "entp", "esfj", "esfp", "estj", "estp", "infj", "infp", "intj", "intp", "isfj", "isfp", "istj", "istp"]

for distribution in distributions:
    print(distribution)
    data = Data(distribution=distribution)

    features, labels = data.get_data()
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state = 0)

    for model in models:
        print(model.__class__.__name__)
        clf = model.fit(X_train, y_train)
        print("Accuracy:", clf.score(X_test, y_test))
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion matrix:\n", pd.DataFrame(cm, columns = named_labels, index = named_labels))
        print()
