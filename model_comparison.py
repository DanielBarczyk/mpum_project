from data import Data
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

distributions = ["full", "partial", "even"]

models = [
    LinearSVC(),
    RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]

for distribution in distributions:
    print(distribution)
    data = Data(distribution=distribution)

    features, labels = data.get_data()
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state = 0)

    for model in models:
        print(model.__class__.__name__)
        clf = model.fit(X_train, y_train)
        print("Accuracy:", clf.score(X_test, y_test))
        print()
