from data import Data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = Data(distribution="full")

features, labels = data.get_data()

features, labels = data.get_data()

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state = 0)

clf = SVC(kernel='linear', C=1).fit(X_train, y_train)
print(clf.score(X_test, y_test))
for i in range(0, 100):
    print("Prediction:", clf.predict(X_test[i].reshape(1, -1)))
    print("Real:", y_test[i], data.index_to_label(y_test[i]))
    print()