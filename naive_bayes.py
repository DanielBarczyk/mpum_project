from data import Data
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB

data = Data()

features, labels = data.get_data()

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.25, random_state = 0)

cnb = ComplementNB().fit(X_train, y_train)
gnb = GaussianNB().fit(X_train, y_train)
mnb = MultinomialNB().fit(X_train, y_train)
bnb = BernoulliNB().fit(X_train, y_train)
ctnb = CategoricalNB().fit(X_train, y_train)

models = [cnb, gnb, mnb, bnb, ctnb]
for m in models:
    print(m.score(X_test, y_test))

for i in range(0, 13):
    m = ComplementNB(alpha = i/8).fit(X_train, y_train)
    print(f'{i/8}:	{m.score(X_test, y_test)}')
