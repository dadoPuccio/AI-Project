from sklearn import tree
from sklearn.model_selection import cross_val_score
import numpy as np
import csv
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# apertura traing set
with open("avila-tr.txt") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    x = []
    z = []
    y = []
    for row in csv_reader:
        for i in range(0, 10):
            z.append(float(row[i]))
        y.append(row[10])
        x.append(z)
        z = []
        line_count += 1
    print(f'Training set has {line_count} samples.')


# ricerca del miglior parametro per min_samples_leaf
n = []
for i in range(1, 6):
    clf = tree.DecisionTreeClassifier(criterion="entropy", min_samples_leaf=i)
    scores = cross_val_score(estimator=clf, X=x, y=y, cv=10)
    n.append((i, np.mean(scores)))

bestN = 0
bestScore = 0
for i in n:
    if bestScore < i[1]:
        bestScore = i[1]
        bestN = i[0]
print(f"The best value found for min_samples_leaf using 10-fold cross-validation is {bestN}, with a score of {bestScore}.")

tmp = np.array((n[0], n[1], n[2], n[3], n[4])).T
np.savetxt("Nl.csv", tmp, fmt="%.8s,%.8s,%.8s,%.8s,%.8s",
           header="", comments="")

print("All scores are available in 'Nl.csv'.")


clf = tree.DecisionTreeClassifier(criterion="entropy", min_samples_leaf=bestN)
clf = clf.fit(x, y)

# apertura test set
with open("avila-ts.txt") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    x = []
    z = []
    y = []
    for row in csv_reader:
        for i in range(0, 10):
            z.append(float(row[i]))
        y.append(row[10])
        x.append(z)
        z = []
        line_count += 1
    print(f'Test set has {line_count} samples.')

# conta quante predizioni sono state corrette per ogni scriba
mapOfCorrect = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0,
                "F": 0, "G": 0, "H": 0, "I": 0, "W": 0, "X": 0, "Y": 0}

# conta il numero di volte in cui è stato assegnato l'"N-esimo" scriba
mapOfPredicitions = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0,
                     "F": 0, "G": 0, "H": 0, "I": 0, "W": 0, "X": 0, "Y": 0}

# conta il totale di esempi di ciascuno scriba
mapOfTotal = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0,
              "F": 0, "G": 0, "H": 0, "I": 0, "W": 0, "X": 0, "Y": 0}

while len(y) > 0:
    pred = clf.predict([x.pop()])
    res = y.pop()[0]
    if(pred[0] == res):
        mapOfCorrect[res] += 1

    mapOfPredicitions[pred[0]] += 1
    mapOfTotal[res] += 1

precision = []
recall = []
keys = list(mapOfCorrect.keys())
for i in keys:
    precision.append(mapOfCorrect[i] / mapOfPredicitions[i] * 100)
    recall.append(mapOfCorrect[i] / mapOfTotal[i] * 100)

tmp = np.array((keys, precision, recall)).T
np.savetxt("results.csv", tmp, fmt="%.6s,%.6s,%.6s",
           header="Writer,Precision,Recall", comments="")

print("Results available in 'results.csv'.")
