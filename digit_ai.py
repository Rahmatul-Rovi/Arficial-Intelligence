from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ১. Digits dataset load kora
digits = datasets.load_digits()

# ২. Image-ke flat array-te convert kora
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# ৩. Support Vector Machine (SVM) Classifier setup
clf = svm.SVC(gamma=0.001)

# ৪. Data-ke split kora (50% train, 50% test)
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

# ৫. AI-ke train kora
clf.fit(X_train, y_train)

# ৬. Prediction kora
predicted = clf.predict(X_test)

# ৭. Accuracy check
print(f"Classification report:\n{metrics.classification_report(y_test, predicted)}")

# Optional: Ekta test image dekhar jonno
plt.imshow(digits.images[10], cmap=plt.cm.gray_r, interpolation='nearest')
plt.title(f"Prediction: {predicted[0]}")
plt.show()