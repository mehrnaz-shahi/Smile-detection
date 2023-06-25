import cv2
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from functions import save_svm, extract_features, load_svm

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
DATA_PATH = "genki/files"
LABEL_PATH = "genki/labels.txt"


def test(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print('y test is : ', y_test)
    print('y predict is : ',  y_pred)
    return accuracy_score(y_test, y_pred)


def extract_labels():
    c = 0
    with open(LABEL_PATH, 'r') as f:
        lines = f.readlines()

    labels = []
    for line in lines:
        c += 1
        labels.append(int(line.split()[0]))

    return labels


# Extracting features and labels
labels = extract_labels()
features = extract_features(DATA_PATH)

# Splitting train and test data
X_train, X_test, Y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel='rbf', C=1))
clf.fit(X_train, Y_train)
# Save SVM model
save_svm(clf)

# Test
svm = load_svm()
print('Test svm model: ...... ')
accuracy = test(svm, X_test, y_test)
print('Accuracy of SVM model is: {:.2f}%'.format(accuracy*100))