from sklearn.metrics import accuracy_score
from functions import hog_feature, lbp_feature, concatenate, load_svm


clf = load_svm()



def test(svm, X_test, Y_test):
    y_pred = svm.predict(X_test)
    return accuracy_score(Y_test, y_pred)


smile_folder_path = 'dataset/smile'
nosmile_folder_path = 'dataset/no_smile'




smile_images = []
no_smile_images = []

smile_images = find_images(smile_folder_path)
no_smile_images = find_images(nosmile_folder_path)

y_test = []
y_pre = []


for smile_img in smile_images:
    # HOG
    hog_features = hog_feature(smile_img)
    #  LBP
    lbp_features = lbp_feature(smile_img)

    # Combination of HOG & LBP
    features = concatenate(hog_features, lbp_features)

    label = clf.predict([features])[0]

    y_pre.append(label)
    y_test.append(1)

for non_smile_img in no_smile_images:
    # HOG
    hog_features = hog_feature(non_smile_img)
    #  LBP
    lbp_features = lbp_feature(non_smile_img)

    # Combination of HOG & LBP
    features = concatenate(hog_features, lbp_features)
    label = clf.predict([features])[0]

    y_pre.append(label)
    y_test.append(0)

accuracy = accuracy_score(y_test, y_pre)
print('Accuracy of SVM model is: {:.2f}%'.format(accuracy * 100))
