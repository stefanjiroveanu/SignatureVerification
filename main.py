import os
import sys

import cv2
import numpy as np
import pandas as pd

from knn.knn import KNN
from signature.signatureverificationhog import compute_phog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from multiprocessing import Pool
from joblib import dump, load
from tkinter.filedialog import askopenfilename
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt




def label_cedar_dataset(current_path, forgery_label):
    data = []
    for person_dir in os.listdir(current_path):
        person_path = os.path.join(current_path, person_dir)
        if os.path.isdir(person_path):
            for file in os.listdir(person_path):
                if file.startswith("original"):
                    label = 1
                elif file.startswith("forgeries"):
                    label = forgery_label
                else:
                    continue

                file_path = os.path.join(person_path, file)
                data.append((file_path, label))

    return pd.DataFrame(data, columns=['image_path', 'label'])


def label_cedar_split_dataset(current_path):  # not working idk why
    test = []
    train = []
    for dir in os.listdir(current_path):
        dir_path = os.path.join(current_path, dir)
        for person_dir in os.listdir(dir_path):
            person_path = os.path.join(dir_path, person_dir)
            if os.path.isdir(person_path):
                for file in os.listdir(person_path):
                    if file.startswith("original"):
                        label = 1
                    elif file.startswith("forgeries"):
                        label = 0
                    else:
                        continue

                    file_path = os.path.join(person_path, file)
                    if dir == "test":
                        test.append((file_path, label))
                    elif dir == "train":
                        train.append((file_path, label))

    return pd.DataFrame(train, columns=['image_path', 'label']), pd.DataFrame(test, columns=['image_path', 'label'])


def resize_image(image, size=(50, 25)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img_resized = resize_image(img)
        _, binary_img = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        feature = compute_phog(binary_img)
        return feature
    else:
        print("Failed to load image", image_path)
        return None


def read_normal_cedar(classifier_name):
    base_path = "CEDAR"
    dataset = []
    if classifier_name == "SVC":
        dataset = label_cedar_dataset(base_path, -1)
    elif classifier_name == "KNN":
        dataset = label_cedar_dataset(base_path, 0)
    image_paths = dataset['image_path'].tolist()
    features = []
    labels = dataset['label'].tolist()
    with Pool(processes=10) as pool:
        features = pool.map(process_image, image_paths)
    features, labels = zip(*[(feature, label) for feature, label in zip(features, labels) if feature is not None])
    X = np.array(features)
    y = np.array(labels)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def read_cedar_split():
    base_path = "CEDAR_SPLIT"
    train, test = label_cedar_split_dataset(base_path)
    train_image_paths = train['image_path'].tolist()
    test_image_paths = test['image_path'].tolist()
    train_labels = train['label'].tolist()
    test_labels = test['label'].tolist()
    train_features = []
    test_features = []
    with Pool(processes=10) as pool:
        train_features = pool.map(process_image, train_image_paths)
    train_features, train_labels = zip(
        *[(feature, label) for feature, label in zip(train_features, train_labels) if feature is not None])
    with Pool(processes=10) as pool:
        test_features = pool.map(process_image, test_image_paths)
    test_features, test_labels = zip(
        *[(feature, label) for feature, label in zip(test_features, test_labels) if feature is not None])
    X_train = np.array(train_features)
    y_train = np.array(train_labels)
    X_test = np.array(test_features)
    y_test = np.array(test_labels)
    return X_train, X_test, y_train, y_test


def find_SVM_best_params():
    global parameters, clf
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    import numpy as np
    parameters = {
        'C': np.logspace(-3, 3, 7),
        'gamma': np.logspace(-3, 3, 7),
        'kernel': ['linear', 'rbf', 'poly']
    }
    clf = GridSearchCV(classifier, parameters, cv=5, n_jobs=-1, verbose=10)
    clf.fit(X_train, y_train)
    print("Best parameters:", clf.best_params_)


if __name__ == "__main__":
    option = int(input("choose between SVM and KNN: \n1 -> SVM \n2 -> KNN\n"))
    classifier_name = ""

    if option == 1:
        classifier = SVC(C=0.001, gamma=0.1, kernel="poly") #0.001 0.1 poly est
        classifier_name = "SVC"
    elif option == 2:
        classifier = KNN(k=1)
        classifier_name = "KNN"
    else:
        print("not a valid option")
        sys.exit()

    if os.path.exists('SVC.joblib') and classifier_name == "SVC" or os.path.exists('KNN.joblib') and classifier_name == "KNN":
        filename = askopenfilename()
        img_features = process_image(filename).reshape(1, -1)
        classifier = load(f"{classifier_name}.joblib")
        prediction = classifier.predict(img_features)
        if prediction[0] > 0:
            print("The signature is Geniune")
        else:
            print("The signature is Fake")
        sys.exit()

    X_train, X_test, y_train, y_test = read_normal_cedar(classifier_name)
    # X_train, X_test, y_train, y_test = read_cedar_split()

    print("Started classifying using", classifier_name)
    classifier.fit(X_train, y_train)

    #find_SVM_best_params()

    predictions = classifier.predict(X_test)
    print(classification_report(y_test, predictions))
    confusian_matrix = metrics.confusion_matrix(y_test, predictions)
    print(confusian_matrix)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusian_matrix, display_labels=[0, 1])
    cm_display.plot()
    plt.show()

    dump(classifier, f"{classifier_name}.joblib")
