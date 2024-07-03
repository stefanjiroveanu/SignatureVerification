import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from multiprocessing import Pool
from signature.signatureverificationhog import compute_phog
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from knn.knn import KNN
from sklearn.svm import SVC


def label_signature_dataset(current_path):
    data = []
    label_map = {}
    current_label = 0
    for person_dir in os.listdir(current_path):
        person_path = os.path.join(current_path, person_dir)
        if os.path.isdir(person_path):
            if person_dir not in label_map:
                label_map[person_dir] = current_label
                current_label += 1
            for file in os.listdir(person_path):
                if "original" in file:
                    file_path = os.path.join(person_path, file)
                    data.append((file_path, label_map[person_dir]))
    return pd.DataFrame(data, columns=['image_path', 'label']), label_map


def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (50, 25), interpolation=cv2.INTER_AREA)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return compute_phog(img)
    else:
        print("Failed to load image:", image_path)
        return None


def extract_features(image_paths):
    with Pool(processes=os.cpu_count()) as pool:
        features = pool.map(process_image, image_paths)
    return [feature for feature in features if feature is not None]


def find_most_similar_signature(input_signature_path, dataset_features, image_paths):
    input_feature = process_image(input_signature_path)
    input_feature = np.array(input_feature).reshape(1, -1)
    similarities = euclidean_distances(input_feature, dataset_features)  # distanta euclidiana
    print(similarities[0])
    most_similar_index = np.argsort(similarities[0])[
                         :10]  # gaseste cele mai similare 10 si calculeaza un scor pentru accuracy
    print(most_similar_index)
    image_paths = np.array(image_paths)
    return image_paths[most_similar_index]


def find_most_similar_signature(input_feature, dataset_features, image_paths):
    distances = euclidean_distances(input_feature, dataset_features)[0]
    print("Distances:", distances)

    similarities = 1 / (1 + distances)
    print("Similarities:", similarities)

    most_similar_indices = np.argsort(similarities)[-10:][::-1]
    max_similarity = max(similarity for similarity in similarities)

    signatures = []
    for index in most_similar_indices:
        signatures.append((image_paths[index], (1 - (similarities[index] / max_similarity)) * 100))
        print(signatures[-1])

    return signatures


def find_accuracy(dataset_features, dataset_labels, option):
    if option == 0:
        knn = KNN(k=3)
        X_train, X_test, y_train, y_test = train_test_split(dataset_features, dataset_labels, test_size=0.2,
                                                            random_state=42)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        print(classification_report(y_test, predictions))
    if option == 1:
        svm = SVC(C=0.001, gamma=0.1, kernel="poly")
        X_train, X_test, y_train, y_test = train_test_split(dataset_features, dataset_labels, test_size=0.2,
                                                            random_state=42)
        svm.fit(X_train, y_train)
        predictions = svm.predict(X_test)
        print(classification_report(y_test, predictions))


def compute_dataset():
    base_path = "/python-docker/CEDAR"
    dataset, label_map = label_signature_dataset(base_path)
    image_paths = dataset['image_path'].tolist()
    dataset_features = np.array(extract_features(image_paths))
    dataset_labels = np.array(dataset['label'].tolist())
    find_accuracy(dataset_features, dataset_labels, 1)
    return [dataset_features, dataset_labels, image_paths]


if __name__ == "__main__":
    #process_image("../my-signatures/1-fake.jpeg")
    # dataset_features, dataset_labels, image_paths = compute_dataset()
    #
    # input_signature_path = askopenfilename()
    # most_similar_signature = find_most_similar_signature(input_signature_path, dataset_features, image_paths)
    # print("Most similar signature is at:", most_similar_signature)
    # other_img = cv2.imread(input_signature_path, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("First image", other_img)

    # for image in most_similar_signature:
    #     img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    #     cv2.imshow(f"Most Similar Signature${image}", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
