import cv2

from joblib import load
from tkinter.filedialog import askopenfilename

if __name__ == '__main__':
    current_image_path = askopenfilename()
    svm = load('svm_for_recognition.joblib')
    current_image = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)
    recognized_image_path = svm.predict(current_image)
    recognized_image = cv2.imread(recognized_image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("input image", current_image)
    cv2.imshow("recognized image", recognized_image)
    cv2.waitKey()


