import os
import cv2
import numpy as np
from skimage.filters import gabor_kernel
from sklearn.model_selection import train_test_split


def get_data(data_dir, labels):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_number = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))
                gray_scaled = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
                resized_image = cv2.resize(gray_scaled, (200, 200))
                data.append([resized_image, class_number])
            except Exception as e:
                print(e)
    return np.array(data)


def create_train_data(train_data):
    x_train = []
    y_train = []
    for feature, label in train_data:
        x_train.append(feature)
        y_train.append(label)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4,stratify=y_train)
    return x_train, y_train, x_test, y_test


def create_val_data(val_data):
    x_val = []
    y_val = []
    for feature, label in val_data:
        x_val.append(feature)
        y_val.append(label)
    return x_val, y_val


def generate_bank_filter1(num_kernels, ksize=(20, 20), lambd=6, psi=0):
    bank = []
    step = np.pi / num_kernels
    for index in range(num_kernels):
        theta = index * step
        for sigma in (2, 4, 6):
            for gamma in (0.25, 0.5):
                kernel = cv2.getGaborKernel(ksize=ksize, sigma=sigma, theta=theta, lambd=lambd, gamma=gamma, psi=psi)
                kernel /= 1.5 * kernel.sum()
                bank.append(kernel)
    return bank


def feature_extraction1(x_train, bank):
    new_x_train = []
    for img in x_train:
        static = np.zeros((len(bank), 1), dtype=np.float64)
        for i, kernel in enumerate(bank):
            new_image = cv2.filter2D(img, ddepth=-1, kernel=kernel)
            new_image = new_image / 255
            static[i] = new_image.mean()
        new_x_train.append(np.transpose(static))
    X_train = np.array(new_x_train)
    return np.squeeze(X_train)


def feature_extract_of_image(image, bank):
    feature = []
    for i, kernel in enumerate(bank):
        new_image = cv2.filter2D(image, ddepth=-1, kernel=kernel)
        new_image = new_image / 255
        feature.append(new_image.mean())
    features = np.array(feature,dtype=np.float64)
    return np.squeeze(features)


def generate_bank_filter2():
    kernels = []
    for theta in (0, np.pi / 4, np.pi / 2, 2 * np.pi / 3):
        for sigma in (5, 7, 9, 11, 15):
            kernel = gabor_kernel(frequency=0.05, theta=theta, sigma_x=sigma, sigma_y=sigma)
            kernels.append(kernel)
    return kernels


def apply_gabor_filter(image, kernels):
    feature_vector = []
    for  kernel in kernels:
        real_filter = cv2.filter2D(image, ddepth=-1, kernel=np.real(kernel))
        imag_filter = cv2.filter2D(image, ddepth=-1, kernel=np.imag(kernel))
        new_image = np.sqrt(real_filter ** 2 + imag_filter ** 2)
        new_image = np.array(new_image, np.uint8)
        feature_vector.append(new_image.mean())
    return np.squeeze(feature_vector)


def feature_extraction2(x_train, banks):
    X_train = []
    for image in x_train:
        feature_vector = apply_gabor_filter(image, banks)
        X_train.append(feature_vector)
    X_train = np.array(X_train)
    return np.squeeze(X_train)


def get_test_data(path,list_label, label, banks):
    x_test, y_test = [], []
    class_index = list_label.index(label)
    for img in os.listdir(path):
        try:
            img_arr = cv2.imread(os.path.join(path, img))
            gray_scaled = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray_scaled, (200, 200))
            feature_vector = apply_gabor_filter(resized_image, banks)
            x_test.append(feature_vector)
            y_test.append(class_index)
        except Exception as e:
            print(e)
    x_test = np.array(x_test)
    return np.squeeze(x_test), y_test

