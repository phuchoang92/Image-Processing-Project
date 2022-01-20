from feature_extraction import *

from sklearn.neighbors import KNeighborsClassifier

labels = ['Kyberge_blanket2', 'aluminium_foil', 'Kyberge_blanket2', 'Kyberge_canvas1', 'Kyberge_ceiling1',
          'Kyberge_ceiling2', 'brown_bread', 'cork', 'cotton']

train = get_data('Splited/train', labels)
val = get_data('Splited/valid', labels)

x_train, y_train, x_test, y_test = create_train_data(train)
x_val, y_val = create_val_data(val)

print("Generate bank of filters")
banks = generate_bank_filter1(num_kernels=4)
kernels = generate_bank_filter2()


X_train = feature_extraction1(x_train, banks)
X_test = feature_extraction1(x_test, banks)
X_val = feature_extraction1(x_val, banks)

print("Classification")
KNN_class = KNeighborsClassifier(n_neighbors=5,p=2, weights='distance')
KNN_class.fit(X_train, y_train)
accuracy = KNN_class.score(X_val,y_val)
print(accuracy)

'''resize_ker = cv2.filter2D(x_train[0], ddepth=-1, kernel=banks[2])
cv2.imshow('sea_image.jpg', resize_ker)
cv2.waitKey(0)'''