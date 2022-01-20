import sys
import random
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import matplotlib.figure
from feature_extraction import *

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier


class App(QMainWindow):
    def __init__(self, parent=None):
        super(App, self).__init__(parent)
        frame = QFrame()
        self.real_only = True
        self.image_label = QLabel()
        self.val_label = ScrollLabel()
        image = QImage("no_img.jpg")
        self.image_label.setPixmap(QPixmap.fromImage(image))

        predict_button = QPushButton("Predict")
        predict_button.clicked.connect(self.image_predict)

        self.predict_label = QLabel()
        self.predict_data_label = QLabel()

        self.image_button = QPushButton("Choose an image")
        self.image_button.clicked.connect(self.image_select)

        self.textEdit = QLineEdit()
        self.data_button = QPushButton("Choose dataset")
        self.data_button.clicked.connect(self.data_select)

        self.test_button = QPushButton("Choose a data")
        self.test_button.clicked.connect(self.select_a_data)

        self.create_data_button = QPushButton("Create data")
        self.create_data_button.clicked.connect(self.create_data)

        self.kernel_box = QGroupBox()
        self.data_box = QGroupBox()
        self.algorithm_box = QGroupBox()
        self.choose_data = QGroupBox()
        self.percent_box = QGroupBox()
        self.Knn_box = QGroupBox()
        self.val_accuracy = QGroupBox()
        val_layout =QVBoxLayout()
        kernel_layout = QGridLayout()
        data_layout = QGridLayout()
        choose_data_layout = QGridLayout()

        self.kernel_choose()
        self.alogrithm_choose()
        self.percentage_choice()
        self.percent_box.setMaximumWidth(300)

        val_layout.addWidget(self.val_label)
        self.val_accuracy.setLayout(val_layout)

        self.Knn_box.setMaximumWidth(200)
        self.Knn_box.setContentsMargins(5, 10, 5, 5)
        self.Knn_box.setTitle("KNN")
        self.algorithm_layout = QHBoxLayout()
        self.algorithm_layout.addWidget(self.Knn_box)
        self.algorithm_layout.addWidget(self.val_accuracy)
        self.algorithm_box.setLayout(self.algorithm_layout)

        choose_data_layout.addWidget(self.data_button, 0, 0, 1, 1)
        choose_data_layout.addWidget(self.textEdit, 1, 0, 1, 2)
        choose_data_layout.addWidget(self.percent_box, 0, 2, 2, 2)
        choose_data_layout.addWidget(self.create_data_button, 0, 4, 1, 1, alignment=QtCore.Qt.AlignCenter)

        self.choose_data.setLayout(choose_data_layout)

        self.figure = matplotlib.figure.Figure()
        self.canvas = FigureCanvas(self.figure)

        self.figure2 = matplotlib.figure.Figure()
        self.canvas2 = FigureCanvas(self.figure2)

        data_layout.addWidget(self.image_button, 0, 0, 1, 1)
        data_layout.addWidget(self.image_label, 2, 0, 1, 1)
        data_layout.addWidget(predict_button, 3, 0, 1, 1)
        data_layout.addWidget(self.predict_label, 4, 0, 1, 1,alignment=QtCore.Qt.AlignCenter)
        data_layout.addWidget(self.test_button, 5, 0, 1, 1)
        data_layout.addWidget(self.predict_data_label, 6, 0, 1, 1,  alignment=QtCore.Qt.AlignCenter)
        data_layout.addWidget(self.canvas2, 0, 2, 10, 10)
        data_layout.addWidget(self.choose_data, 10, 0, 4, 12)
        self.data_box.setLayout(data_layout)
        self.data_box.setMaximumWidth(800)

        kernel_layout.addWidget(self.combo_box, 1, 3, 1, 1)
        kernel_layout.addWidget(self.kernel_button, 1, 4, 1, 1)
        kernel_layout.addWidget(self.canvas, 3, 0, 8, 8)
        self.kernel_box.setLayout(kernel_layout)

        mainLayout = QGridLayout()
        mainLayout.addWidget(self.data_box, 0, 0, 10, 10)
        mainLayout.addWidget(self.kernel_box, 0, 10, 8, 8)
        mainLayout.addWidget(self.algorithm_box, 8, 10, 10, 10)

        self.left = 100
        self.top = 100
        self.width = 1500
        self.height = 800

        self.set_style_sheet()

        frame.setLayout(mainLayout)
        self.setCentralWidget(frame)
        self.setWindowTitle('Texture Classification')
        self.setGeometry(self.left, self.top, self.width, self.height)

    def kernel_choose(self):
        self.combo_box = QComboBox(self)
        kernel_list = ["Only Real Part", "Have Real and Imag Part"]
        self.combo_box.addItems(kernel_list)
        self.kernel_button = QPushButton("Generate")
        self.kernel_button.clicked.connect(self.generate_gabor_filter)

    def select_a_data(self):
        try:
            self.folderpath2 = QFileDialog.getExistingDirectory(self, 'Select Folder')
            self.predict_a_data()
        except:
            self.error_dialog = QtWidgets.QErrorMessage()
            self.error_dialog.showMessage('You miss someting here')

    def data_select(self):
        try:
            self.folderpath = QFileDialog.getExistingDirectory(self, 'Select Folder')
            self.textEdit.setText(self.folderpath)
        except:
            self.error_dialog = QtWidgets.QErrorMessage()
            self.error_dialog.showMessage('Please choose a data set !')

    def predict_a_data(self):
        label = os.path.basename(self.folderpath2)
        test_x, test_y = get_test_data(self.folderpath2,self.list_labels, label, self.bank_filters)
        predictions = self.classifier.predict(test_x)
        self.predict_data_label.setText(str(accuracy_score(test_y, predictions)))

    def create_data(self):
        try:
            self.percentage_data()
            train = get_data('Splited/train', self.list_labels)
            val = get_data('Splited/valid', self.list_labels)

            X_train, self.y_train, x_test, self.y_test = create_train_data(train)
            x_val, self.y_val = create_val_data(val)

            if self.bank_filters is not None:
                if self.real_only:
                    self.X_train = feature_extraction1(X_train, self.bank_filters)
                    self.X_test = feature_extraction1(x_test, self.bank_filters)
                    self.X_val = feature_extraction1(x_val, self.bank_filters)
                else:
                    self.X_train = feature_extraction2(X_train, self.bank_filters)
                    self.X_test = feature_extraction2(x_test, self.bank_filters)
                    self.X_val = feature_extraction2(x_val, self.bank_filters)


            print(self.X_train.shape)
            self.error_dialog = QtWidgets.QErrorMessage()
            self.error_dialog.showMessage('Dataset has been created successfully')
        except:
            self.error_dialog = QtWidgets.QErrorMessage()
            self.error_dialog.showMessage('Please, create bank of filters first')

    def percentage_data(self):
        self.list_labels = []
        for img in os.listdir(self.folderpath):
            self.list_labels.append(img)
        length = len(self.list_labels)
        random.shuffle(self.list_labels)
        if self.plc_radio1.isChecked():
            self.list_labels = self.list_labels[0:int(length / 10)]
        if self.plc_radio2.isChecked():
            self.list_labels = self.list_labels[0:int(length / 5)]
        if self.plc_radio3.isChecked():
            self.list_labels = self.list_labels[0:int(length / 2)]
        print(self.list_labels)

    def percentage_choice(self):
        layout = QVBoxLayout()
        self.percenttage_button = QButtonGroup()

        self.plc_radio1 = QRadioButton('10%')
        self.plc_radio2 = QRadioButton('20%')
        self.plc_radio3 = QRadioButton('50%')
        self.plc_radio4 = QRadioButton('100%')

        self.percenttage_button.addButton(self.plc_radio1)
        self.percenttage_button.addButton(self.plc_radio2)
        self.percenttage_button.addButton(self.plc_radio3)
        self.percenttage_button.addButton(self.plc_radio4)

        layout.setContentsMargins(10, 5, 5, 5)
        layout.addWidget(self.plc_radio1)
        layout.addWidget(self.plc_radio2)
        layout.addWidget(self.plc_radio3)
        layout.addWidget(self.plc_radio4)

        self.percent_box.setLayout(layout)

    def alogrithm_choose(self):
        layout = QFormLayout()

        label = QLabel("Neighbor:")
        self.number_of_neigh = QLineEdit()
        self.fit_button = QPushButton("Fit")
        self.fit_button.clicked.connect(self.Knn)

        layout.addRow(label, self.number_of_neigh)
        layout.addRow(self.fit_button)

        self.Knn_box.setLayout(layout)

    def Knn(self):
        number_of_neigh = int(self.number_of_neigh.text())
        self.classifier = KNeighborsClassifier(n_neighbors=number_of_neigh, p=2, weights='distance')
        self.classifier.fit(self.X_train, self.y_train)
        if True:
            print(self.classifier)
            self.error_dialog = QtWidgets.QErrorMessage()
            self.error_dialog.showMessage('Classifier created successfully')
        predictions = self.classifier.predict(self.X_val)
        self.val_label.setText(classification_report(self.y_val, predictions))

    def image_select(self):
        try:
            file_filter = 'Data File (*.jpg *.png *.data);; Excel File (*.xlsx *.xls)'
            path, _ = QFileDialog.getOpenFileName(parent=self, caption='Select a Data File', filter=file_filter,
                                                  options=QFileDialog.DontUseNativeDialog)
            image = QImage(path)
            new_iamge = image.scaled(168, 168, Qt.KeepAspectRatio)
            self.image_label.setPixmap(QPixmap.fromImage(new_iamge))
            self.image_for_test = cv2.imread(path)
            test_image = cv2.cvtColor(self.image_for_test, cv2.COLOR_BGR2GRAY)
            self.image_for_test = cv2.resize(test_image, (200, 200))
            self.apply_gabor_for_anImage(self.image_for_test)
        except Exception :
            self.error_dialog = QtWidgets.QErrorMessage()
            self.error_dialog.showMessage('Please choose a data set !')

    def image_predict(self):
        if self.real_only:
            feature_vector = feature_extract_of_image(self.image_for_test,self.bank_filters)
        else:
            feature_vector = apply_gabor_filter(self.image_for_test, self.bank_filters)
        feature_vector = feature_vector.reshape(1,-1)
        prediction = self.classifier.predict(feature_vector)
        self.predict_label.setText(self.list_labels[prediction[0]])

    def apply_gabor_for_anImage(self, image):
        self.figure2.clf()
        for i in range(1, len(self.bank_filters) + 1):
            ax = self.figure2.add_subplot(6, 4, i)
            feature_image = cv2.filter2D(image, ddepth=-1, kernel=np.real(self.bank_filters[i-1]))
            feature_image = cv2.resize(feature_image, (50, 50))
            ax.imshow(feature_image, cmap='gray')
            ax.set_aspect('equal')
            ax.axis('off')
        self.figure2.tight_layout(pad=0.5)
        self.canvas2.draw_idle()

    def generate_gabor_filter(self):
        self.figure.clf()
        self.bank_filters = []
        if self.combo_box.currentText() == "Only Real Part":
            self.bank_filters = generate_bank_filter1(num_kernels=4)
        if self.combo_box.currentText() == "Have Real and Imag Part":
            self.bank_filters = generate_bank_filter2()
            self.real_only = False
        for i in range(1, len(self.bank_filters) + 1):
            ax = self.figure.add_subplot(6, 4, i)
            new = cv2.resize(np.real(self.bank_filters[i - 1]), (50, 50))
            ax.imshow(new, cmap='gray')
            ax.set_aspect('equal')
            ax.axis('off')
        self.figure.tight_layout(pad=0.5)
        self.canvas.draw_idle()

    def set_style_sheet(self):
        '''self.setStyleSheet("QLabel{font-size: 20px;} QRadioButton {font : 16px Arial;} "
                           "QPushButton{font : 18px Arial;border-radius : 10px;; border : 2px solid black;"
                           "background:qradialgradient(cx: 0.3, cy: -0.4, fx: 0.3, fy: -0.4,radius: 1.35, stop: 0 #fff, stop: 1 #bbb)}")
        self.label1.setStyleSheet("QLabel{font-size: 15px;}")
        self.label2.setStyleSheet("QLabel{font-size: 15px;}")
        self.buttonBox.setStyleSheet("QGroupBox {border: none;}")'''


class ScrollLabel(QScrollArea):

    # constructor
    def __init__(self, *args, **kwargs):
        QScrollArea.__init__(self, *args, **kwargs)

        # making widget resizable
        self.setWidgetResizable(True)

        # making qwidget object
        content = QWidget(self)
        self.setWidget(content)

        # vertical box layout
        lay = QVBoxLayout(content)

        # creating label
        self.label = QLabel(content)

        # setting alignment to the text
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        # making label multi-line
        self.label.setWordWrap(True)

        # adding label to the layout
        lay.addWidget(self.label)

    # the setText method
    def setText(self, text):
        # setting text to the label
        self.label.setText(text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
