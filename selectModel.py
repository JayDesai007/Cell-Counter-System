from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_selectModel(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(362, 426)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.modelNameListWidget = QtWidgets.QListWidget(Form)
        self.modelNameListWidget.setLineWidth(5)
        self.modelNameListWidget.setObjectName("modelNameListWidget")
        self.verticalLayout.addWidget(self.modelNameListWidget)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.doneButton = QtWidgets.QPushButton(Form)
        self.doneButton.setObjectName("doneButton")
        self.horizontalLayout.addWidget(self.doneButton)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "Select Model:"))
        self.doneButton.setText(_translate("Form", "Done"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    EVALUATETeam = QtWidgets.QWidget()
    ui = Ui_selectModel()
    ui.setupUi(EVALUATETeam)
    EVALUATETeam.show()
    sys.exit(app.exec_())