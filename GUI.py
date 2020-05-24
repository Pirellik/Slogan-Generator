import sys
from PySide2 import QtWidgets, QtGui

class GUI(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(GUI, self).__init__(parent)

        self.logsPath = ""

        mainLayout = QtWidgets.QGridLayout()

        
        l = ["Apparel slogans","Automotive slogans","Beauty slogans","Beverage slogans","Business slogans","Construction slogans","Dining slogans","Educational slogans","Financial service slogans","Casino slogans","Computers slogans","Condoms slogans","Magagines slogans","Motorcycle slogans","Newspapers slogans","Pet food slogans","Radio stations slogans","Real estate slogans","Tobacco slogans","Vitamins slogans","Watch slogans"]
        lista = []

        for i in range(len(l)):
            lista.append(QtWidgets.QCheckBox(l[i]))
            mainLayout.addWidget(lista[i], i + 1, 0)

        wyborKategoriLabel = QtWidgets.QLabel("Wybierz kategorie")

        sredniaLiczbaLabel = QtWidgets.QLabel("Ustawienie dlugosci wektora")
        srednia = QtWidgets.QRadioButton("srednia")
        liczba = QtWidgets.QRadioButton("liczba")
        dlugoscWektoraLabel = QtWidgets.QLabel("Dlugosc wektora")
        dlugoscWektora = QtWidgets.QLineEdit()

        krokRamkiLabel = QtWidgets.QLabel("Ustalenie kroku ramki")
        krokRamki = QtWidgets.QLineEdit()

        liczbaEpokLabel = QtWidgets.QLabel("Wprowadzenie liczby epok")
        liczbaEpok = QtWidgets.QLineEdit()

        self.inputBt = QtWidgets.QPushButton("Wczytaj tekst")
        self.inputBt.clicked.connect(self.onInputBtClicked)

        dokladnaDlugosc = QtWidgets.QRadioButton("Dokladna dlugosc")
        dlugoscMaksymalna = QtWidgets.QRadioButton("Maksymalna dlugosc")

        self.generujBt = QtWidgets.QPushButton("Generuj slogany")

        liczbaSloganowLabel = QtWidgets.QLabel("Podaj liczbe sloganow")
        liczbaSloganow = QtWidgets.QLineEdit()


        dlugoscWektora.setValidator(QtGui.QIntValidator())
        krokRamki.setValidator(QtGui.QIntValidator())
        liczbaEpok.setValidator(QtGui.QIntValidator())

        mainLayout.addWidget(wyborKategoriLabel, 0, 0)
        mainLayout.addWidget(sredniaLiczbaLabel, 0, 2)
    
        mainLayout.addWidget(srednia, 1, 2)
        mainLayout.addWidget(liczba, 1, 3)
        mainLayout.addWidget(dlugoscWektoraLabel, 2, 2)
        mainLayout.addWidget(dlugoscWektora, 3, 2)

        mainLayout.addWidget(krokRamkiLabel, 5, 2)
        mainLayout.addWidget(krokRamki, 6, 2)

        mainLayout.addWidget(liczbaEpokLabel, 8, 2)
        mainLayout.addWidget(liczbaEpok, 9, 2)

        mainLayout.addWidget(liczbaSloganowLabel, 15, 2)
        mainLayout.addWidget(liczbaSloganow, 16, 2)

        mainLayout.addWidget(self.inputBt, 11, 2)

        mainLayout.addWidget(dlugoscMaksymalna, 13, 2)
        mainLayout.addWidget(dokladnaDlugosc, 13, 3)

        mainLayout.addWidget(self.generujBt, 20, 2)

        

        self.setLayout(mainLayout)
        self.setWindowTitle("Generator Sloganow")

    def onInputBtClicked(self):
        self.logsPath = QtWidgets.QFileDialog().getOpenFileName(self, "Wczytaj tekst")
        print(self.logsPath)





app = QtWidgets.QApplication(sys.argv)
obj = GUI()
obj.show()

sys.exit(app.exec_())