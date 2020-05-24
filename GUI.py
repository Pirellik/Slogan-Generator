from threading import Thread
import sys
from slogans_nlp import *
from PySide2 import QtWidgets, QtGui

class GUI(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(GUI, self).__init__(parent)

        self.logsPath = ""

        self.mainLayout = QtWidgets.QGridLayout()

        self.dupa = QtWidgets.QGridLayout()

        
        self.l = ["Apparel slogans","Automotive slogans","Beauty slogans","Beverage slogans","Business slogans","Construction slogans","Dining slogans","Educational slogans","Financial service slogans","Casino slogans","Computers slogans","Condoms slogans","Magagines slogans","Motorcycle slogans","Newspapers slogans","Pet food slogans","Radio stations slogans","Real estate slogans","Tobacco slogans","Vitamins slogans","Watch slogans"]
        self.lista = []

        for i in range(len(self.l)):
            self.lista.append(QtWidgets.QCheckBox(self.l[i]))
            self.mainLayout.addWidget(self.lista[i], i + 1, 0)

        wyborKategoriLabel = QtWidgets.QLabel("Wybierz kategorie")

        sredniaLiczbaLabel = QtWidgets.QLabel("Ustawienie dlugosci wektora")
        self.srednia = QtWidgets.QRadioButton("srednia")
        liczba = QtWidgets.QRadioButton("liczba")
        dlugoscWektoraLabel = QtWidgets.QLabel("Dlugosc wektora")
        self.dlugoscWektora = QtWidgets.QLineEdit()

        krokRamkiLabel = QtWidgets.QLabel("Ustalenie kroku ramki")
        self.krokRamki = QtWidgets.QLineEdit()

        liczbaEpokLabel = QtWidgets.QLabel("Wprowadzenie liczby epok")
        self.liczbaEpok = QtWidgets.QLineEdit()

        self.trainBt = QtWidgets.QPushButton("Trenuj siec")
        self.trainBt.clicked.connect(self.onTrainBtClicked)

        self.inputBt = QtWidgets.QPushButton("Wczytaj model")
        

        self.dokladnaDlugosc = QtWidgets.QRadioButton("Dokladna dlugosc")
        dlugoscMaksymalna = QtWidgets.QRadioButton("Maksymalna dlugosc")
        self.dlugosc = QtWidgets.QLineEdit()


        self.generujBt = QtWidgets.QPushButton("Generuj slogany")
        self.generujBt.clicked.connect(self.onGenerateBtClicked)

        liczbaSloganowLabel = QtWidgets.QLabel("Podaj liczbe sloganow")
        self.liczbaSloganow = QtWidgets.QLineEdit()

        self.progressBarTrain = QtWidgets.QProgressBar(self)
        
        self.progressBar = QtWidgets.QProgressBar(self)


        self.dlugoscWektora.setValidator(QtGui.QIntValidator())
        self.krokRamki.setValidator(QtGui.QIntValidator())
        self.liczbaEpok.setValidator(QtGui.QIntValidator())

        self.mainLayout.addWidget(wyborKategoriLabel, 0, 0)
        self.mainLayout.addWidget(sredniaLiczbaLabel, 0, 2)
    
        self.mainLayout.addWidget(self.srednia, 1, 2)
        self.mainLayout.addWidget(liczba, 1, 3)
        self.mainLayout.addWidget(dlugoscWektoraLabel, 2, 2)
        self.mainLayout.addWidget(self.dlugoscWektora, 3, 2)

        self.mainLayout.addWidget(krokRamkiLabel, 5, 2)
        self.mainLayout.addWidget(self.krokRamki, 6, 2)

        self.mainLayout.addWidget(liczbaEpokLabel, 8, 2)
        self.mainLayout.addWidget(self.liczbaEpok, 9, 2)

        self.mainLayout.addWidget(liczbaSloganowLabel, 15, 2)
        self.mainLayout.addWidget(self.liczbaSloganow, 16, 2)

        self.mainLayout.addWidget(self.trainBt, 10, 2)
        self.mainLayout.addWidget(self.progressBarTrain, 11, 2)
        self.mainLayout.addWidget(self.inputBt, 12, 2)

        self.mainLayout.addWidget(dlugoscMaksymalna, 14, 2)
        self.mainLayout.addWidget(self.dokladnaDlugosc, 14, 3)
        self.mainLayout.addWidget(self.dlugosc, 15, 2)

        self.mainLayout.addWidget(self.generujBt, 20, 2)

        self.mainLayout.addWidget(self.progressBar, 21 ,2)

        self.progressBar.hide()
        self.progressBarTrain.hide()
        self.setLayout(self.mainLayout)
        self.setWindowTitle("Generator Sloganow")

    def onTrainBtClicked(self):
        self.progressBarTrain.show()
        self.progressBarTrain.setMaximum(0)
        self.progressBarTrain.setMinimum(0)
        self.progressBarTrain.setValue(0)
        df = read_data_file('input_data.csv')
        unwanted = list()
        for category, checkbox in zip(self.l, self.lista):
            if not checkbox.isChecked():
                unwanted.append(category)
        df = drop_unwanted_categories(df, unwanted)

        plain_text = convert_to_plain_text(df)
        chars = get_chars(plain_text)
        char_indices, indices_char = get_char_and_indices_dicts(chars)
        slg_lengths = get_slogan_lengths(df)
        if self.srednia.isChecked:
            max_len = get_max_len(True, slg_lengths)
        else:
            max_len = get_max_len(False, slg_lengths, int(self.dlugoscWektora.text()))

        step = int(self.krokRamki.text())
        x, y = get_x_and_y(plain_text, max_len, step, chars, char_indices)
        model = build_model(max_len, chars)
        thread = Thread(target = train_network, args = (model, x, y, int(self.liczbaEpok.text()), self.progressBarTrain))
        thread.start()

    def onGenerateBtClicked(self):
        self.setLayout(self.dupa)
        self.progressBar.show()
        self.progressBar.setMaximum(0)
        self.progressBar.setMinimum(0)
        self.progressBar.setValue(0)
        df = read_data_file('input_data.csv')
        slogan_lengths = get_slogan_lengths(df)
        all_slogans_as_text = convert_to_plain_text(df)
        chars = get_chars(all_slogans_as_text)
        char_indices, indices_char = get_char_and_indices_dicts(chars)
        if self.srednia.isChecked:
            maxlen = get_max_len(True, slg_lengths)
        else:
            maxlen = get_max_len(False, slg_lengths, int(self.dlugoscWektora.text()))
        model = get_saved_model(maxlen, chars, "weights.hdf5")

        number_of_slogans = int(self.liczbaSloganow.text())
        max_slogan_length = int(self.dlugosc.text())
        end_after_pipe_character = self.dokladnaDlugosc.isChecked()
        diversity = 0.3

        thread = Thread(target = self.generate, args = (number_of_slogans, all_slogans_as_text, maxlen, chars, char_indices, indices_char, max_slogan_length, diversity, end_after_pipe_character, self.progressBar))
        thread.start()
    
    def generate(number_of_slogans, all_slogans_as_text, maxlen, chars, char_indices, indices_char, max_slogan_length, diversity, end_after_pipe_character, progressBar):
        for _ in range(number_of_slogans):
            print(generate_text(all_slogans_as_text, maxlen, chars, char_indices, indices_char, max_slogan_length, diversity, end_after_pipe_character))
        progressBar.hide()
        




app = QtWidgets.QApplication(sys.argv)
obj = GUI()
obj.show()

sys.exit(app.exec_())