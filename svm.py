import numpy as np
import matplotlib.image as mpimg
from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics

#FUNCTIA CARE IMI VA NORMALIZA DATELE
def normalizare(data):
    scaler = preprocessing.StandardScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    return scaled_data

#CITIREA SI MEMORAREA DATELOR DE ANTRENARE
train_data = []
train_labels = []
with open("train.txt", "r") as in_file:
    while True:
        text = in_file.readline()
        if text == '':
            break
        img = mpimg.imread(f"./train/{text.split(',')[0]}")
        train_data.append(img)
        train_labels.append(int(text.split(',')[1]))

#TRANSFORMAREA LISTELOR IN NP.ARRAY
train_data = np.array(train_data)
train_labels = np.array(train_labels)

#REDIMENSIONAREA DE LA 3D LA 2D A ARRAY-ULUI
train_data_reshaped = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2])

#CITIREA SI MEMORAREA DATELOR DE TEST
test_data = []
test_images_names = []
with open("test.txt", "r") as in_file:
    while True:
        text = in_file.readline()
        if text == '':
            break
        img = mpimg.imread(f"./test/{text[:len(text)-1]}")
        test_data.append(img)
        test_images_names.append(text[:len(text)-1])

#TRANSFORMAREA LISTEI CU DATELE DE TEST IN NP.ARRAY SI REDIMENSIONAREA LUI
test_data = np.array(test_data)
test_data_reshaped = test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2])

#CITIREA SI MEMORAREA DATELOR DE VALIDARE
validation_data = []
validation_labels = []
with open("validation.txt", "r") as in_file:
    while True:
        text = in_file.readline()
        if text == '':
            break
        img = mpimg.imread(f"./validation/{text.split(',')[0]}")
        validation_data.append(img)
        validation_labels.append(int(text.split(',')[1]))

#TRANSFORMAREA LISTELOR CU DATELE SI LABEL-URILE DE VALIDARE IN NP.ARRAY SI REDIMENSIONAREA CELUI CU IMAGINILE DE VALIDARE DIN 3D IN 2D 
validation_data = np.array(validation_data)
validation_data_reshaped = validation_data.reshape(validation_data.shape[0], validation_data.shape[1] * validation_data.shape[2])
validation_labels = np.array(validation_labels)

#NORMALIZAREA TUTUROR DATELOR
norm_train_data = normalizare(train_data_reshaped)
norm_test_data = normalizare(test_data_reshaped)
norm_validation_data = normalizare(validation_data_reshaped)

#DEFINIREA MODELULUI 
clasif = svm.SVC(C=3, kernel = 'rbf')

#ANTRENAREA MODELULUI
clasif.fit(norm_train_data, train_labels)

#PREDICTIILE CLASIFICATORULUI PENTRU DATELE DE TEST
test_predictions = clasif.predict(norm_test_data)

#SCRIEREA PREDICTIILOR OBTINUTE
with open("predictions7.txt", "w") as output_file:
    output_file.write("id,label\n")
    for i in range(len(test_predictions)):
        output_file.write(test_images_names[i] + ',' + str(int(test_predictions[i])) + "\n")

#ACURATETEA PENTRU DATELE DE VALIDARE
print(clasif.score(norm_validation_data, validation_labels))

#PREDICTIILE PENTRU DATELE DE VALIDARE, NECESARE PENTRU MATRICEA DE CONFUZIE
validation_predictions = clasif.predict(norm_validation_data)
matrix = metrics.confusion_matrix(validation_labels, validation_predictions)

with open("confusion_matrix.txt", "w") as matrix_file:
    matrix_file.write('\n'.join(['\t'.join([str(int(cell)) for cell in row]) for row in matrix]))