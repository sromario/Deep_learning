import os
import numpy as np
import matplotlib.pyplot as plt
from datagen import image_size
from keras._tf_keras.keras.preprocessing.image import load_img
from keras._tf_keras.keras.preprocessing import image
from keras import models

"""
    Agora realizar a previsão em cima de um conjunto de teste.

    01: ler as imnagens da pasta de test.
    02: criar contadores, definir plote da figura, chamar caminho do modelo treinado.
    03: percorre a pasta carregando as imagens, convertendo para array numpy, aplica dimensão extra(no eixo 0),e empinha os batch.
    04: realiza a previsão e defini cat ou dog
    05: plota imagens com resultados
"""

test_dir = 'test' 
files = [os.path.join(dp, f)
            for dp, dn, filenames in os.walk(test_dir)
            for f in filenames
            if os.path.splitext(f)[1] == '.jpeg']
print(files)

cats = 0
dogs = 0
index = 0
plt.figure(figsize=(12,12))

model = models.load_model("binario\model.h5")

for fn in files:
    img = load_img(fn, target_size=image_size)
    y = image.img_to_array(img)
    x = np.expand_dims(y, axis=0)
    images = np.vstack([x])

    classes = model.predict(images, batch_size=10) # previsão
    print("valor previsto modelo:", classes[0])

    category = "dog"
    if classes[0] > 0.5:
        print(fn + " dog")
        dogs += 1
    else:
        category = "cat"
        print(fn + " cat")
        cats += 1

    #plot
    plt.subplot(3, 3, index + 1)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(hspace=0.5)
    plt.xlabel(fn + '(' + "{}".format(category) + ')')
    index += 1

# RESULTADOS
print('TOTAL CATS: ', cats)
print('TOTAL DOGS: ', dogs)
plt.tight_layout()
plt.show()  

