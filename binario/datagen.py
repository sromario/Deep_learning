from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from data import train_dir, validation_dir


"""
pré processamento das imagens, 

Normaliza os pixels das imagens com `ImageDataGenerator`, escalando os valores de 0 a 1.
Cria um gerador de imagens (`flow_from_directory`), definindo:
    - O tamanho das imagens (`target_size`).
    - O tipo de classificação (`class_mode`).
    - O tamanho dos lotes (`batch_size`).

"""
image_width = 150
image_height = 150
image_size = (image_width, image_height)


train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)


train_generator =  train_datagen.flow_from_directory(
    train_dir,
    batch_size=20, 
    class_mode='binary', 
    target_size=(image_width,image_height)
)


validation_generator =  test_datagen.flow_from_directory(
    validation_dir, 
    batch_size=20, 
    class_mode='binary', 
    target_size=(image_width,image_height)
)

