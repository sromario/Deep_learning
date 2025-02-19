import keras
from keras import layers, models, optimizers
from datagen import train_generator, validation_generator

"""
03 camdas convolucionais Conv2D.
03 camdadas de complemento MaxPooling2D.
01 camada Flatten.
01 camada Densa.

Conv2D: Será responsavel por extrair padrões visuais, onde é definido quantidade de filtros aplicados(16) e tamanho deles(3x3)
MaxPooling: diminuir a dimensão da imagens, priorizado os melhores valores de cada regiao(2x2)
Flatten: recebe a ultima saida de Conv2D e MaxPooling2D e transforma em um vetor de uma 1D(uma dimensão)
Densa: recebe todos valores do vetor 1d, passa para os neuronios como forma de aprendizado

Treina o modelo, model.fit passando os hiperparametros.
salva o modelo em h5.
"""
def main():
    model = keras.models.Sequential([

    layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2,2),


    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),


    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid'), # saída de um neuronio e sigmoid para classificação binaria
])


    # ver camadas e saidas
    model.summary()

    # prepara para treinamento
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(learning_rate=0.001), 
                  metrics=['accuracy'])

    
    history = model.fit(train_generator,
                        validation_data=validation_generator, 
                        steps_per_epoch=100, 
                        epochs=15, 
                        validation_steps=50, 
                        verbose=2)

    print(history)
    model.save("model.h5")

if __name__ == '__main__':
    main()

