import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg


def load_images(folder: str, categories, filenames):
    """
    carrega imagens das pastas e classifica como cat ou dog(0 ou 1).
    
    args:
        folder: caminho para as pastas.
        categories: lista para armazenar rotulos(0 = cat, 1 = dog).
        filenames: lista para armazenar o nome dos arquivos.

    returns: 
        os resultados são armazenados nas listas.
    """
    filenames2 = os.listdir(folder)
    for filename in filenames2:
        category = filename.split('.')[0]
        if category == 'dog':
            categories.append(1)
        else:
            categories.append(0)
        filenames.append(filename)


def plot(train_cats_dir, train_dogs_dir):
    """
    Analisa e visualiza os dados do dataset de gatos e cachorros.

    Cria um DataFrame com os nomes dos arquivos e categorias.
    Plota um gráfico de barras para comparar a quantidade de gatos e cachorros.
    Exibe uma grade de imagens de gatos e cachorros para visualização.

    Args:
        train_cats_dir (str): Caminho da pasta de treino de gatos.
        train_dogs_dir (str): Caminho da pasta de treino de cachorros.
    """

    train_cat_fnames = os.listdir(train_cats_dir)
    train_dog_fnames = os.listdir(train_dogs_dir)

    # Criar listas de categorias e filenames
    categories = [0] * len(train_cat_fnames) + [1] * len(train_dog_fnames)
    filenames = train_cat_fnames + train_dog_fnames

    df = pd.DataFrame({'filename': filenames, 'category': categories})
    print(df.head())  # Exibir as primeiras linhas

    #  Plotar gráfico de barras
    plt.figure(figsize=(6, 4))
    plt.bar(['Cats (0)'], df['category'].value_counts()[0], color='r', label='Cats')
    plt.bar(['Dogs (1)'], df['category'].value_counts()[1], color='b', label='Dogs')
    plt.legend()
    plt.title("Distribuição de imagens no dataset")
    plt.show()

    #  Exibir uma grade de imagens de exemplo (8 gatos + 8 cachorros)
    nrows, ncols = 4, 4
    pic_index = 10  # Índice para seleção das imagens

    fig = plt.figure(figsize=(ncols * 4, nrows * 4))
    next_cat_pix = [os.path.join(train_cats_dir, fname) for fname in train_cat_fnames[pic_index - 8:pic_index]]
    next_dog_pix = [os.path.join(train_dogs_dir, fname) for fname in train_dog_fnames[pic_index - 8:pic_index]]

    for i, img_path in enumerate(next_cat_pix + next_dog_pix):
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off')
        img = mpimg.imread(img_path)
        plt.imshow(img)

    plt.show()


base_dir = "C:/Users/romario.santos/Documents/Deep_learning/binario/cats_and_dogs"
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")

def main():
    """
        configura os diretorios, de treino e validação de gatos e cachorros.
        lista os arquivos de imagens e exibe quantidade total de cada classe.
        plotar grafico e exibir imagens dod ataset

    """

    train_cats_dir = os.path.join(train_dir, "cats")
    train_dogs_dir = os.path.join(train_dir, "dogs")


    validation_cats_dir = os.path.join(validation_dir, "cats")
    validation_dogs_dir = os.path.join(validation_dir, "dogs")


    train_cat_fnames = os.listdir(train_cats_dir)
    train_dog_fnames = os.listdir(train_dogs_dir)


    print(train_cat_fnames[:10])
    print(train_dog_fnames[:10])

    print('total imagens cat treinamento :', len(os.listdir(train_cats_dir)))
    print('total imagens dog treinamento :', len(os.listdir(train_dogs_dir)))
    print('total imagens cat validação :', len(os.listdir(validation_cats_dir)))
    print('total imagens dog validação :', len(os.listdir(validation_dogs_dir)))


    categories = []
    filenames = []

    load_images(train_dogs_dir, categories, filenames)
    load_images(train_cats_dir, categories, filenames)


    plot(train_cats_dir, train_dogs_dir)

if __name__ == '__main__': 
    main()


