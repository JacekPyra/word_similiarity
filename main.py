from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

word2vec = KeyedVectors.load("word2vec_100_3_polish.bin")

secret = "domek"
print(word2vec.similar_by_word(secret))

secret_vector = word2vec.get_vector(secret)


index = word2vec.key_to_index[secret]
print(index)

vectors = word2vec.vectors

# Perform PCA
pca = PCA(n_components=4)
pca_result = pca.fit_transform(vectors)


# # Plot
# fig, ax = plt.subplots(figsize=(10, 10))
# ax.scatter(pca_result[:,0], pca_result[:,1])
# for i, word in enumerate(word2vec.index_to_key[70:200]):
#     ax.annotate(word, xy=(pca_result[i,0], pca_result[i,1]))
#     print(pca_result)
# plt.show()

print(pca_result[index])

while True:
    guess = input("Zgadnij słowo: ")
    try:
        index_guess = word2vec.key_to_index[guess]
        print(pca_result[index_guess])

        print(word2vec.distance(secret, guess))
    except KeyError as e:
        print("Nie znam takiego słowa")