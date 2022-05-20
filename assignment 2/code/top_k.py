import numpy as np
import utils


def most_similar(word, k, embeddings):
    word_embedding = embeddings[word]
    top_k = {}
    for w2, w2_embedding in embeddings.items():
        if w2 == word:
            continue
        cos_sim = cosine_similarity(word_embedding, w2_embedding)
        if len(top_k) < k:
            top_k[w2] = cos_sim
            continue
        word_to_remove_from_top_k = get_word_to_remove_from_top_k(cos_sim, top_k)
        if word_to_remove_from_top_k is None:
            continue
        del top_k[word_to_remove_from_top_k[0]]
        top_k[w2] = cos_sim
    return top_k


def cosine_similarity(u, v):
    u_norm = np.sqrt(np.dot(u, u))
    v_norm = np.sqrt(np.dot(v, v))
    return np.dot(u, v) / (u_norm * v_norm)


def get_word_to_remove_from_top_k(cos_sim, top_k):
    first_w = list(top_k.keys())[0]
    min_similarity = (first_w, top_k[first_w])
    for w, similarity in top_k.items():
        if not similarity < min_similarity[1]:
            continue
        min_similarity = (w, similarity)
    if cos_sim > min_similarity[1]:
        return min_similarity
    else:
        return None


def main():
    k = 5
    embeddings = utils.get_existing_embeddings(f'{utils.EMBEDDINGS_PATH}/wordVectors.txt',
                                               f'{utils.EMBEDDINGS_PATH}/vocab.txt')
    words_to_check = ['dog', 'england', 'john', 'explode', 'office']
    for word in words_to_check:
        most_similar_words = most_similar(word, k, embeddings)
        print(word, most_similar_words)


if __name__ == '__main__':
    main()
