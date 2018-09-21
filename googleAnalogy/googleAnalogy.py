import math
import os
import sys
import numpy

from gensim.models import KeyedVectors



def read_word_vectors(filename):
    word_vecs = {}
    file_object = open(filename, 'r')

    for line_num, line in enumerate(file_object):
        line = line.strip()
        word = line.split()[0]
        word_vecs[word] = numpy.zeros(len(line.split()) - 1, dtype=float)
        for index, vec_val in enumerate(line.split()[1:]):
            word_vecs[word][index] = float(vec_val)
            # normalize weight vector
        word_vecs[word] /= math.sqrt((word_vecs[word] ** 2).sum() + 1e-6)

    sys.stderr.write("Vectors from: " + filename + " \n")
    return word_vecs


if __name__ == '__main__':
    #word_vec_file = "/home/nishanthini/FYP/Evaluations/Ploysemy_eval_final/BenchMarkMyWork/googleAnalogy/" \
    #                "model/vectors_it3.txt"
    word_vec_file = sys.argv[1]
    #google_analogy_dir = "/home/nishanthini/FYP/Evaluations/Ploysemy_eval_final/BenchMarkMyWork" \
    #                     "/googleAnalogy/processData/"
    google_analogy_file = sys.argv[2]

    word_vecs = read_word_vectors(word_vec_file)
    english_model = KeyedVectors.load_word2vec_format(word_vec_file, binary=False)

    print('_________________________________________________________________________________')

    print('_________________________________________________________________________________')

    not_found, total_size = (0, 0)
    counti = 0
    total = 0
    found = 0

    for i, filename in enumerate(os.listdir(google_analogy_dir)):
        print(filename)
        for line in open(os.path.join(google_analogy_dir, filename), 'r'):
            total += 1
            c = 0
            line = line.strip().lower()
            word1, word2, word3, word4 = line.split(" ")
            vec_word1 = word1 + "_"
            vec_word2 = word2 + "_"
            vec_word3 = word3 + "_"
            word_vec1 = {}
            word_vec2 = {}
            word_vec3 = {}
            for key in word_vecs:
                if key.startswith(vec_word1) and "GRAM" not in key:
                    # print (key)
                    c += 1
                    word_vec1[key] = word_vecs[key]
                elif key.startswith(vec_word2) and "GRAM" not in key:
                    c += 1
                    # print(key)
                    word_vec2[key] = word_vecs[key]
                elif key.startswith(vec_word3) and "GRAM" not in key:
                    c += 1
                    # print(key)
                    word_vec3[key] = word_vecs[key]
                else:
                    continue
            # print(c)
            if c >= 3:
                # i = list(word_vec1.keys())[0]
                # j = list(word_vec2.keys())[0]
                # k = list(word_vec3.keys())[0]
                # print(i)
                # print(j)
                # print(k)
                # print("**************************************************************")
                wordFound = False
                for i in word_vec1:
                    for j in word_vec2:
                        for k in word_vec3:

                            outResult = english_model.most_similar(positive=[j, k],
                                                           negative=[i], topn=10)
                            # print(outResult)
                            for vec_word in outResult:
                                word = vec_word[0].split("_")[0].split("\'")[0]

                                # print(word + "_____" + word4)
                                if word == word4:
                                    if wordFound:
                                        break
                                    elif not wordFound:
                                        found += 1
                                        # print(str(found) + "   of " + str(total) + "  " + word2 + " - " + word1
                                        #       + " + " + word3 + " = " + word)
                                        wordFound = True
                                    break
                print (total)


        print(total)
        print(found)
        # filename.close()



