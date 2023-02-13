import torch
import matplotlib.pyplot as plt
from os.path import exists

plt.interactive(True)

letters = '.abcdefghijklmnopqrstuvwxyz'
letter_to_arr_index = {letter: index for index, letter in enumerate(letters)}
arr_index_to_letter = {index: letter for index, letter in enumerate(letters)}


# def generate_array():
#     if exists('bigram_arr.pt'):
#         return torch.load('bigram_arr.pt')
#     words = open('names.txt', 'r').read().splitlines()
#     bigram_arr = torch.zeros((27, 27), dtype=torch.int32)
#     for word in words:
#         chars = ["."] + list(word) + ["."]
#         for char1, char2 in zip(chars, chars[1:]):
#             index1 = letter_to_arr_index[char1]
#             index2 = letter_to_arr_index[char2]
#             bigram_arr[index1, index2] += 1
#     torch.save(bigram_arr, 'bigram_arr.pt')
#     return bigram_arr



def generate_graph(bigram_arr):
    plt.figure(figsize=(16, 16))
    plt.imshow(bigram_arr, cmap='Blues')
    for i in range(27):
        for j in range(27):
            # Label
            label = arr_index_to_letter[i] + arr_index_to_letter[j]
            plt.text(j, i, label, ha="center", va="bottom", color='gray')
            # Number
            plt.text(j, i, bigram_arr[i, j].item(), ha="center", va="top", color='gray')
    plt.axis('off')
    plt.savefig('myfig.png')

# bigrams = generate_array()
# generate_graph(bigrams)

words = open('names.txt', 'r').read().splitlines()
bigrams = torch.zeros((27, 27), dtype=torch.int32)

for word in words:
    chars = ["."] + list(word) + ["."]
    for char1, char2 in zip(chars, chars[1:]):
        index1 = letter_to_arr_index[char1]
        index2 = letter_to_arr_index[char2]
        bigrams[index1, index2] += 1


gen = torch.Generator().manual_seed(2147483647)


# (bigrams+1) adds one to the count of each bigram
# it is there to smooth the model, so there is always more than a 0% chance of a bigram existing
probabilities = (bigrams+1).float()
# The array is normalized across the columns, returning a column vector 'probabilities'
# probabilities.sum() -> 27
# Each row adds up to one, and there are 27 rows
# If keepdim was false, the vector would get morphed into a row vector (1 by 27 array) when broadcasting division operation
probabilities /= probabilities.sum(1, keepdim=True)


for i in range(10):
    index = 0
    output = []
    while True:
        probability = probabilities[index]
        index = torch.multinomial(probability, num_samples=1, replacement=True, generator=gen).item()
        output.append(arr_index_to_letter[index])
        if index == 0:
            break
    print(''.join(output))

# Likelihood is the multiplicative sum of the probabilities of all the bigrams in a word
# The likelihood percentage can get crazy small, so for convenience we use log likelihood
# The log likelihood is the additive sum of all the log of the probabilities of all bigrams in a word

log_likelihood = 0.0
num_iterations = 0
for word in words:
    chars = ["."] + list(word) + ["."]
    for char1, char2 in zip(chars, chars[1:]):
        index1 = letter_to_arr_index[char1]
        index2 = letter_to_arr_index[char2]
        prob = probabilities[index1, index2]
        log_prob = torch.log(prob)
        log_likelihood += log_prob
        num_iterations += 1


negative_log_likelihood = -log_likelihood
average_negative_log_likelihood = negative_log_likelihood / num_iterations
print(negative_log_likelihood, average_negative_log_likelihood)

# GOAL: maximize likelihood of the data with respect to model parameters (statistical modeling)
# equivalent to maximizing the log likelihood (because log is monotonic)
# equivalent to minimizing the negative log likelihood
# equivalent to minimizing the average negative log likelihood







