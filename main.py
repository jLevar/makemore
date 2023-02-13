import torch
import json
import matplotlib.pyplot as plt
from os.path import exists

plt.interactive(True)

letters = '.abcdefghijklmnopqrstuvwxyz'
L_TO_IDX = {letter: index for index, letter in enumerate(letters)}
IDX_TO_L = {index: letter for index, letter in enumerate(letters)}


def generate_data():
    if exists('bigram_arr.pt') and exists('words.txt') and exists('bigram_graph.png'):
        with open("words.txt", "r") as f:
            words = json.load(f)
        return torch.load('bigram_arr.pt'), words
    # Generate data
    words = open('names.txt', 'r').read().splitlines()
    bigram_arr = torch.zeros((27, 27), dtype=torch.int32)
    for word in words:
        chars = ["."] + list(word) + ["."]
        for char1, char2 in zip(chars, chars[1:]):
            index1 = L_TO_IDX[char1]
            index2 = L_TO_IDX[char2]
            bigram_arr[index1, index2] += 1
    generate_graph(bigram_arr)
    # Save to file and return
    torch.save(bigram_arr, 'bigram_arr.pt')
    with open("words.txt", "w") as f:
        json.dump(words, f)
    return bigram_arr, words


def generate_graph(bigram_arr):
    plt.figure(figsize=(16, 16))
    plt.imshow(bigram_arr, cmap='Blues')
    for i in range(27):
        for j in range(27):
            # Label
            label = IDX_TO_L[i] + IDX_TO_L[j]
            plt.text(j, i, label, ha="center", va="bottom", color='gray')
            # Number
            plt.text(j, i, bigram_arr[i, j].item(), ha="center", va="top", color='gray')
    plt.axis('off')
    plt.savefig('bigram_graph.png')


def generate_predictions(num_predictions, seed=2147483647):
    gen = torch.Generator().manual_seed(seed)
    predictions = []
    for i in range(num_predictions):
        index = 0
        output = []
        while True:
            probability = probabilities[index]
            index = torch.multinomial(probability, num_samples=1, replacement=True, generator=gen).item()
            output.append(IDX_TO_L[index])
            if index == 0:
                break
        predictions.append(''.join(output))
    return predictions


# Likelihood is the multiplicative sum of the probabilities of all the bigrams in a word
# The likelihood percentage can get crazy small, so for convenience we use log likelihood
# The log likelihood is the additive sum of all the log of the probabilities of all bigrams in a word

# GOAL: maximize likelihood of the data with respect to model parameters (statistical modeling)
# equivalent to maximizing the log likelihood (because log is monotonic)
# equivalent to minimizing the negative log likelihood
# equivalent to minimizing the average negative log likelihood
def calc_likelihood_score():
    log_likelihood = 0.0
    num_iterations = 0
    for word in words:
        chars = ["."] + list(word) + ["."]
        for char1, char2 in zip(chars, chars[1:]):
            index1 = L_TO_IDX[char1]
            index2 = L_TO_IDX[char2]
            prob = probabilities[index1, index2]
            log_prob = torch.log(prob)
            log_likelihood += log_prob
            num_iterations += 1

    negative_log_likelihood = -log_likelihood
    average_negative_log_likelihood = negative_log_likelihood / num_iterations
    return average_negative_log_likelihood.item()


bigrams, words = generate_data()

# (bigrams+1) adds one to the count of each bigram
# it is there to smooth the model, so there is always more than a 0% chance of a bigram existing
# The array is normalized across the columns, returning a column vector 'probabilities'
# probabilities.sum() -> 27
# Each row adds up to one, and there are 27 rows
# If keepdim was false, the vector would get morphed into a row vector (1 by 27 array) when broadcasting division operation
probabilities = (bigrams+1).float()
probabilities /= probabilities.sum(1, keepdim=True)
for prediction in generate_predictions(10):
    print(prediction)

print(calc_likelihood_score())


