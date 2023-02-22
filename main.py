import torch
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F
from os.path import exists
plt.interactive(True)

LETTERS = '.abcdefghijklmnopqrstuvwxyz'
L_TO_IDX = {letter: index for index, letter in enumerate(LETTERS)}
IDX_TO_L = {index: letter for index, letter in enumerate(LETTERS)}


def get_words_from_file():
    if exists('words.txt'):
        with open("words.txt", "r") as f:
            return json.load(f)
    else:
        words = open('names.txt', 'r').read().splitlines()
        with open("words.txt", "w") as f:
            json.dump(words, f)
        return words


def count_bigrams(words):
    if exists('bigram_arr.pt') and exists('bigram_graph.png'):
        return torch.load('bigram_arr.pt')
    bigram_counts = torch.zeros((27, 27), dtype=torch.int32)

    for word in words:
        chars = ["."] + list(word) + ["."]
        for char1, char2 in zip(chars, chars[1:]):
            index1 = L_TO_IDX[char1]
            index2 = L_TO_IDX[char2]
            bigram_counts[index1, index2] += 1

    generate_graph(bigram_counts)
    torch.save(bigram_counts, 'bigram_arr.pt')
    return bigram_counts


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


def sample_from_counts(probabilities, num_predictions):
    predictions = []
    for i in range(num_predictions):
        index = 0
        output = []
        while True:
            probability = probabilities[index]
            index = torch.multinomial(probability, num_samples=1, replacement=True).item()
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
def calc_likelihood_score(probabilities, words):
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
    return average_negative_log_likelihood


def neural_net(num_iterations, inputs, outputs):
    # randomly initialize 27 neurons' weights. each neuron receives 27 inputs
    W = torch.randn((27, 27), requires_grad=True)
    num_inputs = inputs.nelement()
    learning_rate = 50
    #####  Gradient Descent #####
    for k in range(num_iterations):
        ### Forward Pass ###
        inputs_enc = F.one_hot(inputs, num_classes=27).float()  # encoding the inputs (one_hot encoding) before passing into neural net
        logits = inputs_enc @ W  # log-counts

        # Soft-Max
        counts = logits.exp()  # counts
        probabilities = counts / counts.sum(1, keepdim=True)  # normalized counts / probability for next character

        loss = -probabilities[torch.arange(num_inputs), outputs].log().mean() + 0.001*(W**2).mean()
        # negative log likelihood + w gravity model smoothing

        ### Backward Pass ###
        W.grad = None  # Set gradient to 0
        loss.backward()
        # .backward() goes back and gets W from the loss, and stores in W.grad the effect that the current probability has ...
        # ... on the loss. We then want to update these gradients to try and reduce loss

        ### Update ###
        W.data += -learning_rate * W.grad
        print(f"{num_iterations - k} iterations remaining")

    print(f"Training finished with {loss} loss")
    return W


### Sampling from Model
def sample_from_nn(W, num_samples):
    for i in range(num_samples):
        out = []
        idx = 0
        while True:
            xenc = F.one_hot(torch.tensor([idx]), num_classes=27).float()
            logits = xenc @ W
            counts = logits.exp()
            p = counts / counts.sum(1, keepdim=True)

            idx = torch.multinomial(p, num_samples=1, replacement=True).item()
            out.append(IDX_TO_L[idx])

            if idx == 0:
                break
        print(''.join(out))


def main_0():
    words = get_words_from_file()
    # Create training set (in, out)
    inputs, outputs = [], []
    for word in words:
        chars = ["."] + list(word) + ["."]
        for char1, char2 in zip(chars, chars[1:]):
            index1 = L_TO_IDX[char1]
            index2 = L_TO_IDX[char2]
            inputs.append(index1)
            outputs.append(index2)

    walter = neural_net(121, torch.tensor(inputs), torch.tensor(outputs))
    sample_from_nn(walter, 100)


def main_1():
    words = get_words_from_file()
    bigrams = count_bigrams(words)

    # (bigrams+1) adds one to the count of each bigram
    # it is there to smooth the model, so there is always more than a 0% chance of a bigram existing
    # The array is normalized across the columns, returning a column vector 'probabilities'
    # probabilities.sum() -> 27
    # Each row adds up to one, and there are 27 rows
    # If keepdim was false, the vector would get morphed into a row vector (1 by 27 array) when broadcasting ...
    # ... division operation

    probabilities = (bigrams+1).float()
    probabilities /= probabilities.sum(1, keepdim=True)
    for sample in sample_from_counts(probabilities, 25):
        print(sample)
    print(f"Loss: {calc_likelihood_score(probabilities, words)}")


main_0()
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
main_1()
