import sys
import time

import regex
import tiktoken._educational
import collections
import random


# class KGETokenizer(tiktoken._educational.SimpleBytePairEncoding):
#     def __init__(self, deliminator: str, mergeable_ranks: dict[bytes, int]):
#         super().__init__(pat_str=deliminator, mergeable_ranks=mergeable_ranks)
#
#     @staticmethod
#     def train(training_data: list[str], vocab_size: int, deliminator: str):
#         """Generates a BPE vocabulary for KGE!"""
#         mergeable_ranks = kge_bpe_train(data=training_data, vocab_size=vocab_size,
#                                         tie_breaker="default")
#         return KGETokenizer(deliminator=deliminator, mergeable_ranks=mergeable_ranks)


def kge_bpe_train(data: list[str], vocab_size: int, tie_breaker: str = "default", visualise: str = "") -> dict[
    bytes, int]:
    """ Generates a BPE vocabulary from a KG.

    :param list data: A collection of strings used to build the vocabulary.
    :param int vocab_size: Final size of the vocabulary.
    :param str tie_breaker: {"default", "ascending_size", "descending_size"}, defaults to "default"
    :param str visualise: {"", "colour", "simple"}, defaults to ""
    :return: a dictionary mapping all tokens to their respective rank, i.e. the merge order.
    """
    # First, add tokens for each individual byte value
    prev_tim = time.time()
    if vocab_size < 2 ** 8:
        raise ValueError("vocab_size must be at least 256, so we can encode all bytes")
    ranks = {}
    for i in range(2 ** 8):
        ranks[bytes([i])] = i
    elements = []
    for element in data:
        elements += regex.findall(r"[\s:_]|[^\s:_]+", element)
    words: list[list[bytes]] = [[bytes([b]) for b in word.encode("ascii")] for word in elements]
    print("Creating a vocabulary of size %d" % vocab_size)
    # Now, use our data to figure out which merges we should make
    while len(ranks) < vocab_size:
        # Find the most common pair. This will become our next token
        if len(ranks) % (vocab_size // 10) == 0:
            print(len(ranks) / vocab_size * 10, "%")
            print(time.time() - prev_tim, "has passed")
            prev_tim = time.time()
        stats = collections.Counter()
        for piece in words:
            for pair in zip(piece[:-1], piece[1:]):
                stats[pair] += 1
        if stats:
            match tie_breaker:
                case "default":
                    most_common_pair = max(stats, key=lambda x: (stats[x], random.random()))
                case "ascending_size":
                    most_common_pair = max(stats, key=lambda x: (stats[x], -(len(x[0]) + len(x[1]))))
                case "descending_size":
                    most_common_pair = max(stats, key=lambda x: (stats[x], (len(x[0]) + len(x[1]))))
                case _:
                    raise ValueError(f"unknown tie_breaker: {tie_breaker}")
        else:
            break
        token_bytes = most_common_pair[0] + most_common_pair[1]
        token = len(ranks)
        # Add the new token!
        ranks[token_bytes] = token

        # Now merge that most common pair in all the words. That is, update our training data
        # to reflect our decision to make that pair into a new token.
        new_words = []
        for word in words:
            new_word = []
            i = 0
            while i < len(word) - 1:
                if (word[i], word[i + 1]) == most_common_pair:
                    # We found our pair! Merge it
                    new_word.append(token_bytes)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            if i == len(word) - 1:
                new_word.append(word[i])
            new_words.append(new_word)
        words = new_words

    # See the intermediate merges play out!
    if visualise:
        if visualise in ["colour", "color"]:
            print("Now the first fifty words in our training data look like:")
            tiktoken._educational.visualise_tokens([token for word in words[:50] for token in word])
        elif visualise == "simple":
            print("Now the first twenty words in our training data look like:")
            for word in words[:20]:
                print(word)
    return ranks


def faster_bpe(data: list[str], vocab_size: int, tie_breaker: str = "default", visualise: str = "") -> dict[bytes, int]:
    """ Generates a BPE vocabulary from a KG, noticeably faster when there is repeated words.

    :param list data: A collection of strings used to build the vocabulary.
    :param int vocab_size: Final size of the vocabulary.
    :param str tie_breaker: {"default", "ascending_size", "descending_size"}, defaults to "default"
    :param str visualise: {"", "colour", "simple"}, defaults to ""
    :return: a dictionary mapping all tokens to their respective rank, i.e. the merge order.
    """
    # First, add tokens for each individual byte value
    prev_tim = time.time()
    if vocab_size < 2 ** 8:
        raise ValueError("vocab_size must be at least 256, so we can encode all bytes")
    ranks = {}
    for i in range(2 ** 8):
        ranks[bytes([i])] = i
    elements = collections.Counter()
    total_word_count = 0
    for element in data:
        for word in regex.findall(r"[\s:_]|[^\s:_]+", element):
            elements[word] += 1
            total_word_count += 1
    words: list[(list[bytes], int)] = [([bytes([b]) for b in word.encode("ascii")], count) for word, count in elements.items()]
    print("Total number of words: %d, number of unique words: %d"%(total_word_count, len(words)))
    print("Creating a vocabulary of size %d" % vocab_size)
    # Now, use our data to figure out which merges we should make
    while len(ranks) < vocab_size:
        # Find the most common pair. This will become our next token
        if len(ranks) % (vocab_size // 10) == 0:
            print(len(ranks) / vocab_size * 10, "%")
            print(time.time() - prev_tim, "has passed")
            prev_tim = time.time()
        stats = collections.Counter()
        for piece, count in words:
            for pair in zip(piece[:-1], piece[1:]):
                stats[pair] += count
        if stats:
            match tie_breaker:
                case "default":
                    most_common_pair = max(stats, key=lambda x: (stats[x], random.random()))
                case "ascending_size":
                    most_common_pair = max(stats, key=lambda x: (stats[x], -(len(x[0]) + len(x[1]))))
                case "descending_size":
                    most_common_pair = max(stats, key=lambda x: (stats[x], (len(x[0]) + len(x[1]))))
                case _:
                    raise ValueError(f"unknown tie_breaker: {tie_breaker}")
        else:
            break
        token_bytes = most_common_pair[0] + most_common_pair[1]
        token = len(ranks)
        # Add the new token!
        ranks[token_bytes] = token

        # Now merge that most common pair in all the words. That is, update our training data
        # to reflect our decision to make that pair into a new token.
        new_words = []
        for word, count in words:
            new_word = ([], count)
            i = 0
            while i < len(word) - 1:
                if (word[i], word[i + 1]) == most_common_pair:
                    # We found our pair! Merge it
                    new_word[0].append(token_bytes)
                    i += 2
                else:
                    new_word[0].append(word[i])
                    i += 1
            if i == len(word) - 1:
                new_word[0].append(word[i])
            new_words.append(new_word)
        words = new_words

    # See the intermediate merges play out!
    if visualise:
        if visualise in ["colour", "color"]:
            print("Now the first fifty words in our training data look like:")
            tiktoken._educational.visualise_tokens([token for word, count in words[:50] for token in word])
        elif visualise == "simple":
            print("Now the first twenty words in our training data look like:")
            for word in words[:20]:
                print(word)
    return ranks
