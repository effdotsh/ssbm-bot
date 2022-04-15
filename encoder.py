import numpy as np


def generate_nested_arr(layers: list, number):
    if len(layers) == 1:
        return np.array(list(range(number, number + layers[0]))), number + layers[0]
    arr = []
    for _ in range(layers[0]):
        a, number = generate_nested_arr(layers[1:], number)
        arr.append(a)
    return np.array(arr), number


def encode_to_number(path, maxes):
    assert len(path) == len(maxes)
    arr, _ = generate_nested_arr(maxes, 0)
    ans = arr
    for i in path:
        ans = ans[i]
    return ans


def decode_from_number(key, maxes):
    arr, _ = generate_nested_arr(maxes, 0)
    return np.array(np.where(arr == key)).flatten()
