from functools import cmp_to_key

import numpy
import numpy as np
from itertools import permutations
from collections import Counter


def standart_order(m):
    numb = pow(2, m)
    result = []
    for i in range(numb):
        binNumb = bin(i)
        binNumb = binNumb[:1:-1]
        if (len(str(binNumb)) < m):
            part = str(binNumb)
            while (len(part) != m):
                part = part + "0"
            result.append(part)
        else:
            result.append(str(binNumb))
    return result


def f(i_array, x_array):
    result = 1
    if (len(i_array) == 0):
        return result
    else:
        for i in i_array:
            result = result * (x_array[int(i)] + 1)
        if (result > 1):
            result = 0
        return result


def v(i_array, m):
    n = pow(2, m)
    v = []
    numbers = standart_order(m)
    for number in numbers:
        x_array = []
        for digit in number:
            x_array.append(int(digit))
        v.append(f(i_array, x_array))
    return v


def duplicate(input):
    result = list()
    input = map(tuple, input)
    freqDict = Counter(input)
    for (row, freq) in freqDict.items():
        if freq > 1:
            result.append(row)
    return result


def gen_str_from_numb(r):
    str_num = ""
    for r_i in range(r + 1):
        str_num = str_num + str(r_i)
    return str_num


def gen_i_array(r, m):
    result = list()
    str_num = gen_str_from_numb(r)
    for i in permutations(str_num, m):
        result.append(sorted(i))
    if (m > 1):
        return duplicate(result)
    else:
        return result


def gen_i_arrays(r):
    result = []
    for r_i in range(r + 1):
        result = result + gen_i_array(r, r_i)
    return result


def find_last_one_index(v):
    counter = 0
    for v_i in reversed(v):
        counter = counter + 1
        if (v_i == 1):
            return counter - 1


def sort_v_for_same_size(v):
    for i, j in list(permutations(gen_str_from_numb(len(v) - 1), 2)):
        i = int(i)
        j = int(j)
        if (find_last_one_index(v[i]) > find_last_one_index(v[j])):
            v[i], v[j] = v[j], v[i]
    return v


def RM_for_size(r, m, size):
    result = []
    for i in gen_i_arrays(r):
        if (len(i) == size):
            result.append(v(i, m))
    return result


def RM(r, m):
    result = []
    if (r >= m):
        return "incorrect values"
    for r_i in range(r + 1):
        result = result + sort_v_for_same_size(RM_for_size(r, m, r_i))
    return result


def generate_inverse_nums_len_m(m):
    max_num = 2 ** m
    nums = []
    for i in range(max_num):
        cur = format(i, 'b')[::-1]
        while len(cur) < m:
            cur = cur + '0'
        cur_num_arr = []
        for j in range(len(cur)):
            cur_num_arr.append(int(cur[j]))
        nums.append(cur_num_arr)
    return nums


def generate_bitmasks(r, m):
    max_num = 2 ** m

    masks = []
    for i in range(r + 1):
        masks.append([])

    for i in range(max_num):
        mask = format(i, 'b')
        while len(mask) < m:
            mask = '0' + mask
        mask = list(map(int, mask))
        true_count = 0
        for j in range(len(mask)):
            true_count = true_count + int(mask[j])
        if true_count <= r:
            masks[true_count].append(mask)
    return masks


def mask_to_I(mask):
    I = []
    for i in range(len(mask)):
        if mask[i] == 1:
            I.append(i)
    return I


def inverse_binary_num(num):
    inversed_num = []
    for i in range(len(num)):
        if num[i] == 1:
            inversed_num.append(0)
        else:
            inversed_num.append(1)
    return inversed_num


def base_function(I, word):
    for i in range(len(I)):
        if word[i] == 1:
            return 0
    return 1

def conv_func(i, x_arr):
    arr = []
    curr = list(map(str, list(map(lambda x: base_function(i, x), x_arr))))
    arr.append([''.join(curr[::-1]), i])
    return arr


def compare(x, y):
    return int(y[0] < x[0]) - int(x[0] < y[0])


def I_ordered_by_RM(r, m):
    x_arr = generate_inverse_nums_len_m(m)
    i_arr = list(map(lambda x: list(map(lambda y: mask_to_I(y), x)), generate_bitmasks(r, m)))

    res = []
    for x in i_arr:
        curr = list(map(lambda i: conv_func(i, x_arr), x))
        curr_1 = sorted(np.copy(curr), key=cmp_to_key(compare))
        curr_2 = list(map(lambda l: ''.join(map(str,l[1])), curr_1))
        res.append(curr_2)

    return res

def decoding(word, r, m):
    inverse_words_len_m = generate_inverse_nums_len_m(m)
    masks = generate_bitmasks(r, m)

    word_copy = numpy.copy(word)

    res = []
    len_of_answer = 0

    for i in range(r, 0, -1):
        arr = []
        masks_i_len = len(masks[i])

        for j in range(masks_i_len):
            len_of_answer = len_of_answer + 1
            cur_mask = masks[i][j]
            I = mask_to_I(cur_mask)
            inversed_mask = inverse_binary_num(cur_mask)
            I_c = mask_to_I(inversed_mask)

            H_I = []
            for k in range(len(inverse_words_len_m)):
                if base_function(I, inverse_words_len_m[k]) == 1:
                    H_I.append(inverse_words_len_m[k])

            v_t = []
            for k in range(len(H_I)):
                v_t.append([])
                for l in range(len(inverse_words_len_m)):
                    H_I_k = H_I[k]
                    xor = []
                    for p in range(len(H_I_k)):
                        xor.append((H_I_k[p] + inverse_words_len_m[l][p]) % 2)
                    v_t[k].append(base_function(I_c, xor))

            word_multipy_v = []
            for k in range(len(v_t)):
                for l in range(len(word_copy)):
                    count_of_one = 0
                    if word_copy[l] * v_t[k][l] == 1:
                        count_of_one = count_of_one + 1
                    word_multipy_v.append(count_of_one % 2)

            m_decision = 0
            count_of_one = 0
            for k in range(len(word_multipy_v)):
                if word_multipy_v[k] == 1:
                    count_of_one = count_of_one + 1
            if count_of_one * 2 > len(word_multipy_v):
                m_decision = 1
            else:
                m_decision = 0

            if m_decision == 1:
                new_arr_elem = []
                for k in range(len(inverse_words_len_m)):
                    new_arr_elem.append(base_function(I, inverse_words_len_m[k]))
                arr.append(new_arr_elem)
                res.append(''.join(map(str, I)))

        for j in range(len(arr)):
            for k in range(len(word_copy)):
                word_copy[k] = (word_copy[k] + arr[j][k]) % 2

    answer = []
    for i in range(len_of_answer):
        answer.append(0)

    ordered_I = I_ordered_by_RM(r, m)

    for i in range(len(res)):
        index = ordered_I.index(res[i])
        answer[index] = 1

    return answer


if __name__ == '__main__':
    word = [0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    r = 2
    m = 4
    G = RM(r, m)
    print(G)
    u = decoding(word, r, m)
    v = np.matmul(u, G)
    print('word = ', word, ' u = ', u, ' v = ', v)
