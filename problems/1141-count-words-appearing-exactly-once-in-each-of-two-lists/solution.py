def count_common_unique(list1, list2):
    # list1: list of strings
    # list2: list of strings
    # return an integer
    dict1 = {}
    dict2 = {}
    for el in list1:
        if el not in dict1:
            dict1[el] = 1
        else:
            dict1[el] += 1

    for el in list2:
        if el not in dict2:
            dict2[el] = 1
        else:
            dict2[el] += 1

    inter_cnt = 0
    for el, freq in dict1.items():
        if freq == 1 and dict2.get(el) == 1:
            inter_cnt += 1
    return inter_cnt