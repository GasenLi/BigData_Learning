import math
import re

# 有特殊字符的话直接在其中添加
words = open("words-by-frequency.txt").read().split()
word_cost = dict((k, math.log((i + 1) * math.log(len(words)))) for i, k in enumerate(words))
max_word = max(len(x) for x in words)


def cut_list(word):
    # 对于一个长单词，可能是由多个单词构成，该方法就是对长单词进行判断，
    # 然后进行拆分
    def infer_spaces(string):
        '''
            Uses dynamic programming to infer the location of spaces in a string without spaces.
            .使用动态编程来推断不带空格的字符串中空格的位置。
        '''

        def best_match(i, cost, s):
            candidates = enumerate(reversed(cost[max(0, i - max_word):i]))
            return min((c + word_cost.get(s[i - k - 1:i], 9e999), k + 1) for k, c in candidates)

        cost = [0]
        for i in range(1, len(string) + 1):
            c, k = best_match(i, cost, string)
            cost.append(c)
        out = []
        i = len(string)
        while i > 0:
            c, k = best_match(i, cost, string)
            assert c == cost[i]
            out.append(string[i - k:i])
            i -= k
        return ' '.join(reversed(out))

    split = word.split()
    for index, word in enumerate(split):
        if len(word) >= 12:
            split[index] = infer_spaces(word)
    return ' '.join(split)


# 数字字母分割
def separate_num_and_letter(word):
    return ' '.join(re.findall(r'[0-9]+|[a-z]+', word))


# 单词拼写错误检查
def word_correction(dict, word):
    if dict.check(word) is False:
        suggest = dict.suggest(word)
        if len(dict.suggest(word)) == 1:
            return suggest[0].lower()
        else:
            return word.lower()
    else:
        return word.lower()


# 判断一个字符串是否是数字
def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(string)
        return True
    except (TypeError, ValueError):
        pass

    return False
