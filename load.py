import json


def load_data(filename):
    D = []
    # index = 0
    with open(filename, 'r', encoding='utf-8') as f:
        for l in f:

            # if index > 1000:
            #    break
            # index += 1
            l = json.loads(l)
            d = {'text': l['text'], 'spo_list': []}
            for spo in l['spo_list']:
                for k, v in spo['object'].items():
                    d['spo_list'].append(
                        (spo['subject'], spo['predicate'] + '_' + k, v)
                    )
            D.append(d)
    return D


def load_schema():
    with open('data/duie_schema.json', encoding='utf-8') as f:
        id2predicate, predicate2id, n = {}, {}, 0
        predicate2type = {}
        for l in f:

            l = json.loads(l)
            predicate2type[l['predicate']] = (l['subject_type'], l['object_type'])
            for k, _ in sorted(l['object_type'].items()):
                key = l['predicate'] + '_' + k
                id2predicate[n] = key
                predicate2id[key] = n
                n += 1
        return id2predicate, predicate2id, n


def search(pattern, sequence):
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i: i + n] == pattern:
            return i
    return -1


if __name__ == "__main__":
    load_data("data/duie_train.json")

