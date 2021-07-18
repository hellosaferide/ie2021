all_list = []
lists = []
vis = []
count = 0

def create(l):
    for i in range(len(l)):
        v = []
        for j in range(len(l[i])):
            v.append(0)
        vis.append(v)

    find(0, l)


def find(current, l):
    global lists
    global vis
    global all_list
    global count
    if current == len(l):
        all_list.append(lists)
        count += 1
        lists = []
        print(count)
        return

    for t in range(len(l[current])):
        if vis[current][t] == 0:
            vis[current][t] = 1
            lists.append(l[current][t])
            find(current + 1, l)
            vis[current][t] = 0


if __name__ == '__main__':

    inputs = []
    for i in range(54):
        input = []
        for j in range(3):
            input.append(i * j)
        inputs.append(input)

    create(inputs)
    print(all_list)
