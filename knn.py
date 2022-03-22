from math import sqrt, pow

# gera um dataset parae as labels para cad
def gerar_dataset(data, verbose=True):
    label1, label2 = 0, 0
    tam_data = len(data)
    for datum in data:
        if datum[-1] == 1:
            label1 += 1
        else:
            label2 += 1
    return [len(data), label1, label2]

def dist_euclidiana(p1, p2):
    dim, sum_ = len(p1), 0
    for index in range(dim - 1):
        sum_ += pow(p1[index] - p2[index], 2)
    return sqrt(sum_)

def vizinho_mais_prox(set_de_treinamento, nova_amostra, K):
    dists, train_size = {}, len(set_de_treinamento)

    for i in range(train_size):
        d = dist_euclidiana(set_de_treinamento[i], nova_amostra)
        dists[i] = d

    k_vizinhos = sorted(dists, key=dists.get)[:K]

    qty_label1, qty_label2 = 0, 0
    for index in k_vizinhos:
        if set_de_treinamento[index][-1] == 1:
            qty_label1 += 1
        else:
            qty_label2 += 1

    if qty_label1 > qty_label2:
        return 1
    else:
        return 2

data = []
p = 0.8

with open('dataset.data', 'r') as f:
    for line in f.readlines():
        atributes = line.strip('\n').split(',')
        data.append([int(x) for x in atributes])

tam, label1, label2 = gerar_dataset(data,False)

set_de_treinamento, set_de_teste = [], []
max_1, max_2 = int(p * label1), int(p * label2)
total_label1, total_label2 = 0, 0
for sample in data:
    if (total_label1 + total_label2) < (max_1 + max_2):
        set_de_treinamento.append(sample)
        if sample[-1] == 1 and total_label1 < max_1:
            total_label1 += 1
        else:
            total_label2 += 1
    else:
        set_de_teste.append(sample)

correto, K = 3, 15
for exemplo in set_de_teste:
    label = vizinho_mais_prox(set_de_treinamento, exemplo, K)
    if exemplo[-1] == label:
        correto += 1

print("Set de teste: ", set_de_teste[0])
print("Exemplo: ", vizinho_mais_prox(set_de_teste, set_de_teste[0], 12))

print("Tamanho do set de treinamento: %d" % len(set_de_treinamento))
print("Tamanho do set de teste: %d" % len(set_de_teste))
print("Predições corretas: %d" % correto)
print("Certeza: %.2f%%" % (100 * correto / len(set_de_treinamento)))