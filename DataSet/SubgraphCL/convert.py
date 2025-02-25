import pandas as pd
from tqdm import tqdm
import networkx as nx

data_label = 'depression'

A = False
node_labels = False
graph_id = False
graph_label = True
text_bert = False
entity_bert = False

if A:
    adj = pd.read_csv(data_label + '.cites', sep='\t', header=None)
    '''G, adj_temp, num = nx.Graph(), [], 0
    for i in range(len(adj)):
        if adj.values[i][0] == adj.values[i][1]:
            break
        a, b = adj.values[i][0], adj.values[i][1]
        G.add_edge(a, b)
    for c in nx.connected_components(G):
        num += 1
        nodeSet = G.subgraph(c).nodes()
        if num != 1:
            for n, m in G.subgraph(c).adjacency():
                adj_temp.append(n)'''
    for i in tqdm(range(adj.shape[0])):
        if adj.values[i][0] == adj.values[i][1]:
            break
        # if adj.values[i][0] not in adj_temp and adj.values[i][1] not in adj_temp:
        pd.DataFrame([[adj.values[i][0]+1, adj.values[i][1]+1]]).to_csv('depression_A.txt', mode='a', header=False, sep=',',
                                         index=False, encoding='UTF-8')

if node_labels:
    type_list = ['text', 'entity', 'time', 'sent', 'person', 'emo', 'net', 'twt', 'scale', 'topic']
    # type_list = ['text', 'topic', 'entity', 'time', 'sent', 'person', 'emo', 'net', 'twt']
    # type_list = ['text', 'topic', 'entity']
    idx = {}
    for i in tqdm(range(len(type_list))):
        data = pd.read_csv(data_label + '.emb2_' + type_list[i], sep=',', header=None)
        data_id = pd.read_csv(data_label + '.content.' + type_list[i], sep='\t', header=None)
        for j in tqdm(range(data.shape[0])):
            idx[int(data_id.values[j][0])] = str(data.values[j]).replace('[', '').replace(']', '').replace("'", '') \
                .replace(' ', ',').replace(',,', ',').replace('\n', '')

    for i in tqdm(sorted(idx.keys())):
        pd.DataFrame([idx[i].split(',')]).to_csv('depression_node_labels.txt', mode='a', header=False, sep=',',
                                             index=False, encoding='UTF-8')


if graph_label:
    text_label = pd.read_csv(data_label + '.txt', sep='\t', header=None)
    map_index = pd.read_csv('mapindex.txt', sep='\t', header=None)

    for i in tqdm(range(map_index.shape[0])):
        if map_index.values[i][0].isdigit():
            if text_label.values[int(map_index.values[i][0]) - 1][1] == 'positive':
                pd.DataFrame([[map_index.values[i][1], '1']]).to_csv(
                    'depression_word_node_labels.txt', mode='a', sep='\t', header=False, index=False, encoding='UTF-8')
            else:
                pd.DataFrame([[map_index.values[i][1], '0']]).to_csv(
                    'depression_word_node_labels.txt', mode='a', sep='\t', header=False, index=False, encoding='UTF-8')

if graph_id:
    # 读数据
    A = pd.read_csv('depression_A.txt', sep=',', header=None)
    nodes_all = pd.read_csv('depression_node_labels.txt', sep='\t', header=None)
    label = pd.read_csv('depression_word_node_labels.txt', sep='\t', header=None)

    # 初始化
    Matrix = A.values.tolist()
    nodes = [i for i in range(len(nodes_all))]
    label = label[0].tolist()

    # 构建图
    G = nx.Graph()
    for i in tqdm(range(len(Matrix))):
        a, b = Matrix[i][0], Matrix[i][1]
        G.add_edge(a, b)

    for node in tqdm(range(1, len(nodes_all) + 1)):
        G.add_node(node)

    num, adj, ans = 0, [], {}
    # 得到节点对应的图
    for c in nx.connected_components(G):
        nodeSet = G.subgraph(c).nodes()
        adj.append([])
        flag = 0
        for n, m in G.subgraph(c).adjacency():
            ans[n - 1] = num + 1
            if n in label:
                adj[num].append(n)
                flag = 1
        if not flag:
            print(nodeSet)
        num += 1

    length = len(adj[0])
    for i in range(len(adj)):
        if len(adj[i]) > length:
            length = len(adj[i])
    print('最大sub_size：', length)

    for i in tqdm(sorted(ans)):
        pd.DataFrame([ans[i]]).to_csv('depression_graph_indicator.txt', mode='a', header=False,
                                      index=False, encoding='UTF-8')



if text_bert:
    text = pd.read_csv(data_label + '.content.text', sep='\t', header=None)
    vec = pd.read_csv('text_bert.txt', sep='\t', header=None)

    for i in tqdm(range(text.shape[0])):
        temp = str(vec.values[i]).replace('[', '').replace(']', '').replace("'", '') \
                    .replace(' ', ',').replace(',,', ',').replace('\n', '').split(',')
        temp.insert(0, text.values[i][0])
        temp.append(text.values[i][-1])
        pd.DataFrame([temp]).to_csv(data_label + '.content.text1', mode='a', sep='\t', header=False, index=False, encoding='UTF-8')

if entity_bert:
    entity = pd.read_csv(data_label + '.content.entity', sep='\t', header=None)
    vec = pd.read_csv('entity_bert.txt', sep='\t', header=None)
    name = pd.read_csv('entity_name.txt', sep='\t', header=None)
    index = pd.read_csv('entityIndex.txt', sep='\t', header=None)
    flag = 0

    for i in tqdm(range(name.shape[0])):
        for j in range(flag, index.shape[0]):
            if name.values[i] == index.values[j]:
                temp = str(vec.values[i]).replace('[', '').replace(']', '').replace("'", '') \
                    .replace(' ', ',').replace(',,', ',').replace('\n', '').split(',')
                temp.insert(0, entity.values[i][0])
                temp.append(entity.values[i][-1])
                pd.DataFrame([temp]).to_csv(data_label + '.content.entity1', mode='a', sep='\t', header=False,
                                            index=False, encoding='UTF-8')
                flag = j
                break
