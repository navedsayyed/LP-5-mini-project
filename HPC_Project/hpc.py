try:
    import cupy as xp
    print(xp.__version__)
    print(xp.cuda.runtime.getDeviceCount())
    print("GPU Mode (CuPy)")
except ImportError:
    import numpy as xp
    print("CPU Mode (NumPy)")

import heapq
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx

text = input("Enter text: ")
print("\nInput:", text)


# def get_frequency(text):
#     arr = xp.array(list(text))
#     unique, counts = xp.unique(arr, return_counts=True)
#     return {str(k): int(v) for k, v in zip(unique, counts)}

def get_frequency(text):
    arr = xp.array([ord(c) for c in text])   # convert char → int
    unique, counts = xp.unique(arr, return_counts=True)
    return {chr(int(k)): int(v) for k, v in zip(unique, counts)}
freq = get_frequency(text)
print("Frequency:", freq)


class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_tree(freq):
    heap = [Node(c, f) for c, f in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)

        merged = Node(None, n1.freq + n2.freq)
        merged.left, merged.right = n1, n2

        heapq.heappush(heap, merged)

    return heap[0]


root = build_tree(freq)


def generate_codes(node, prefix="", code_map=None):
    if code_map is None:
        code_map = {}

    if node.char is not None:
        code_map[node.char] = prefix
        return code_map

    generate_codes(node.left, prefix + "0", code_map)
    generate_codes(node.right, prefix + "1", code_map)

    return code_map


codes = generate_codes(root)
print("Codes:", codes)

encoded = ''.join(codes[c] for c in text)
print("Encoded:", encoded)


def decode(encoded, root):
    result = []
    node = root

    for bit in encoded:
        node = node.left if bit == '0' else node.right
        if node.char:
            result.append(node.char)
            node = root

    return ''.join(result)


print("Decoded:", decode(encoded, root))


def plot_tree(root):
    G = nx.DiGraph()

    def add_edges(node, parent=None, label=""):
        if node is None:
            return

        node_label = f"{node.char}:{node.freq}" if node.char else f"{node.freq}"
        G.add_node(node_label)

        if parent:
            G.add_edge(parent, node_label, label=label)

        if node.left:
            add_edges(node.left, node_label, "0")
        if node.right:
            add_edges(node.right, node_label, "1")

    add_edges(root)

    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'label')

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Huffman Tree")
    plt.show()


plot_tree(root)