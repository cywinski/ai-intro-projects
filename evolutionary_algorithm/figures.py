from matplotlib import pyplot as plt
import networkx as nx


def make_figures(fit_val, vert, edges, iter):
    fig, ax = plt.subplots(nrows=1, ncols=3)

    ax[0].plot(iter, fit_val)
    ax[1].plot(iter, vert)
    ax[2].plot(iter, edges)

    ax[0].set_xlabel('population')
    ax[1].set_xlabel('population')
    ax[2].set_xlabel('population')

    ax[0].set_ylabel('values of fitness score')
    ax[1].set_ylabel('number of vertexes')
    ax[2].set_ylabel('number of uncovered edges')

    ax[0].grid()
    ax[1].grid()
    ax[2].grid()

    plt.tight_layout()
    plt.show()

def draw_graph(graph):

    # Extract nodes from graph
    nodes = set([n1 for n1, n2 in graph] + [n2 for n1, n2 in graph])

    # Create networkx graph
    G=nx.Graph()

    # Add nodes
    for node in nodes:
        G.add_node(node)

    # Add edges
    for edge in graph:
        G.add_edge(edge[0], edge[1])

    # Draw graph
    pos = nx.shell_layout(G)

    nx.draw_networkx_nodes(G,pos,node_size=200, 
                           alpha=0.7, node_color='red')
    nx.draw_networkx_edges(G,pos,width=0.5,
                           alpha=0.5)
    nx.draw_networkx_labels(G, pos)

    plt.show()