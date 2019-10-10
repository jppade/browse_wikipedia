from bs4 import BeautifulSoup as bs
import requests
import numpy as np
import time
import networkx as nx
import matplotlib.pyplot as plt
import random

"""Remark: Browsing the web is relatively slow. This is because the function
 requests.get() is relatively slow. Try to lower the limit value in order
 to get a deeper graph."""

def get_hrefs(name, limit=100):
    """
    @name is a string which corresponds to the name of the Wikipedia entry.
    @limit is an integer.
    The first @limit pages which link to the page @name are returned (at most).
    """
    page = "https://en.wikipedia.org/w/index.php?title=Special:WhatLinksHere/"\
            + name + "&namespace=0&limit=" + str(limit)
    page_response = requests.get(page)
    page_content = bs(page_response.content, "html.parser")
    # Get all hrefs from the site's content
    return page_content.find_all("a")


def get_views(name):
     """Returns number of views in the past 30 days of Wikipedia entry with
     name @name."""
     page = "https://en.wikipedia.org/w/index.php?title="\
                                + name + "&action=info"
     page_response = requests.get(page)
     page_content = bs(page_response.content, "html.parser")
     try:
         page_views = str(page_content).split("Page views in the past 30 days")\
         [1].split("\">")[1].split("<")[0]
     except IndexError:
         # This error arises when a Wikipedia entry is faulty
         page_views = str(0)
     return int(page_views.replace(",", ""))


def get_predecessors(name, connections, num_pred, limit=100):
    """Get Wiki entries which are linked to the entry
    https://en.wikipedia.org/wiki/name. In case they are not yet contained in
    the dictionary @connections they are written into it.
    @num_pred is the maximal number of entries added to connections.
    @limit is the length of the list of entries which are linked to the entry 
    corresponding to @name."""
    if name not in connections.keys():  
        links = get_hrefs(name, limit=num_pred)
        # Restrict to hrefs to Wikipedia main articles (excludes for instance
        # references and discussions).
        strng_link = "a href=\"/wiki/"
        themes = []
        count = 0
        for link in links:
            if strng_link in str(link):
                # Isolate name of Wikipedia entry
                theme = str(link).split("<a href=\"/wiki/")[1].split("\"")[0]
                if theme == name:
                    # Prevent loops
                    continue
                if (":" not in theme and "User" not in theme 
                    and "Help" not in theme and count < num_pred):
                    # Excludes references and discussions by ":" not in theme
                    themes.append(theme)
                    count += 1
        # Make the list unique:
        themes = list(np.unique(themes))
        connections[name] = themes
    return connections


def get_pred_ranked(name, connections, num_pred, limit=100):
    """Get the @num_pred most viewed Wiki entries which are linked to the entry
    "https://en.wikipedia.org/wiki/name". In case they are not yet contained in
    the dictionary @connections they are written into it.
    @num_pred is the maximal number of entries added to connections.
    @limit is the length of the list of entries which are linked to the entry 
    corresponding to @name."""
    if name not in connections.keys():            
        links = get_hrefs(name, limit=limit)
        # Restrict to hrefs to Wikipedia main articles (excludes for instance
        # references and discussions).
        strng_link = "a href=\"/wiki/"
        themes = []
        count = 0
        for link in links:
            if strng_link in str(link):
                # Isolate name of Wikipedia entry
                theme = str(link).split("<a href=\"/wiki/")[1].split("\"")[0]
                if theme == name:
                    # Prevent loops
                    continue
                if ":" not in theme and "User" not in theme and "Help" not in theme:
                    # Excludes references and discussions by ":" not in theme
                    page_theme_views = get_views(theme)
                    if len(themes) < num_pred:
                        # Fill the themes list up
                        themes.append((theme, page_theme_views))
                    else:
                        count +=1
                        """ If the themes list is full, replace the one with 
                        least views"""
                        views = [theme[1] for theme in themes]
                        if page_theme_views > min(views):
                            ind_min = views.index(min(views))
                            del themes[ind_min]
                            themes.append((theme, page_theme_views))
        # Make the list unique:
        themes = list(np.unique([theme[0] for theme in themes]))
        connections[name] = themes
    return connections
        
        

def get_all_predecessors(connections, num_pred, method=get_predecessors, limit=100):
    """ Extend the dictionary of keys=sites, values=predecessors by the
    predecessors of the predecessors (the latter thereby become keys in the
    new dictionary)."""
    # First, get all predecessors (values) of the dictionary and collect them
    # in a unique(!) list.
    words_list = list()
    for site in connections:
        for predec in connections[site]:
            if predec not in connections and predec not in words_list:
                words_list.append(predec)
    # Now, determine the predecessors for each (unique) word
    for word in words_list:
        connections = method(word, connections, num_pred, limit=limit)
    return connections


def inc2adj(incidence_list):
    """ Transforms a dictionary into the adjacency matrix of a directed graph.
    All elements of the dictionary, i.e. keys and values are interpreted as
    nodes of the graph. The values of a key are interpreted as ingoing nodes
    of the node associated with the key. Furthermore, the dictionary node_values
    is returned. It associates to each site-name (key) a unique integer (value)."""
    # The number of nodes corresponds to the number of unique entries in the
    # incidence list (keys + values).
    num_nodes = len(list(set(list(incidence_list.keys())
                             + [value for key in incidence_list.keys() for value in incidence_list[key]])))
    adj = np.zeros((num_nodes, )*2)
    # Make a dictionary which assigns a unique integer to each entry in the
    # incidence list. Begin with numerating the keys.
    node_values = {key: i for i, key in enumerate(incidence_list.keys())}
    # begin with filling the adjacency matrix for the keys
    count = len(incidence_list)
    for key in incidence_list.keys():
        for value in incidence_list[key]:
            if value in node_values:
                # If the value already has a number, put a 1 in the adjacency matrix
                adj[node_values[key], node_values[value]] = 1
            else:
                # Number the value and put it in the adjacency matrix
                node_values[value] = count
                adj[node_values[key], count] = 1
                count += 1
    return adj, node_values


def plot_digraph(adjacency, node_values):
    """ Plots a digraph from an adjacency matrix using the networkx module.
     Needs a dictionary node_values with node names as keys."""
    plt.figure()
    # Set nodes and (directed) edges
    num_nodes = len(adjacency)
    positions = [(np.real(np.exp(2 * np.pi * 1j * k / num_nodes)), np.imag(np.exp(2 * np.pi * 1j * k / num_nodes)))
                 for k in range(num_nodes)]
    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(i, pos=positions[i])
        for k in range(num_nodes):
            if adjacency[i, k] > 0:
                G.add_edge(k, i)
    pos = nx.get_node_attributes(G, 'pos')
    labeldict = dict({(i, key) for i, key in enumerate(node_values.keys())})

    # Draw network
    edges = G.edges()
    nx.draw_networkx_labels(G, pos, labels=labeldict, font_size=10, font_weight='normal')
    nx.draw_networkx_nodes(G, pos, node_color='pink')
    nx.draw_networkx_edges(G, pos, edges=edges, alpha=0.4, arrowsize=20)
    plt.show()


def reduce_graph(adjacency, node_labels, centrality, rate):
    """ This function reduces a large adjacency matrix according to some
    measure. Measures can be node-degree, eigenvector centrality, or more
    generally any centrality measure. All nodes which have a centrality
    value above rate*max(centrality) are included in the new graph."""

    # Determine nodes which have a centrality value above threshold
    values = centrality(adjacency)
    M = max(values)
    node_list = [k for k in range(len(adjacency)) if values[k] > M*rate]
    node_list.sort(key=lambda x: values[x], reverse=True)

    # Get labels of the corresponding nodes
    aux_dict = {value: key for key, value in node_labels.items()}
    node_labels_new = {aux_dict[node]:i for i, node in enumerate(node_list)}

    # Build reduced adjacency matrix with associated node labels
    adjacency_reduced = np.zeros((len(node_list), )*2)
    count = 0
    for node in node_list:
        adjacency_reduced[count, :] = [adjacency[node, num] for num in node_list]
        count += 1

    return adjacency_reduced, node_labels_new


def degree_centrality(adjacency, side='right'):
    """ Compute the degree centrality of the graph given by adjacency matrix.
    By default, the in-degree centrality is computed. If side='left' the out-
    degree centrality is computed. """
    if side == 'left':
        adjacency = adjacency.transpose()
    return np.array([sum(adjacency[k, :]) for k in range(len(adjacency))])


def eigenvector_centrality(adjacency, side='right'):
    """ Return the eigenvector centrality of an adjacency matrix. The matrix
    must be given as a (square) numpy array. If side='left', the left
    eigenvector, corresponding to the centrality of the node corresponding to
    the in-degree, is computed. Otherwise the right, corresponding to the
    out-degree. """
    if side == 'left':
        adjacency = adjacency.transpose()
    w, v = np.linalg.eig(adjacency)
    
    # Find index of largest eigenvalue (which exists and is positive by the
    # Perron-Frobenius theorem)
    ind = np.argmax(w)
    eigenvector = np.round(v[:, ind].real, 12)

    return [abs(entry) for entry in eigenvector]    
