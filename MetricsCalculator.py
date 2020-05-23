import networkx as nx
import osmnx as ox
from functools import partial
import xml.etree.ElementTree as ET
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import json
import threading
import time
from matplotlib.lines import Line2D

from utils.utils import *


class MetricsCalculator():

    def __init__(self, path_to_osm: str) -> None:
        self.graph = ox.core.graph_from_file(path_to_osm, simplify=False, retain_all=True)
        self.objs = []
        self.root = ET.parse(path_to_osm).getroot()
        self.weights = {}
        self.nodes = []
        self.clusters_results = {}
        self.inf_objs = [4353602429, 411827206, 469191096, 433569978, 241927948, 220628145, 475876249,
                         4346949771, 456682436, 1628030509, 4353588566, 175060062, 192073549, 4355578113,
                         4347321633, 412537658, 1412634107, 176134383, 192127240, 1185494724]
        self.chosen_objs = []
        self.chosen_inf_obj = 0
        self.non_oriented_adj_list = {}
        for obj in self.inf_objs:
            self.weights[obj] = random.random() + 1
        for obj in self.graph.adj.keys():
            if obj not in self.weights:
                self.weights[obj] = 1

        self.func_dict_nearest = {
            "fwd_node": partial(nearest_list_for_list, self.graph.adj, self.chosen_objs, self.inf_objs, self.weights),
            "fwd_inf": partial(nearest_list_for_list, self.graph.adj, self.inf_objs, self.chosen_objs, self.weights),
            "fwd_bwd_node": partial(nearest_fwd_bwd_list_for_list, self.graph.adj, self.chosen_objs, self.inf_objs,
                                    self.weights),
            "fwd_bwd_inf": partial(nearest_fwd_bwd_list_for_list, self.graph.adj, self.inf_objs, self.chosen_objs,
                                   self.weights),
            "bwd_node": partial(nearest_bwd_list_for_list, self.graph.adj, self.chosen_objs, self.inf_objs, self.weights),
            "bwd_inf": partial(nearest_fwd_bwd_list_for_list, self.graph.adj, self.inf_objs, self.chosen_objs, self.weights)
        }
        self.func_dict_distances = {
            "fwd_node": partial(distances_fwd, self.graph.adj, self.chosen_objs, self.inf_objs, self.weights),
            "fwd_inf": partial(distances_fwd, self.graph.adj, self.inf_objs, self.chosen_objs, self.weights),
            "bwd_node": partial(distances_bwd, self.graph.adj, self.chosen_objs, self.inf_objs, self.weights),
            "bwd_inf": partial(distances_bwd, self.graph.adj, self.inf_objs, self.chosen_objs, self.weights),
            "fwd_bwd_node": partial(distances_fwd_bwd, self.graph.adj, self.chosen_objs, self.inf_objs, self.weights),
            "fwd_bwd_inf": partial(distances_fwd_bwd, self.graph.adj, self.inf_objs, self.chosen_objs, self.weights)
        }

    def crop_and_save_graph(self, save=False):
        if save:
            fig, ax = ox.plot_graph(self.graph, save=True, show=False, filename='Ekb_graph', file_format='png',
                                    node_alpha=0, edge_color='b', edge_linewidth=0.7, dpi=200)
        ox.core.remove_isolated_nodes(self.graph)
        # removing_nodes = []
        # for i in self.graph.nodes:
        #     node = self.graph.nodes[i]
        #     if node['x'] < 60.46 or node['x'] > 60.86 or node['y'] < 56.74:
        #         removing_nodes.append(i)
        # self.graph.remove_nodes_from(removing_nodes)
        self.update_nodes_list()
        self.set_non_oriented_adj_list()

        removing_nodes = []
        distances, preds = distances_fwd(self.non_oriented_adj_list, [self.inf_objs[0]], self.nodes, self.weights)
        dists = distances[self.inf_objs[0]]
        for i in dists:
            if dists[i] == float('inf'):
                removing_nodes.append(i)

        self.graph.remove_nodes_from(removing_nodes)
        self.update_nodes_list()
        self.set_non_oriented_adj_list()

        inf_objs_set = set(self.inf_objs)
        self.objs = list(filter(lambda x: x not in inf_objs_set, self.nodes))

        if save:
            fig, ax = ox.plot_graph(self.graph, save=True, show=False, filename='Ekb_graph_cropped', dpi=200,
                                    file_format='png', node_alpha=0, edge_color='b', edge_linewidth=0.7)

    def set_inf_objs(self, objs):
        min_dists = [float("inf") for _ in range(len(objs))]
        self.inf_objs = [0 for _ in range(len(objs))]
        for i in self.nodes:
            phi1 = self.graph.nodes[i]['y']
            l1 = self.graph.nodes[i]['x']
            for j in range(len(objs)):
                phi2 = objs[j]['y']
                l2 = objs[j]['x']
                cos = math.sin(phi1) * math.sin(phi2) + math.cos(phi1) * math.cos(phi2) * math.cos(l1 - l2)
                dist = math.acos(cos) * 6371
                if dist < min_dists[j]:
                    min_dists[j] = dist
                    self.inf_objs[j] = i
        inf_objs_set = set(self.inf_objs)
        self.objs = list(filter(lambda x: x not in inf_objs_set, self.nodes))

    def update_nodes_list(self):
        self.nodes = list(self.graph.nodes.keys())

    # 1.a
    def nearest(self, mode: str, start: str, csv_file: str="./csv/nearest.csv") -> dict:
        dict_nearest = self.func_dict_nearest[mode + "_" + start]()

        if csv_file is not None:
            with open(csv_file, 'w') as f:
                csv_writer = csv.writer(f, delimiter='\t')
                csv_writer.writerow(['Vertex', 'Nearest'])
                for vertex in dict_nearest.keys():
                    csv_writer.writerow([str(vertex), ','.join(str(idx) for idx in dict_nearest[vertex])])

        return dict_nearest

    #1.b
    def closer_than_x(self, distance: int, mode: str, start: str, csv_file: str="./csv/closer_than_x.csv") -> dict:
        closer_than_x = {}
        distances, _ = self.func_dict_distances[mode + "_" + start]()
        if start == "node":
            tmp = {'from': self.chosen_objs,
                   'to': self.inf_objs}
        elif start == "inf":
            tmp = {'from': self.inf_objs,
                   'to': self.chosen_objs}

        for obj in tmp['from']:
            for obj2 in tmp['to']:
                if distances[obj][obj2] <= distance * 1000:
                    if obj not in closer_than_x:
                        closer_than_x[obj] = [obj2]
                    else:
                        closer_than_x[obj].append(obj2)

        if csv_file is not None:
            with open(csv_file, 'w') as f:
                csv_writer = csv.writer(f, delimiter='\t')
                csv_writer.writerow(['Vertex', 'Closer than {} kilometers'.format(str(distance))])
                for vertex in closer_than_x.keys():
                    csv_writer.writerow([str(vertex), ','.join(str(idx) for idx in closer_than_x[vertex])])

        return closer_than_x

    # 2
    def min_furthest_for_inf(self, mode: str) -> tuple:
        distances, _ = self.func_dict_distances[mode + "_" + "inf"]()
        min_ = float("inf")
        min_id = -1
        for obj in self.inf_objs:
            if min_ > max([(distances[obj][obj2], obj) for obj2 in self.objs if distances[obj][obj2] != float("inf")])[0]:
                min_, min_id = max([(distances[obj][obj2], obj) for obj2 in self.objs if distances[obj][obj2] != float("inf")])
        return (min_, min_id)

    # 3
    def closest_inf_in_summary(self, mode: str, start: str) -> tuple:
        distances, _ = self.func_dict_distances["fwd_inf"]()
        min_ = float("inf")
        min_id = -1
        for obj in self.inf_objs:
            dist = 0
            for obj2 in self.chosen_objs:
                if distances[obj][obj2] != float("inf"):
                    dist += distances[obj][obj2]
            if dist < min_:
                min_ = dist
                min_id = obj
        return min_, min_id

    # 4
    def min_weight_tree(self, csv_file: str="./csv/tree_of_min_weight_paths.csv") -> tuple:
        distances, preds = self.func_dict_distances['fwd_inf']()
        min_ = float("inf")
        min_id = -1
        min_edges = None
        for obj in self.inf_objs:
            edges = set()
            sum_ = 0
            for obj2 in self.chosen_objs:
                pred = preds[obj]
                curr = obj2
                curr_sum = 0
                while curr != obj:
                    if (curr, pred[curr]) not in edges:
                        if pred[curr] is None or curr is None:
                            curr_sum = 0
                            break
                        else:
                            edges.add((curr, pred[curr]))
                            curr_sum += (distances[obj][curr] - distances[obj][pred[curr]])
                    curr = pred[curr]
                sum_ += curr_sum
            if sum_ < min_:
                min_ = sum_
                min_id = obj
                min_edges = edges

        dict_ = {}
        for pair in min_edges:
            if pair[0] in dict_:
                dict_[pair[0]].append(pair[1])
            else:
                dict_[pair[0]] = [pair[1]]

        if csv_file is not None:
            with open(csv_file, 'w') as f:
                csv_writer = csv.writer(f, delimiter='\t')
                csv_writer.writerow(['Vertex', 'Adjacent vertexes'])
                for vertex in dict_.keys():
                    csv_writer.writerow([str(vertex), ','.join(str(idx) for idx in dict_[vertex])])

        return (min_, min_id)


    def list_to_obj_tree(self, objs, start_obj, filename, skip_inf_dists=False, write=True):
        distances, preds = distances_fwd(self.graph.adj, [start_obj], objs, self.weights)
        # distances, pred = distances_bwd(self.graph.adj, objs, [start_obj], self.weights)
        edges = set()
        sum_ = 0
        routes_list = []
        tree_dict = {}
        index = -1
        objs_without_routes = []
        for obj in objs:
            index += 1
            if distances[start_obj][obj] == float('inf'):
                if skip_inf_dists:
                    return 'no tree'
                else:
                    print('no way to ' + str(obj))
                    objs_without_routes.append((index, obj))
                    continue
            pred = preds[start_obj]
            curr = obj
            path_list = [curr]
            path_len = 0
            while curr != start_obj:
                path_list.append(pred[curr])
                if (curr, pred[curr]) not in edges:
                    edges.add((curr, pred[curr]))

                    if pred[curr] not in tree_dict:
                        tree_dict[pred[curr]] = [curr]
                    else:
                        tree_dict[pred[curr]].append(curr)

                    sum_ += (distances[start_obj][curr] - distances[start_obj][pred[curr]])
                    path_len += distances[start_obj][curr] - distances[start_obj][pred[curr]]
                curr = pred[curr]
            routes_list.append(list(reversed(path_list)))

        if write:
            with open(filename, 'w') as f:
                csv_writer = csv.writer(f, delimiter='\t')
                csv_writer.writerow(['Vertex', 'Adjacent vertexes'])
                for vertex in tree_dict.keys():
                    csv_writer.writerow([str(vertex), ','.join(str(idx) for idx in tree_dict[vertex])])

        return sum_, routes_list, objs_without_routes

    def write_csv(self, filename, rows):
        csv_file = open(os.path.join('./csv', filename), 'w')
        wr = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL, dialect='excel', lineterminator='\n')
        for row in rows:
            wr.writerow(row)

    def objs_into_clusters(self, k, filename: str = 'clusters.csv', write: bool = False):
        objs = self.chosen_objs
        if k > len(objs):
            return
        clusters = []
        dists_dict = {}

        for i in range(len(objs)):
            clusters.append([i])
            dists_dict[i] = {}
            for j in range(len(objs)):
                dists_dict[i][j] = float('inf')

        distances, _ = distances_bwd(self.non_oriented_adj_list, objs, objs, self.weights)
        for i in range(len(distances)):
            dists = distances[objs[i]]
            for j in range(len(dists)):
                if i == j:
                    continue
                if dists[objs[j]] < dists_dict[i][j]:
                    dists_dict[i][j] = dists_dict[j][i] = dists[objs[j]]

        history = [clusters]
        for _ in range(len(objs) - 1):
            min_ = float('inf')
            min_start = 0
            min_end = 1
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    max_ = -1
                    max_start = -1
                    max_end = -1
                    for ind1 in clusters[i]:
                        for ind2 in clusters[j]:
                            if dists_dict[ind1][ind2] != float('inf') and dists_dict[ind1][ind2] > max_:
                                max_ = dists_dict[ind1][ind2]
                                max_start = i
                                max_end = j
                    if max_ < min_ and max_ != -1:
                        min_ = max_
                        min_start = max_start
                        min_end = max_end
            new_clusters = []
            for i in range(len(clusters)):
                if i == min_start:
                    new_clusters.append(clusters[i] + clusters[min_end])
                    continue
                if i == min_end:
                    continue
                new_clusters.append(clusters[i])
            clusters = new_clusters
            history.append(clusters)

        if write:
            rows = [('', 'Cluster №', 'nodes')]
            for i in range(len(history) - 1, 0, -1):
                rows.append(('{} clusters'.format(len(history[i])), 'info'))
                for j in range(len(history[i])):
                    rows.append(('', str(j + 1), ','.join(str(node) for node in history[i][j])))
            self.write_csv(filename, rows)

        return clusters, history

    def dendrogram(self, clusters, history):
        fig, ax = plt.subplots()
        dict_ = {}
        for i in range(len(clusters[0])):
            dict_[clusters[0][i]] = i
        for i in range(1, len(history)):
            for key in dict_:
                ax.plot([dict_[key], dict_[key]], [i - 1, i], 'b')
            for j in range(len(history[i])):
                if len(history[i][j]) != len(history[i - 1][j]):
                    x1 = dict_[history[i][j][0]]
                    x2 = dict_[history[i][j][len(history[i][j]) - 1]]
                    ax.plot([x1, x2], [i, i], 'b')
                    new_ind = (x1 + x2) / 2
                    for elem in history[i][j]:
                        dict_[elem] = new_ind
                    break
        plt.xlabel(', '.join(str(s) for s in history[-1][0]))
        fig.savefig('images/dendrogram.png')

    def work_with_centroids(self, clusters, write: bool = False):
        inf_obj = self.chosen_inf_obj
        objs = self.chosen_objs
        obj_centroids = []
        all_routes = []
        objs_without_routes = []
        centroid_nodes = []
        number = 1
        for cluster in clusters:
            center_y = np.sum([self.graph.nodes[objs[i]]['y'] for i in cluster]) / len(cluster)
            center_x = np.sum([self.graph.nodes[objs[i]]['x'] for i in cluster]) / len(cluster)

            min_dist = float('inf')
            centr_obj_id = -1
            for id_ in self.nodes:
                try:
                    y = self.graph.nodes[id_]['y']
                    x = self.graph.nodes[id_]['x']
                    cos = math.sin(y) * math.sin(center_y) + math.cos(y) * math.cos(center_y) * math.cos(x - center_x)
                    dist = math.acos(cos) * 6371
                    if dist < min_dist:
                        min_dist = dist
                        centr_obj_id = id_
                except:
                    continue
            print('centroid ' + str(number) + ' id: ' + str(centr_obj_id))
            obj_centroids.append(centr_obj_id)
            cluster_objs = [objs[i] for i in cluster]
            weight, routes_list, objs_wt_routes = self.list_to_obj_tree(cluster_objs, centr_obj_id,
                                                        filename='./csv/cluster_tree_number_{}.csv'.format(number))
            all_routes += routes_list
            objs_without_routes += objs_wt_routes
            centroid_nodes.append(self.graph.nodes[centr_obj_id])
            number += 1

        name = str(len(clusters)) + '_clusters_trees'
        self.save_tree_plot(all_routes, centroid_nodes, name, objs_without_routes)

        name = str(len(clusters)) + '_centroids_tree'
        sum_, routes, objs_wt_routes = self.list_to_obj_tree(obj_centroids, inf_obj, './csv/' + name + '.csv')
        self.save_tree_plot(routes, [self.graph.nodes[inf_obj]], name, objs_wt_routes)

        return sum_

    def save_tree_plot(self, routes_list, blue_nodes, name, objs_wt_routes=None):
        if objs_wt_routes is None:
            objs_wt_routes = []
        fig, ax = ox.plot.plot_graph_routes(self.graph, routes_list, show=False, close=False, node_alpha=0,
                                            edge_color='lightgray', edge_alpha=1, edge_linewidth=0.8,
                                            route_color='#00cc66', route_linewidth=0.8, route_alpha=1,
                                            orig_dest_node_size=10, orig_dest_node_color='m', orig_dest_node_alpha=1)
        for node in blue_nodes:
            ax.scatter(node['x'], node['y'], c='#0000ff', s=10, zorder=10)

        for i in range(len(objs_wt_routes)):
            node = self.graph.nodes[objs_wt_routes[i][1]]
            ax.scatter(node['x'], node['y'], c='r', s=5, zorder=10)
            ax.annotate(xy=(node['x'], node['y']), s=str(objs_wt_routes[i][0]), size=4,
                        xytext=(node['x'] + 0.0025, node['y']))

        for i in range(len(routes_list)):
            node = self.graph.nodes[routes_list[i][-1]]
            text = ''
            try:
                text = str(self.chosen_objs.index(node['osmid']))
            except:
                text = str(i + 1)
            ax.annotate(xy=(node['x'], node['y']), s=text, size=4, xytext=(node['x'] + 0.0025, node['y']))
        ox.plot.save_and_show(fig, ax, save=True, show=False, filename=name,
                              file_format='png', close=True, dpi=200, axis_off=True)

    def save_adjacency_matrix(self, filename: str):
        self.df = nx.to_pandas_adjacency(self.graph, dtype=np.uint8)
        self.df.to_csv(filename)

    def save_adjacency_list(self, filename: str):
        with open(filename, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['Vertex', 'Adjacent vertexes'])
            for vertex in self.graph.adj.keys():
                writer.writerow([str(vertex), ','.join(str(vert) for vert in self.graph.adj[vertex])])

    def save_chosen_objs_to_csv(self):
        inf_obj = self.chosen_inf_obj
        csv_file = open(os.path.join('./csv', 'chosen_objs.csv'), 'w')
        wr = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL, dialect='excel', lineterminator='\n')
        wr.writerow(('num', 'id', 'x', 'y', 'is_inf'))
        wr.writerow((-1, inf_obj, self.graph.nodes[inf_obj]['x'], self.graph.nodes[inf_obj]['y'], True))
        for i in range(len(self.chosen_objs)):
            node = self.graph.nodes[self.chosen_objs[i]]
            wr.writerow((i, self.chosen_objs[i], node['x'], node['y'], False))

    def save_clusters(self, clusters, is_show=False):
        fig, ax = ox.plot.plot_graph(self.graph, show=False, close=False, node_alpha=0,
                                     edge_color='lightgray', edge_alpha=1, edge_linewidth=0.8)
        colors = ['#ff6666', '#66cc00', '#00cccc', '#0000ff', '#99004c']
        name = str(len(clusters)) + '_clusters'
        for i in range(len(clusters)):
            for elem in clusters[i]:
                node = self.graph.nodes[self.chosen_objs[elem]]
                ax.scatter(node['x'], node['y'], c=colors[i], s=10, zorder=10)
                ax.annotate(xy=(node['x'], node['y']), s=str(elem), size=4, xytext=(node['x'] + 0.0025, node['y']))
        ox.plot.save_and_show(fig, ax, save=True, show=is_show, filename=name,
                              file_format='png', close=True, dpi=200, axis_off=True)

    def work_with_clusters(self, history, amount):
        clusters = history[len(history) - amount]
        self.save_clusters(clusters)
        sum_ = self.work_with_centroids(clusters, write=True)
        print(str(amount) + ' clusters', clusters)
        self.clusters_results[amount] = sum_
        return sum_

    def set_objs(self, n):
        objs = []
        objs_set = set()
        for i in range(n):
            id_ = self.objs[random.randint(0, len(self.objs) - 1)]
            if id_ not in objs_set:
                objs.append(id_)
                objs_set.add(id_)
                print(i, id_)
        self.chosen_objs = objs
        self.chosen_inf_obj = self.inf_objs[random.randint(0, len(self.inf_objs) - 1)]

    def add_obj(self):
        objs_set = set(self.chosen_objs)
        while True:
            id_ = self.objs[random.randint(0, len(self.objs) - 1)]
            if id_ not in objs_set:
                self.chosen_objs.append(id_)
                break

    def set_inf_obj(self, num):
        if 0 <= num < len(self.inf_objs):
            self.chosen_inf_obj = self.inf_objs[num]

    def set_non_oriented_adj_list(self):
        adj_list = {}
        for node in self.nodes:
            adj_list[node] = {}

        for id_1 in self.graph.adj:
            for id_2 in self.graph.adj[id_1]:
                try:
                    if self.graph.adj[id_1][id_2][0]['length'] < adj_list[id_1][id_2][0]['length']:
                        adj_list[id_1][id_2] = self.graph.adj[id_1][id_2]
                except:
                    adj_list[id_1][id_2] = self.graph.adj[id_1][id_2]
                try:
                    if self.graph.adj[id_1][id_2][0]['length'] < adj_list[id_2][id_1][0]['length']:
                        adj_list[id_2][id_1] = self.graph.adj[id_1][id_2]
                except:
                    adj_list[id_2][id_1] = self.graph.adj[id_1][id_2]

        # return adj_list
        self.non_oriented_adj_list = adj_list


if __name__ == "__main__":
    print("hello world")
    m = MetricsCalculator('./Graph_Uni_Proj/Ekb.osm')
    # print(len(m.graph.adj))
    m.crop_and_save_graph()
    #print(m.objs)
    m.set_objs(10)
    print(m.min_weight_tree(csv_file=None))

    # dists, preds = dijkstra(m.graph.adj, 420036591, m.weights)
    # print(dists[3755440425])
    # hospitals = 7
    # fire_departments = 5
    # fig, ax = ox.plot_graph(m.graph, save=False, show=False, node_alpha=0, edge_color='lightgray', edge_linewidth=0.7)
    # for i in range(len(m.inf_objs)):
    #     color = 'k'
    #     if i < hospitals:
    #         color = 'g'
    #     elif i < hospitals + fire_departments:
    #         color = 'r'
    #     else:
    #         color = 'b'
    #     node = m.graph.nodes[m.inf_objs[i]]
    #     ax.scatter(node['x'], node['y'], c=color, s=7, zorder=10)
    #     ax.annotate(xy=(node['x'], node['y']), s=str(i), size=4, xytext=(node['x'] + 0.0025, node['y']))
    #
    # legend_elems = [Line2D([0], [0], marker='o', color='w', label='hospital',
    #                           markerfacecolor='g', markersize=7),
    #                 Line2D([0], [0], marker='o', color='w', label='fire department',
    #                           markerfacecolor='r', markersize=7),
    #                 Line2D([0], [0], marker='o', color='w', label='shop',
    #                           markerfacecolor='b', markersize=7)]
    # ax.legend(handles=legend_elems, loc='lower left', fontsize= 'x-small')
    # ox.plot.save_and_show(fig, ax, save=True, show=False, filename='inf_objs', file_format='png',
    #                       close=True, dpi=200, axis_off=True)
