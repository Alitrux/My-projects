#include "dendrogram.hpp" // This is the header for the class implemented here

#include "cloud.hpp"      // Variable c of type cloud& is used in the constructor and several other methods
#include "edge.hpp"       // Used in several methods, e.g. dendrogram::merge(edge *e)
#include "graph.hpp"      // Variable g of type graph * is used in the constructor and several other methods

#include <iostream> // This provides the input/output functionality
#include <cassert>  // This provides the assert() method

using namespace std;

dendrogram::dendrogram(graph &_g) {
    g = &_g;

    int n = g->get_num_nodes();

    parent = new int[n];
    rank = new int[n];

    left = new int[n];
    down = new int[n];

    height = new double[n];

    clusters = new int[n];

    for (int i = 0; i < n; i++) {
        parent[i] = -1;
        rank[i] = 0;

        left[i] = -1;
        down[i] = -1;

        height[i] = -1;

        clusters[i] = -1;
    }

    cut_height = -1;
    total_clusters = 0;
    nonsingle_clusters = 0;
}

dendrogram::~dendrogram() {
    delete[] parent;
    delete[] left;
    delete[] down;
    delete[] rank;
    delete[] height;
    delete[] clusters;

    if (sign_heights != nullptr)
        delete[] sign_heights;
}

int dendrogram::find(int i) {
    assert(0 <= i && i < g->get_num_nodes());
    int result = i;
    while (get_parent(result) != -1)
        result = get_parent(result);
    return result;
}

void dendrogram::merge(edge *e) {
    // TODO: Exercise 5
    // Plan:
    // 1. Find the representatives
    // 2. Choose the highest
    // 3. Adjust parent, left, and down
    // 4. Update ranks
    // 5. Update heights

    int cluster1 = find(e->get_p1());
    int cluster2 = find(e->get_p2());
    int representative;
    int non_representative;

    if (cluster1 == cluster2)
        return;

    if (get_rank(cluster1) > get_rank(cluster2))
    {
        representative = cluster1;
        non_representative = cluster2;
        parent[cluster2] = representative;
    }
    else if (get_rank(cluster2) > get_rank(cluster1))
    {
        representative = cluster2;
        non_representative = cluster1;
        parent[cluster1] = representative;
    }
    else if (cluster1 > cluster2)
    {
        representative = cluster1;
        non_representative = cluster2;
        parent[cluster2] = representative;
    }
    else
    {
        representative = cluster2;
        non_representative = cluster1;
        parent[cluster1] = representative;
    }

    left[non_representative] = down[representative];
    down[representative] = non_representative;

    rank[representative] = max(rank[representative], rank[non_representative] + 1);
    height[non_representative] = e->get_length()/2;


}

void dendrogram::build() {
    g->start_iteration();

    for (edge *e = g->get_next(); e != nullptr; e = g->get_next()) {
        merge(e);
    }
}

double dendrogram::get_dendro_height() {
    return height[down[find(0)]];
}

void dendrogram::set_clusters(int i, double h) {
    assert(0 <= i && i < g->get_num_nodes());

    if (parent[i] != -1 && height[i] <= h)
        clusters[i] = clusters[parent[i]];
    else {
        clusters[i] = i;
        ++total_clusters;
    }


    if (left[i] != -1)
        set_clusters(left[i], h);
    if (down[i] != -1)
        set_clusters(down[i], h);

}

void dendrogram::set_clusters(double h) {
    if (cut_height != h) {
        cut_height = h;
        set_clusters(find(0), h);
    }
}

int dendrogram::_count_ns_clusters() {
    int n = g->get_num_nodes();
    int count = 0;
    // TODO: Exercise 8
    int clusterSizes[n];
    for (int i = 0; i < n; ++i) {
        clusterSizes[i] = 0;
    }

    for (int i = 0; i < n; ++i) {
        ++clusterSizes[clusters[i]];
    }
    for (int i = 0; i < n; ++i) {
        if (clusterSizes[i] >= 2)
            ++count;
    }
    return count;
}

int dendrogram::count_ns_clusters() {
    if (nonsingle_clusters == 0)
        nonsingle_clusters = _count_ns_clusters();
    return nonsingle_clusters;
}

void dendrogram::clear_clusters() {
    for (int i = 0; i < g->get_num_nodes(); i++) {
        clusters[i] = -1;
    }
    nonsingle_clusters = 0;
    total_clusters = 0;
    cut_height = -1;
}

double dendrogram::get_cluster_height(int cluster) {
    assert(0 <= cluster && cluster < g->get_num_nodes());
    // TODO: Exercise 9
    if (down[cluster] == -1)
        return 0;
    //On descend d'un et on va à gauche jusqu'à réentrer dans le cluster
    int current_index = down[cluster];
    while (clusters[current_index] != cluster)
        current_index = left[current_index];
    return height[current_index];
}

/******** Significant heights ********/

int count_non_zero(double *unfiltered, int countu) {
    int countf = 0;
    for (int i = 0; i < countu; i++) {
        if (unfiltered[i] > 0)
            countf++;
    }

    return countf;
}

double *filter_double_array(double *unfiltered, int countu, int countf) {
    double *filtered = new double[countf];
    int f = 0;
    for (int i = 0; i < countu; i++) {
        if (unfiltered[i] > 0)
            filtered[f++] = unfiltered[i];
    }
    return filtered;
}

void dendrogram::find_heights(double eps) {
    double h = get_dendro_height();
    int slots = 1 / eps + 1;
    double *buckets = new double[slots];
    for (int i = 0; i < slots; i++)
        buckets[i] = 0;

    for (int i = 0; i < g->get_num_nodes(); i++) {
        if (height[i] != -1 && height[i] > buckets[(int)(height[i] / eps / h)])
            buckets[(int)(height[i] / eps / h)] = height[i];
    }

    count_sign_heights = count_non_zero(buckets, slots);

    if (sign_heights != nullptr)
        delete[] sign_heights;

    sign_heights = filter_double_array(buckets, slots, count_sign_heights);
    delete[] buckets;
}

/***** GETTERS *****/

long dendrogram::get_n() {
    return g->get_num_nodes();
}

int dendrogram::get_parent(int i) {
    assert(0 <= i && i < g->get_num_nodes());
    return parent[i];
}

int dendrogram::get_down(int i) {
    assert(0 <= i && i < g->get_num_nodes());
    return down[i];
}

int dendrogram::get_left(int i) {
    assert(0 <= i && i < g->get_num_nodes());
    return left[i];
}

int dendrogram::get_rank(int i) {
    assert(0 <= i && i < g->get_num_nodes());
    return rank[i];
}

int dendrogram::get_cluster(int i) {
    assert(0 <= i && i < g->get_num_nodes());
    return clusters[i];
}

std::string& dendrogram::get_name(int i) {
    assert(0 <= i && i < g->get_num_nodes());
    return g->get_name(i);
}

double dendrogram::get_height(int i) {
    assert(0 <= i && i < g->get_num_nodes());
    return height[i];
}

int dendrogram::get_count_sign_heights() {
    return count_sign_heights;
}

double dendrogram::get_sign_height(int i) {
    assert(0 <= i && i < count_sign_heights);
    return sign_heights[i];
}

int dendrogram::get_total_clusters() {
    return total_clusters;
}

double dendrogram::get_cut_height() {
    return cut_height;
}

/****** Methods for testing ******/

void dendrogram::print_node(int i) {
    cout << "Node " << i
         << "(parent =  " << parent[i]
         << ", down = " << down[i]
         << ", left = " << left[i]
         << ", rank = " << rank[i]
         << ", height = " << height[i]
         << ", cluster = " << clusters[i]
         << ")";
}

void dendrogram::print_dendrogram() {
    cout << "node\tparent\trank\tleft\tdown\theight\tcluster" << endl;
    for (int i = 0; i < get_n(); i++)
        cout << i << "\t"
             << parent[i] << "\t"
             << rank[i] << "\t"
             << left[i] << "\t"
             << down[i] << "\t"
             << height[i] << "\t"
             << clusters[i] << "\t"
             << endl;
}

void dendrogram::print_clusters() {
    int print_after[g->get_num_nodes()];
    for (size_t i = 0; i < g->get_num_nodes(); i++) {
        print_after[i] = -1;
    }
    // Go through all nodes
    for (size_t i = 0; i < g->get_num_nodes(); i++) {
        if (clusters[i] != i) {
            print_after[i] = print_after[clusters[i]];
            print_after[clusters[i]] = i;
        }
    }

    // Print the clusters
    for (int i = 0; i < g->get_num_nodes(); i++) {
        // For each encountered cluster representative
        if (clusters[i] == i) {
            // Print the corresponding cluster
            cout << "Cluster \"" << clusters[i]
                 << "\" (node: " << i << ";"
                 << " height: " << get_cluster_height(i)
                 << ")" << endl;

            int next = print_after[i];
            // Print all points in the cluster
            while (next != -1) {
                std::cout << get_name(next) << std::endl;
                next = print_after[next];
            }
        }
    }
}

void dendrogram::iterate_sign_heights() {
    static int i = 0;
    double cut = get_sign_height(i);
    cout << "Setting clusters at height " << cut
         << "..." << endl;
    clear_clusters();
    set_clusters(cut);
    cout << "\t" << count_ns_clusters() << " non-singleton cluster"
         << (count_ns_clusters() > 1 ? "s" : "")
         << " below " << cut << endl;
    i = (i + 1) % get_count_sign_heights();
}

void dendrogram::set_parent(int *_parent) {
    for (int i = 0; i < g->get_num_nodes(); i++)
        parent[i] = _parent[i];
}
