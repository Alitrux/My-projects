
#include "KnnClassification.hpp"
#include <iostream>
#include <vector>
#include <ANN/ANN.h>


KnnClassification::KnnClassification(int k, Dataset *dataset, int col_class)
: Classification(dataset, col_class) {
    m_k = k;
    int dim = dataset->get_dim() - 1;
    int num_pts = dataset->get_n_samples();
    m_data_pts = annAllocPts(num_pts, dim);

    for (int i = 0; i < num_pts; ++i) {
        const std::vector<double> point = dataset->get_instance(i);
        for (int j = 0; j < dim; ++j) {
            m_data_pts[i][j] = point[j+1];
        }
    }

    m_kd_tree = new ANNkd_tree(m_data_pts, num_pts, dim);
}

KnnClassification::~KnnClassification() {
    annDeallocPts(m_data_pts);
    delete m_kd_tree;
}
int KnnClassification::estimate(const ANNpoint &x, double threshold) const {
    ANNdistArray dist = new ANNdist[m_k];
    ANNidxArray idx = new ANNidx[m_k];
    m_kd_tree->annkSearch(x, m_k, idx, dist);
    
    double s = 0;
    for (int i = 0; i < m_k; ++i) {
        float temp = m_dataset->get_instance(idx[i])[m_col_class];
        s += temp;
    }

    return s > m_k * threshold;
}

int KnnClassification::get_k() const {
    return m_k;
}

ANNkd_tree *KnnClassification::get_kd_tree() {
    return m_kd_tree;
}

const ANNpointArray KnnClassification::get_points() const {
    return m_data_pts;
}
