#include <iostream>
#include <ANN/ANN.h>
#include "KnnRegression.hpp"

KnnRegression::KnnRegression(int k, Dataset* dataset, int col_regr)
: Regression(dataset, col_regr) {
	m_k = k;
	int dim = dataset->get_dim() - 1;
    int num_pts = dataset->get_nbr_samples();
    m_dataPts = annAllocPts(num_pts, dim);
	for (int i = 0; i < num_pts; ++i) {
        const std::vector<double> point = dataset->get_instance(i);
        for (int j = 0; j < dim; ++j) {
            m_dataPts[i][j] = point[j + 1*(j>=col_regr)];
        }
    }
	m_kdTree = new ANNkd_tree(m_dataPts, num_pts, dim);
}

KnnRegression::~KnnRegression() {
	delete m_kdTree;
}

double KnnRegression::estimate(const Eigen::VectorXd & x) const {
	assert(x.size()==m_dataset->get_dim()-1);
	double coords[x.size()];
	for (int i = 0; i < x.size(); ++i) {
		coords[i] = x(i);
	}
	ANNdistArray dist = new ANNdist[m_k];
    ANNidxArray idx = new ANNidx[m_k];
    m_kdTree->annkSearch(coords, m_k, idx, dist);
    
    double s = 0;
    for (int i = 0; i < m_k; ++i) {
        float temp = m_dataset->get_instance(idx[i])[m_col_regr];
        s += temp;
    }

    return s/m_k;
}

int KnnRegression::get_k() const {
	return m_k;
}

ANNkd_tree* KnnRegression::get_kdTree() const {
	return m_kdTree;
}
