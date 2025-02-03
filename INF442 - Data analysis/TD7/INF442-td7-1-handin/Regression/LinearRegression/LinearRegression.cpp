#include<iostream>
#include<cassert>
#include "LinearRegression.hpp"
#include "Dataset.hpp"
#include "Regression.hpp"

LinearRegression::LinearRegression(Dataset* dataset, int col_regr) 
: Regression(dataset, col_regr) {
	m_beta = NULL;
	set_coefficients();
}

LinearRegression::~LinearRegression() {
	if (m_beta != NULL) {
		m_beta->resize(0);
		delete m_beta;
	}
}

Eigen::MatrixXd LinearRegression::construct_matrix() {
	
	Eigen::MatrixXd X(m_dataset->get_nbr_samples(), m_dataset->get_dim());

	for (int i = 0; i < m_dataset->get_nbr_samples(); i++) {
		X(i, 0) = 1;
		bool flag = false;
		for (int j = 0; j < m_dataset->get_dim(); j++) {
			if (j == m_col_regr) {
				flag = true;
				continue;
			}
			X(i, j + 1 - flag) = m_dataset->get_instance(i)[j];
		}
	}
	return X;
}

Eigen::VectorXd LinearRegression::construct_y() {
	Eigen::VectorXd y(m_dataset->get_nbr_samples());
	for (int i = 0; i < m_dataset->get_nbr_samples(); i++) {
		y(i) = m_dataset->get_instance(i)[m_col_regr];
	}
	return y;
}

void LinearRegression::set_coefficients() {
	Eigen::MatrixXd X = construct_matrix();
	Eigen::VectorXd y = construct_y();
	Eigen::MatrixXd tX = X.transpose();
	Eigen::MatrixXd tX_X = tX * X;
	Eigen::MatrixXd tX_X_inv = tX_X.inverse();
	Eigen::VectorXd coefficients = tX_X_inv * tX * y;
	m_beta = new Eigen::VectorXd(coefficients);
}

const Eigen::VectorXd* LinearRegression::get_coefficients() const {
	if (!m_beta) {
		std::cout <<"Coefficients have not been allocated." <<std::endl;
		return NULL;
	}
	return m_beta;
}

void LinearRegression::show_coefficients() const {
	if (!m_beta) {
		std::cout << "Coefficients have not been allocated." <<std::endl;
		return;
	}
	
	if (m_beta->size() != m_dataset->get_dim()) {  // ( beta_0 beta_1 ... beta_{d} )
		std::cout << "Warning, unexpected size of coefficients vector: " << m_beta->size() << std::endl;
	}
	
	std::cout<< "beta = (";
	for (int i=0; i<m_beta->size(); i++) {
		std::cout << " " << (*m_beta)[i];
	}
	std::cout << " )" <<std::endl;
}

void LinearRegression::print_raw_coefficients() const {
	std::cout<< "{ ";
	for (int i = 0; i < m_beta->size() - 1; i++) {
		std::cout << (*m_beta)[i] << ", ";
	}
	std::cout << (*m_beta)[m_beta->size() - 1];
	std::cout << " }" << std::endl;
}

void LinearRegression::sum_of_squares(Dataset* dataset, double& ess, double& rss, double& tss) const {
	assert(dataset->get_dim()==m_dataset->get_dim());
	
	Eigen::MatrixXd X(dataset->get_nbr_samples(), dataset->get_dim());
	for (int i = 0; i < dataset->get_nbr_samples(); i++) {
		X(i, 0) = 1;
		bool flag = false;
		for (int j = 0; j < dataset->get_dim(); j++) {
			if (j == m_col_regr) {
				flag = true;
				continue;
			}
			X(i, j + 1 - flag) = dataset->get_instance(i)[j];
		}
	}
	Eigen::VectorXd y(dataset->get_nbr_samples());
	for (int i = 0; i < dataset->get_nbr_samples(); i++) {
		y(i) = dataset->get_instance(i)[m_col_regr];
	}

	ess = 0;
	tss = 0;
	rss = 0;
	for (int i = 0; i < dataset->get_nbr_samples(); i++) {
		double y_hat = 0;
		for (int j = 0; j < dataset->get_dim(); j++) {
			y_hat += y(j) * X(i, j);
		}
		y_hat /= dataset->get_dim();
		ess += (y_hat - y(i)) * (y_hat - y(i));

		tss += (y(i) - estimate(X.row(i))) * (y(i) - estimate(X.row(i)));
	}
	rss = tss - ess;

}

double LinearRegression::estimate(const Eigen::VectorXd & x) const {
	double estimate = (*m_beta)[0];
	for (int i = 0; i < x.size(); i++) {
		estimate += x[i] * (*m_beta)[i + 1];
	}
	return estimate;
}
