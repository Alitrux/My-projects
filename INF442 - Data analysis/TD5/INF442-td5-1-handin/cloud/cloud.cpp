#include <cassert> // for assert
#include <cfloat> // for DBL_MAX
#include <algorithm> // for sort
#include <cmath> // for sqrt
#include <iostream> // for cout

#include <cloud.hpp>

cloud::cloud(int _d, int _nmax) {
	point::set_dim(_d);

	nmax = _nmax;
	n = 0;

	points = new point[nmax];
}

cloud::~cloud() {
	delete[] points;
}

int cloud::get_n() const {
	return n;
}

point& cloud::get_point(int i) {
	return points[i];
}

// TODO 2.2.2: implement double cloud::k_dist_knn - distance to k-th nearest neighbor
double cloud::k_dist_knn(const point& p, int k) const {
	assert(k <= n);

	double sortedPoints[n];
	for (int i = 0; i < n; ++i) {
		sortedPoints[i] = p.dist(points[i]);
	}
	std::sort(sortedPoints, sortedPoints + n);
	return sortedPoints[k-1];
}

// TODO 2.2.2 if you wish to implement the Optional Exercise 3: return k nearest neighbors
point* cloud::knn(const point& p, int k) const {
	assert(k <= n);

	point *neighbors = new point[k];
	return neighbors;
}

// TODO 3 - Optional: implement one iteration of meanshift
point* cloud::shift(int k)
{
	point* Q = new point[n];

	return Q;
}

void cloud::meanshift(int nsteps, int k, bool verbose) {
	for (int r = 1; r <= nsteps; r++) {
		if (verbose)
			std::cerr << "Step " << r << ":" << std::endl;
		point *Q = shift(k);
		for(int i = 0; i < n; i++)
			for(int m = 0; m < point::get_dim(); m++)
				get_point(i).coords[m] = Q[i].coords[m];
		std::cerr << "\r           \r   done." << std::endl;
	}

	if (verbose) {
		for (int i = 0; i < n; i++) {
			for(int m = 0; m < point::get_dim(); m++)
				std::cout << get_point(i).coords[m] << ' ';
			std::cout << std::endl;
		}
	}
}

// Add to a cloud point by point; for tests only
void cloud::add_point(point& p)
{
	assert(n < nmax);

	for(int m = 0; m < point::get_dim(); m++) {
		points[n].coords[m] = p.coords[m];
	}

	n++;
}

// Read a file into a cloud; for tests only
void cloud::load(std::ifstream& is) {
	assert(is.is_open());

	// point to read into
	point p;
	p.label = 0;

	// while not at end of file
	while(is.peek() != EOF) {
		// read new points
		for(int m = 0; m < point::get_dim(); m++) {
			is >> p.coords[m];
		}

		add_point(p);

		// consume \n
		is.get();
	}
}

double cloud::standard_deviation() {
	point mean;
	for(int m = 0; m < point::get_dim(); m++) {
		for(int i = 0; i < n; i++)
			mean.coords[m] += points[i].coords[m];
		mean.coords[m] /= n;
	}

	double sum = 0.0;
	for(int i = 0; i < n; i++)
		sum += mean.dist(points[i]) * mean.dist(points[i]);

	return std::sqrt(sum / (n - 1));
}

// Min value for a given coordinate; was used to plot using gtkmm; deprecated
double cloud::min_in_coord(int m) {
	double minval = DBL_MAX;

	for(int i = 0; i < n; i++)
		if(points[i].coords[m] < minval)
			minval = points[i].coords[m];

	return minval;
}

// Max value for a given coordinate; was used to plot using gtkmm; deprecated
double cloud::max_in_coord(int m) {
	double maxval = DBL_MIN;

	for(int i = 0; i < n; i++)
		if(points[i].coords[m] > maxval)
			maxval = points[i].coords[m];

	return maxval;
}
