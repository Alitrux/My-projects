#include <cmath> // for pow, atan, should you need them

#include <point.hpp>
#include <flat.hpp>

// TODO 2.1.1: implement volume and profile
// HINT: pi = std::atan(1) * 4.0
double flat::volume() const {
	double dim = data->get_point(0).get_dim();
	double pi = std::atan(1) * 4.0;

	return std::pow(pi, dim/2) / std::tgamma(dim/2 + 1);
}

double flat::profile(double t) const {
	if (t > 1)
		return 0;
	return 1;
}
