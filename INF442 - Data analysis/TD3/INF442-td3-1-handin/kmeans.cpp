#include <iostream>
#include <cassert>
#include <cmath>	// for sqrt, fabs
#include <cfloat>	// for DBL_MAX
#include <cstdlib>	// for rand, srand
#include <ctime>	// for rand seed
#include <fstream>
#include <cstdio>	// for EOF
#include <string>
#include <algorithm>	// for count
#include <vector>

using std::rand;
using std::srand;
using std::time;

class point
{
    public:

        static int d;
        double *coords;
        int label;

        point() {
        	coords = new double[d] {0.0};
        	label = 0;
        }
        ~point() {delete[] coords;}

        void print() const {
        	for (int i=0; i < d-1; ++i) {
        		std::cout <<coords[i] <<"\t";
        	}
        	std::cout <<coords[d-1] <<"\n";
        }
        double squared_dist(const point &q) const {
        	double sum = 0;
        	for (int i = 0; i < d; ++i) {
        		sum += (this->coords[i] - q.coords[i])*(this->coords[i] - q.coords[i]);
        	}
        	return sum;
        }
};

int point::d;

class cloud
{
	private:

	int d;
	int n;
	int k;

	// maximum possible number of points
	int nmax;

	point *points;
	point *centers;


	public:

	cloud(int _d, int _nmax, int _k)
	{
		d = _d;
		point::d = _d;
		n = 0;
		k = _k;

		nmax = _nmax;

		points = new point[nmax];
		centers = new point[k];

		srand(time(0));
	}

	~cloud()
	{
		delete[] centers;
		delete[] points;
	}

	void add_point(const point &p, int label)
	{
		for(int m = 0; m < d; m++)
		{
			points[n].coords[m] = p.coords[m];
		}

		points[n].label = label;

		n++;
	}

	int get_d() const
	{
		return d;
	}

	int get_n() const
	{
		return n;
	}

	int get_k() const
	{
		return k;
	}

	point &get_point(int i)
	{
		return points[i];
	}

	point &get_center(int j)
	{
		return centers[j];
	}

	void set_center(const point &p, int j)
	{
		for(int m = 0; m < d; m++)
			centers[j].coords[m] = p.coords[m];
	}

	double intracluster_variance() const
	{
		double sum {0.0};
		for (int i = 0; i < this->n; ++i) {
			sum += this->points[i].squared_dist(this->centers[this->points[i].label]);
		}

		return sum / this->n;
	}

	int set_voronoi_labels()
	{
		int changedPoints {0};
		for (int i = 0; i < this->n; ++i) {
			double min = this->points[i].squared_dist(this->centers[0]);
			int minIndex {0};
			for (int j = 1; j < this->k; ++j) {
				if (this->points[i].squared_dist(this->centers[j]) < min) {
					min = this->points[i].squared_dist(this->centers[j]);
					minIndex = j;
				}
			}
			if (minIndex != this->points[i].label) {
				this->points[i].label = minIndex;
				++changedPoints;
			}
		}
		return changedPoints;
	}

	void set_centroid_centers()
	{
		int clusterSize[this->k] {0};
		for (int i = 0; i < this->n; ++i) {
			int clusterNumber = this->points[i].label;
			if (clusterSize[clusterNumber] == 0) {
				for (int j = 0; j < this->d; ++j) {
					this->centers[clusterNumber].coords[j] = 0.0;
				}
			}
			++clusterSize[clusterNumber];
			for (int j = 0; j < this->d; ++j) {
				this->centers[clusterNumber].coords[j] += this->points[i].coords[j];
			}
		}

		for (int i = 0; i < k; ++i) {
			if (clusterSize[i] != 0) {
				for (int j = 0; j < this->d; ++j) {
					this->centers[i].coords[j] /= clusterSize[i];
				}
			}
		}

	}

	void init_random_partition()
	{
		for (int i = 0; i < this->n; ++i) {
			this->points[i].label = rand() % this->k;
		}
		set_centroid_centers();
	}

	void lloyd()
	{
		init_random_partition();
		while (set_voronoi_labels() != 0) {
			set_centroid_centers();
		}

	}

	void init_forgy()
	{
		int centerNumbers[k] = {0};
		for (int i = 0; i < k; ++i) {
			int randomCenter;
			while(1) {
				 randomCenter = rand() % n;
				 bool flag = true;
				 for (int j = 0; j < i; ++j) {
				 	if (centerNumbers[j] == randomCenter) {
				 		flag = false;
				 	}
				 }
				 if (flag)
				 	break;
			}
			centers[i] = points[randomCenter];
			centerNumbers[i] = randomCenter;
		}
	}

	void init_plusplus()
	{
		centers[0] = points[rand() % n];
		double d[n]; //An array containing the distances to C squared
		double sum = 0; //the sum of the distances to C squared
		for (int i = 0; i < n; ++i) {
			d[i] = centers[0].squared_dist(points[i]);
			sum += d[i];
		}


		for (int i = 1; i < k; ++i)
		{
			//updating distances
			for (int j = 0; j < n; ++j) {
				if (centers[i-1].squared_dist(points[j]) < d[j]) {
					sum -= d[j] + centers[i-1].squared_dist(points[j]);
					d[j] = centers[i-1].squared_dist(points[j]);
				}
			}
			double randomNumber = (double)rand() / RAND_MAX;
			int searchIndex = 0;
			double search = 0.0;
			while (searchIndex < n) {
				search += d[searchIndex]/sum;
				if (search > randomNumber) {
					centers[i] = points[searchIndex];
					break;
				}
				++searchIndex;
			}

		}
	}
};