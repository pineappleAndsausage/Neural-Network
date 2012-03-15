#include "Neuron.h"


af::Neuron::Neuron(void)
{
}


af::Neuron::~Neuron(void)
{
}
double af::Neuron::dSigmoid(double h)
{
	double t = sigmoid(h);
	return t*(1-t);
}
double af::Neuron::sigmoid(double h)
{
	return 1.0/ (1.0 + exp(-h));
}
double af::Neuron::gaussianRandom()
{
	static int have_deviate = 0;
	static double g1, g2;

	if (have_deviate) {
		
		have_deviate = 0;
		return g2;
	}
	else
	{
		double x, y;


		double z = 1.0 ;

		while ( z >= 1.0 ) {
			x = 2.0*(double(rand()) / RAND_MAX) - 1.0;
			y = 2.0*(double(rand()) / RAND_MAX) - 1.0;
			z = x * x + y * y;
		}

		z = sqrt( -2.0 * log( z ) / z );

		g1 = x * z;
		g2 = y * z;
		have_deviate = 1;

		return g1;
	}
}

