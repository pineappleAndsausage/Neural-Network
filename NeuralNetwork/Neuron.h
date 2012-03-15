#pragma once
#include <vector>
namespace af
{
class Neuron
{
public:
	Neuron(void);
	~Neuron(void);
protected:
	double gaussianRandom();
	double sigmoid(double h);
	double dSigmoid(double h);
};
}

