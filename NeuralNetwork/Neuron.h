#pragma once
#include <vector>
namespace af
{
	

typedef std::vector<double> Input;
typedef std::vector<double> Output;

using namespace std;

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

