#pragma once
#include <vector>
namespace af
{
	

typedef std::vector<double> Input;
typedef std::vector<double> Output;

using namespace std;

class Sigmoid
{
public:
	Sigmoid() {};
	double operator()(double h) const { return  1.0/ (1.0 + exp(-h));};
	double operator()(double diff, double input, double output) const
	{
		// w' = w + learning_rate * error_signal * dSigmoid(e) * input[i];
		// output = sigmoid(e), dSigmoid = output * ( 1- output);		
		return diff * (output) * (1 - output) * input;
	};
};
class LinearFunction
{
public:
	LinearFunction() {};
	double operator()(double h) const { return  h;};
	double operator()(double diff, double input, double output) const
	{
		return diff * input;
	}
};

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

