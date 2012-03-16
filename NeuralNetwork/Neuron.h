#pragma once
#include <vector>
#include <iostream>
namespace af
{
	

typedef std::vector<double> Input;
typedef std::vector<double> Output;

using namespace std;

class ActivationFunction
{

public:	
	
	virtual double operator()(double h) const = 0;
	virtual double operator()(double input, double output) const = 0;
};

class Sigmoid: public ActivationFunction
{
public:
	Sigmoid() {};	
	double operator()(double h) const { return  1.0/ (1.0 + exp(-h));};
	double operator()(double input, double output) const
	{
		// w' = w + learning_rate * error_signal * dSigmoid(e) * input[i];
		// output = sigmoid(e), dSigmoid = output * ( 1- output);		
		return (output) * (1 - output) * input;
	};
};
class LinearFunction: public ActivationFunction
{
public:
	LinearFunction() {};	
	double operator()(double h) const { return  h;};
	double operator()(double input, double output) const
	{
		return input;
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

