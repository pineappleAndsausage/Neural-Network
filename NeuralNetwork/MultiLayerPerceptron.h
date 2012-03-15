#pragma once
#include "LayerPerceptron.h"
#include "Neuron.h"
#include <vector>

namespace af
{
class MultiLayerPerceptron 
{	
protected:
	bool m_init;		
	double m_learning_rate;	
	int m_loop_cnt;

	std::vector<LayerPerceptron> m_layers;

protected:	

	void update(const std::vector<double> &desired_ouput, const std::vector<std::vector<double>> &actual_output, const std::vector<double> &input);
	std::vector<std::vector<double>> feedforward(const std::vector<double> &input);
	
public:
	MultiLayerPerceptron(void);
	virtual ~MultiLayerPerceptron(void);

	void init(int n_input, const std::vector<int> &layer,  int n_output, int n_loop = 1, double learning_rate = 0.1);	
	void learning(const std::vector<std::vector<double>> &input, const std::vector<std::vector<double>> &output);
	std::vector<double> run(const std::vector<double> &input);
};
}

