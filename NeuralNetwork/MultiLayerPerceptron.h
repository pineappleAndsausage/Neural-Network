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

	vector<LayerPerceptron> m_layers;

protected:	

	//actual_output_set : output at each layer
	void update(const Output &desired_ouput, const vector<Output> &actual_output_set, const Input &input);
	vector<Output> feedforward(const Input &input);
	
public:
	MultiLayerPerceptron(void);
	virtual ~MultiLayerPerceptron(void);
	/*n_input : # of input, n_layers : # of each layer, n_loop : # of loop, 
	func_type 1 : sigmoid function for output layer, 2 : linear function for output layer*/
	void init(int n_input, const std::vector<int> &n_layers,  int n_output, int n_loop = 1, double learning_rate = 0.1, int func_type = 1);	
	void learning(const vector<Input> &input_set, const vector<Output> &output_set);
	Output run(const Input &input);
};
}

