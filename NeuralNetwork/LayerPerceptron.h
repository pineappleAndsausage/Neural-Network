#pragma once
#include <vector>
#include "Neuron.h"
#include "SinglePerceptron.h"
namespace af
{

class LayerPerceptron
{
public:
	friend class MultiLayerPerceptron;

protected:
	bool m_init;	
	double m_bias_unit;
	double m_learning_rate;	
	int m_loop_cnt;	

	vector<SinglePerceptron> m_layer;
	
	int get_weight_size() { return m_layer.size() > 0 ? m_layer[0].get_weight_size() : 0; }
	Output feedforward(const Input &input);
	Input backpropagate(const Output &input);	
	void update(const Output &diff, const Output &actual_output, const Input &input);

public:
	LayerPerceptron(void);
	virtual ~LayerPerceptron(void);
	//n_input : # of input, n_loop : # of loop
	//func_type 1 : sigmoid function , 2 : linear function
	void init(int n_input, int n_output, shared_ptr<ActivationFunction> func, int n_loop = 1, double learning_rate = 0.1);	
	void learning(const vector<Input> &input_set, const vector<Output> &output_set);
	Output run(const Input &input);


};
}
