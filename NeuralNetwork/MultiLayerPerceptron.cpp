#include "MultiLayerPerceptron.h"
#include <iostream>

using namespace std;
template<class ActivationFunction>
af::MultiLayerPerceptron<ActivationFunction>::MultiLayerPerceptron(void) : m_init(false)
{
}

template<class ActivationFunction>
af::MultiLayerPerceptron<ActivationFunction>::~MultiLayerPerceptron(void)
{
}

//func_type of output layer 
template<class ActivationFunction>
void af::MultiLayerPerceptron<ActivationFunction>::init(int n_input, const std::vector<int> &n_layers, int n_output, int n_loop, double learning_rate)
{
	m_init = true;
	m_loop_cnt = n_loop;
	m_learning_rate = learning_rate;

	std::vector<int>layers;
	layers.push_back(n_input);
	for(int i = 0; i < (int)n_layers.size(); i++)
	{
		layers.push_back(n_layers[i]);
	}
	layers.push_back(n_output);

	//layer 0 : input, layer n-1 : output
	m_layers.resize(layers.size() - 1);
	for(int i = 0; i < (int)layers.size() - 1; i++)
	{	
		if(i == layers.size() - 2)
			m_layers[i].init(layers[i],layers[i+1],1,learning_rate);		
		else
			m_layers[i].init(layers[i],layers[i+1],1,learning_rate);		
	}
}
template<class ActivationFunction>
void af::MultiLayerPerceptron<ActivationFunction>::learning(const vector<Input> &input_set, const vector<Output> &output_set)
{
	if(!m_init)
	{
		cout << "Has not been initialize" << endl;
		return;
	}
	
	//training	
	for(int k = 0; k < m_loop_cnt; k++)
	for(int i = 0; i < (int)input_set.size(); i++)
	{
		//feedforward		
		vector<Output> actual_outputs = feedforward(input_set[i]);

		//backpropagation and update
		update(output_set[i],actual_outputs,input_set[i]);		
	}
}
template<class ActivationFunction>
af::vector<af::Output> af::MultiLayerPerceptron<ActivationFunction>::feedforward(const Input &input)
{	
	Input inner_input(input.begin(),input.end());
	vector<Output> inner_output_set;
	inner_output_set.push_back(inner_input);
	for(int i = 0; i < (int)m_layers.size(); i++)
	{
		inner_input = m_layers[i].feedforward(inner_input);
		inner_output_set.push_back(inner_input);
	}
	return inner_output_set;
}
template<class ActivationFunction>
void af::MultiLayerPerceptron<ActivationFunction>::update(const Output &desired_output, const vector<Output> &actual_output_set, const Input &input)
{
	Output error_signal(desired_output.size());
	int n = actual_output_set.size() - 1;
	for(int i = 0; i < (int)error_signal.size(); i++)
	{
		error_signal[i] = desired_output[i] - actual_output_set[n][i];
	}
	
	for(int i = (int)m_layers.size() - 1; i >= 0; i--)
	{		
		Input back =  m_layers[i].backpropagate(error_signal);
		m_layers[i].update(error_signal,actual_output_set[i+1],actual_output_set[i]);
		error_signal = back;		
	}
}
template<class ActivationFunction>
af::Output af::MultiLayerPerceptron<ActivationFunction>::run(const Input &input)
{<
	vector<Output> result = feedforward(input);
	return result[result.size() - 1];
}