#include "MultiLayerPerceptron.h"
#include <iostream>

af::MultiLayerPerceptron::MultiLayerPerceptron(void) : m_init(false)
{
}


af::MultiLayerPerceptron::~MultiLayerPerceptron(void)
{
}

void af::MultiLayerPerceptron::init(int n_input, const std::vector<int> &n_layers, int n_output, int n_loop, double learning_rate)
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
		m_layers[i].init(layers[i],layers[i+1],1,learning_rate);		
	}
}

void af::MultiLayerPerceptron::learning(const std::vector<std::vector<double>> &input, const std::vector<std::vector<double>> &output)
{
	if(!m_init)
	{
		std::cout << "Has not been initialize" << std::endl;
		return;
	}
	
	//training	
	for(int k = 0; k < m_loop_cnt; k++)
	for(int i = 0; i < (int)input.size(); i++)
	{
		//feedforward		
		std::vector<std::vector<double>> actual_outputs = feedforward(input[i]);

		//backpropagation and update
		update(output[i],actual_outputs,input[i]);		
	}
}
std::vector<std::vector<double>> af::MultiLayerPerceptron::feedforward(const std::vector<double> &input)
{	
	std::vector<double> inner_input(input.begin(),input.end());
	std::vector<std::vector<double>> inner_output;
	inner_output.push_back(inner_input);
	for(int i = 0; i < (int)m_layers.size(); i++)
	{
		inner_input = m_layers[i].feedforward(inner_input);
		inner_output.push_back(inner_input);
	}
	return inner_output;
}

void af::MultiLayerPerceptron::update(const std::vector<double> &desired_output, const std::vector<std::vector<double>> &actual_output, const std::vector<double> &input)
{
	std::vector<double> error_signal(desired_output.size());
	int n = actual_output.size() - 1;
	for(int i = 0; i < (int)error_signal.size(); i++)
	{
		error_signal[i] = desired_output[i] - actual_output[n][i];
	}
	
	for(int i = (int)m_layers.size() - 1; i >= 0; i--)
	{		
		std::vector<double> back =  m_layers[i].backpropagate(error_signal);
		m_layers[i].update(error_signal,actual_output[i+1],actual_output[i]);
		error_signal = back;		
	}
}

std::vector<double> af::MultiLayerPerceptron::run(const std::vector<double> &input)
{
	std::vector<std::vector<double>> result = feedforward(input);
	return result[result.size() - 1];
}