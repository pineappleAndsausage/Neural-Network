#include "LayerPerceptron.h"
#include <iostream>

af::LayerPerceptron::LayerPerceptron(void) : m_init(false)
{
}


af::LayerPerceptron::~LayerPerceptron(void)
{
}
void af::LayerPerceptron::init(int n_input, int n_output, int n_loop, double learning_rate)
{
	m_init = true;
	m_loop_cnt = n_loop;
	m_learning_rate = learning_rate;
		
	m_layer.resize(n_output);

	for(int i = 0; i < (int)m_layer.size(); i++)
	{		
		m_layer[i].init(n_input,1,learning_rate);		
	}
}

void af::LayerPerceptron::learning(const std::vector<std::vector<double>> &input, const std::vector<std::vector<double>> &output)
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
		std::vector<double> actual_output = feedforward(input[i]);
		std::vector<double> diff(output[i].size());
		for(int j = 0; j < (int)output[i].size(); j++)
			diff[j] = output[i][j] - actual_output[j];		
		update(diff,actual_output,input[i]);		
	}

}

std::vector<double> af::LayerPerceptron::feedforward(const std::vector<double> &input)
{
	std::vector<double> actual_output(m_layer.size());
	
	for(int i = 0; i < (int)m_layer.size(); i++)
	{
		actual_output[i] = m_layer[i].run(input);		
	}	
	
	return actual_output;
}

std::vector<double> af::LayerPerceptron::backpropagate(const std::vector<double> &input)
{	
	std::vector<double> actual_output(m_layer[0].m_weights.size());	
	for(int i = 0; i < (int)input.size(); i++)
	{
		std::vector<double> temp = m_layer[i].backpropagate(input[i]);

		for(int j = 0; j < (int)temp.size(); j++)
		{
			if(j == 0)
				actual_output[j] = temp[j];
			else
				actual_output[j] += temp[j];
		}
	}
	return actual_output;
}

void af::LayerPerceptron::update2(const std::vector<double> &desired_ouput, const std::vector<double> &actual_output, const std::vector<double> &input)
{
	for(int i = 0; i < (int)desired_ouput.size(); i++)
		m_layer[i].update2(desired_ouput[i],actual_output[i],input);
}

void af::LayerPerceptron::update(const std::vector<double> &diff,const std::vector<double> &output,  const std::vector<double> &input)
{
	for(int i = 0; i < (int)diff.size(); i++)
		m_layer[i].update(diff[i],output[i], input);
}

std::vector<double> af::LayerPerceptron::run(const std::vector<double> &input)
{
	return feedforward(input);
}