#include "LayerPerceptron.h"
#include <iostream>

using namespace std;

af::LayerPerceptron::LayerPerceptron(void) : m_init(false)
{
}


af::LayerPerceptron::~LayerPerceptron(void)
{
}
void af::LayerPerceptron::init(int n_input, int n_output, int n_loop, double learning_rate, int func_type)
{
	m_init = true;
	m_loop_cnt = n_loop;
	m_learning_rate = learning_rate;
	
	m_layer.resize(n_output);

	for(int i = 0; i < (int)m_layer.size(); i++)
	{		
		m_layer[i].init(n_input,1,learning_rate,func_type);		
	}
}

void af::LayerPerceptron::learning(const vector<Input> &input_set, const vector<Output> &output_set)
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
		Output actual_output = feedforward(input_set[i]);
		Output diff(output_set[i].size());
		for(int j = 0; j < (int)output_set[i].size(); j++)
			diff[j] = output_set[i][j] - actual_output[j];		
		update(diff,actual_output,input_set[i]);		
	}

}

af::Output af::LayerPerceptron::feedforward(const Input &input)
{
	Output actual_output(m_layer.size());
	
	for(int i = 0; i < (int)m_layer.size(); i++)
	{
		actual_output[i] = m_layer[i].run(input);		
	}	
	
	return actual_output;
}

af::Input af::LayerPerceptron::backpropagate(const Output &input)
{	
	Input actual_output(get_weight_size());	
	for(int i = 0; i < (int)input.size(); i++)
	{
		Input back = m_layer[i].backpropagate(input[i]);

		for(int j = 0; j < (int)back.size(); j++)
		{
			if(j == 0)
				actual_output[j] = back[j];
			else
				actual_output[j] += back[j];
		}
	}
	return actual_output;
}

void af::LayerPerceptron::update(const Output &diff,const Output &output,  const Input &input)
{
	for(int i = 0; i < (int)diff.size(); i++)
		m_layer[i].update(diff[i],output[i], input);
}

af::Output af::LayerPerceptron::run(const Input &input)
{
	return feedforward(input);
}