#include "SinglePerceptron.h"
#include <iostream>

af::SinglePerceptron::SinglePerceptron(void) : m_init(false)
{
	
}


af::SinglePerceptron::~SinglePerceptron(void)
{
}

//func_type = 1 : sigmoid, 2 : linear
void af::SinglePerceptron::init(int n_input, int n_loop, double learning_rate, int func_type)
{	
	m_init = true;
	m_loop_cnt = n_loop;
	m_learning_rate = learning_rate;
	m_func_type = func_type;

	//setWeight
	m_weights.resize(n_input);
	for(int i = 0; i < n_input; i++)
	{
		m_weights[i] = gaussianRandom();
	}
	m_bias_unit = gaussianRandom();
}

void af::SinglePerceptron::delta_rule(double diff, double output, const std::vector<double> &input)
{	
	//update weights
	for(int i = 0; i < (int)input.size(); i++)
	{		
		// w' = w + learning_rate * error_signal * dSigmoid(e) * input[i];
		// output = sigmoid(e), dSigmoid = output * ( 1- output);		
		if(m_func_type == 1)
			m_weights[i] += m_learning_rate * diff * (output) * (1 - output) * input[i];
		else
			m_weights[i] += m_learning_rate * diff * input[i];
	}
	m_bias_unit += m_learning_rate * diff;
}

double af::SinglePerceptron::run(const std::vector<double> &input)
{
	if(!m_init)
	{
		std::cout << "Has not been initialize" << std::endl;
		return -1;
	}
	return calc(input,m_weights);	
}

void af::SinglePerceptron::learning(const std::vector<std::vector<double>> &input, const std::vector<double> &output)
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
		double actual_output = run(input[i]);				
		update(output[i]-actual_output,actual_output, input[i]);
	}
}
std::vector<double> af::SinglePerceptron::backpropagate(double input)
{
	std::vector<double> result;
	for(int i = 0; i < (int) m_weights.size(); i++)
		result.push_back(m_weights[i]*input);
	return result; 
}
double af::SinglePerceptron::calc(const std::vector<double> &input, const std::vector<double> &weight)
{	
	double sum = 0.0;
	for(int i = 0; i < (int)input.size(); i++)
	{
		sum += input[i] * m_weights[i];
	}
	sum += m_bias_unit;
	if(m_func_type == 1)
		return sigmoid(sum);	
	else
		return sum;	
}

void af::SinglePerceptron::update(double diff, double output, const std::vector<double> &input)
{
	delta_rule(diff,output, input);
}
