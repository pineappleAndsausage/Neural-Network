#include "SinglePerceptron.h"
#include <iostream>

using namespace std;
template<class ActivationFunction>
af::SinglePerceptron<ActivationFunction>::SinglePerceptron(void) :  m_init(false)
{
	
}

template<class ActivationFunction>
af::SinglePerceptron<ActivationFunction>::~SinglePerceptron(void)
{
}

//func_type = 1 : sigmoid, 2 : linear
template<class ActivationFunction>
void af::SinglePerceptron<ActivationFunction>::init(int n_input, int n_loop, double learning_rate)
{	
	m_init = true;
	m_loop_cnt = n_loop;
	m_learning_rate = learning_rate;
	m_func = func;

	//setWeight
	m_weights.resize(n_input);
	for(int i = 0; i < n_input; i++)
	{
		m_weights[i] = gaussianRandom();
	}
	m_bias_unit = gaussianRandom();
}
template<class ActivationFunction>
void af::SinglePerceptron<ActivationFunction>::delta_rule(double diff, double output, const Input &input)
{	
	//update weights
	for(int i = 0; i < (int)input.size(); i++)
	{	
		m_weights[i] +=  m_learning_rate * m_func(diff,input[i],output);
	}
	m_bias_unit += m_learning_rate * diff;
}
template<class ActivationFunction>
double af::SinglePerceptron<ActivationFunction>::run(const Input &input)
{
	if(!m_init)
	{
		std::cout << "Has not been initialize" << std::endl;
		return -1;
	}
	return calc(input,m_weights);	
}
template<class ActivationFunction>
void af::SinglePerceptron<ActivationFunction>::learning(const vector<Input> &input_set, const vector<double> &output_set)
{
	if(!m_init)
	{
		std::cout << "Has not been initialize" << std::endl;
		return;
	}
	
	//training
	for(int k = 0; k < m_loop_cnt; k++)
	for(int i = 0; i < (int)input_set.size(); i++)
	{
		double actual_output = run(input_set[i]);				
		update(output_set[i]-actual_output,actual_output, input_set[i]);
	}
}
template<class ActivationFunction>
af::Input af::SinglePerceptron<ActivationFunction>::backpropagate(double input)
{
	std::vector<double> result;
	for(int i = 0; i < (int) m_weights.size(); i++)
		result.push_back(m_weights[i]*input);
	return result; 
}
template<class ActivationFunction>
double af::SinglePerceptron<ActivationFunction>::calc(const Input &input, const std::vector<double> &weight)
{	
	double sum = 0.0;
	for(int i = 0; i < (int)input.size(); i++)
	{
		sum += input[i] * m_weights[i];
	}
	sum += m_bias_unit;
	return m_func(sum);
}
template<class ActivationFunction>
void af::SinglePerceptron<ActivationFunction>::update(double diff, double output, const Input &input)
{
	delta_rule(diff,output, input);
}
