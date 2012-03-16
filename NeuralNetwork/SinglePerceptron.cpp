#include "SinglePerceptron.h"
#include <iostream>

using namespace std;

af::SinglePerceptron::SinglePerceptron(void) : m_func(NULL),  m_init(false)
{
	
}


af::SinglePerceptron::~SinglePerceptron(void)
{
}

//func_type = 1 : sigmoid, 2 : linear
void af::SinglePerceptron::init(int n_input, shared_ptr<ActivationFunction> func, int n_loop, double learning_rate)
{	
	if(func == NULL) throw "Activation function is NULL";
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

void af::SinglePerceptron::delta_rule(double diff, double output, const Input &input)
{	
	//update weights
	for(int i = 0; i < (int)input.size(); i++)
	{	
		m_weights[i] +=  m_learning_rate * (*m_func)(diff,input[i],output);
	}
	m_bias_unit += m_learning_rate * diff;
}

double af::SinglePerceptron::run(const Input &input)
{
	if(!m_init) throw "Has not been initialize";
		
	return calc(input,m_weights);	
}

void af::SinglePerceptron::learning(const vector<Input> &input_set, const vector<double> &output_set)
{
	if(!m_init) throw "Has not been initialize";
	
	//training
	for(int k = 0; k < m_loop_cnt; k++)
	for(int i = 0; i < (int)input_set.size(); i++)
	{
		double actual_output = run(input_set[i]);				
		update(output_set[i]-actual_output,actual_output, input_set[i]);
	}
}
af::Input af::SinglePerceptron::backpropagate(double input)
{
	std::vector<double> result;
	for(int i = 0; i < (int) m_weights.size(); i++)
		result.push_back(m_weights[i]*input);
	return result; 
}
double af::SinglePerceptron::calc(const Input &input, const std::vector<double> &weight)
{	
	double sum = 0.0;
	for(int i = 0; i < (int)input.size(); i++)
	{
		sum += input[i] * m_weights[i];
	}
	sum += m_bias_unit;
	return (*m_func)(sum);
}

void af::SinglePerceptron::update(double diff, double output, const Input &input)
{
	delta_rule(diff,output, input);
}
