#include "SinglePerceptron.h"
#include <iostream>

af::SinglePerceptron::SinglePerceptron(void)
{
	m_init = false;
}


af::SinglePerceptron::~SinglePerceptron(void)
{
}

double af::SinglePerceptron::gaussianRandom()
{
	static int have_deviate = 0;
	static double g1, g2;

	if (have_deviate) {
		
		have_deviate = 0;
		return g2;
	}
	else
	{
		double x, y;


		double z = 1.0 ;

		while ( z >= 1.0 ) {
			x = 2.0*(double(rand()) / RAND_MAX) - 1.0;
			y = 2.0*(double(rand()) / RAND_MAX) - 1.0;
			z = x * x + y * y;
		}

		z = sqrt( -2.0 * log( z ) / z );

		g1 = x * z;
		g2 = y * z;
		have_deviate = 1;

		return g1;
	}
}

void af::SinglePerceptron::init(int n_input, int n_loop, double learning_rate)
{	
	m_init = true;
	m_loop_cnt = n_loop;
	m_learning_rate = learning_rate;

	//setWeight
	m_weights.resize(n_input);
	for(int i = 0; i < n_input; i++)
	{
		m_weights[i] = gaussianRandom();
	}
	m_bias_unit = gaussianRandom();
}
void af::SinglePerceptron::adapts(double desired_ouput, double actual_output, const std::vector<double> &input)
{	
	//update weights
	for(int i = 0; i < (int)input.size(); i++)
	{
		m_weights[i] += m_learning_rate * (desired_ouput - actual_output) * input[i];
	}
	m_bias_unit += m_learning_rate * (desired_ouput - actual_output);
}

double af::SinglePerceptron::run(const std::vector<double> &input)
{
	if(!m_init)
	{
		std::cout << "Has not been initialize" << std::endl;
		return -1;
	}

	//linear threshold unit
	double sum = 0.0;
	for(int i = 0; i < (int)input.size(); i++)
	{
		sum += input[i] * m_weights[i];
	}
	sum += m_bias_unit;
	if(sum > 0)
		return 1;
	else
		return 0;
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
		double actural_output = run(input[i]);
		adapts(output[i],actural_output, input[i]);
	}
}
