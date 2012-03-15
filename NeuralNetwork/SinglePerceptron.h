#pragma once
#include <vector>
#include "Neuron.h"

namespace af
{
class SinglePerceptron : public Neuron
{
public:
	friend class LayerPerceptron;

protected:
	bool m_init;
	std::vector<double> m_weights;	
	double m_bias_unit;
	double m_learning_rate;	
	int m_loop_cnt;

protected:
	void delta_rule(double diff, double output, const std::vector<double> &input);
	std::vector<double> backpropagate(double input);	
	virtual double calc(const std::vector<double> &input, const std::vector<double> &weight);
	virtual void update2(double desired_ouput, double actual_output, const std::vector<double> &input);
	virtual void update(double diff,  double output, const std::vector<double> &input);

public:
	SinglePerceptron();
	virtual ~SinglePerceptron();
	void init(int n_input, int n_loop = 1, double learning_rate = 0.1);	
	void learning(const std::vector<std::vector<double>> &input, const std::vector<double> &output);	
	double run(const std::vector<double> &input);
	
};
}
