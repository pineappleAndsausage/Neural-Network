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
	int m_func_type;

protected:
	void delta_rule(double diff, double output, const std::vector<double> &input);
	std::vector<double> backpropagate(double input);	
	virtual double calc(const std::vector<double> &input, const std::vector<double> &weight);	
	virtual void update(double diff,  double output, const std::vector<double> &input);

public:
	SinglePerceptron();
	virtual ~SinglePerceptron();
	//func_type 1 : sigmoid function , 2 : linear function
	void init(int n_input, int n_loop = 1, double learning_rate = 0.1, int func_type = 1);	
	void learning(const std::vector<std::vector<double>> &input, const std::vector<double> &output);	
	double run(const std::vector<double> &input);
	
};
}
