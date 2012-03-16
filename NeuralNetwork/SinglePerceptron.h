#pragma once
#include <vector>
#include <memory>
#include "Neuron.h"


namespace af
{

class SinglePerceptron : public Neuron
{
public:
	friend class LayerPerceptron;

protected:
	tr1::shared_ptr<ActivationFunction> m_func;
	bool m_init;
	vector<double> m_weights;	
	double m_bias_unit;
	double m_learning_rate;	
	int m_loop_cnt;	

protected:
	int get_weight_size() { return m_weights.size(); }
	void delta_rule(double diff, double output, const Input &input);
	Input backpropagate(double input);	
	virtual double calc(const Input &input, const vector<double> &weight);	
	virtual void update(double diff,  double output, const Input &input);

public:
	SinglePerceptron();
	virtual ~SinglePerceptron();
	//n_input : # of input, n_loop : # of loop
	//func_type 1 : sigmoid function , 2 : linear function
	void init(int n_input,shared_ptr<ActivationFunction> func, int n_loop = 1, double learning_rate = 0.1);	
	void learning(const vector<Input> &input_set, const vector<double> &output_set);	
	double run(const Input &input);
	
};
}
