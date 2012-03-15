#pragma once
#include <vector>
#include "Neuron.h"
#include "SinglePerceptron.h"
namespace af
{
class LayerPerceptron
{
public:
	friend class MultiLayerPerceptron;

protected:
	bool m_init;	
	double m_bias_unit;
	double m_learning_rate;	
	int m_loop_cnt;	

	std::vector<af::SinglePerceptron> m_layer;
	
	std::vector<double> feedforward(const std::vector<double> &input);
	std::vector<double> backpropagate(const std::vector<double> &input);	
	void update(const std::vector<double> &diff, const std::vector<double> &actual_output, const std::vector<double> &input);

public:
	LayerPerceptron(void);
	virtual ~LayerPerceptron(void);
	//func_type 1 : sigmoid function , 2 : linear function
	void init(int n_input, int n_output, int n_loop = 1, double learning_rate = 0.1, int func_type = 1);	
	void learning(const std::vector<std::vector<double>> &input, const std::vector<std::vector<double>> &output);
	std::vector<double> run(const std::vector<double> &input);


};
}
