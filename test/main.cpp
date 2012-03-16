#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include "../NeuralNetwork/NeuralNetwork.h"

void test_xor();
void test_multilayer();

int main()
{	
	
	try
	{
		test_xor();
		//test_multilayer();
	}
	catch(const char* err)
	{
		std::cerr << err << std::endl;
	}
	//getchar();
	return 0;
}

void test_multilayer()
{
	std::ifstream f("test.txt");
	int n;
	f >> n;
	std::vector<std::vector<double>> input(n);
	std::vector<std::vector<double>> output(n);
	for(int i = 0; i < n; i++)
	{
		input[i].resize(3);
		output[i].resize(1);
		int temp;
		f >> temp >> input[i][0] >> input[i][1] >> input[i][2] >> output[i][0];
	}
	double scale = 100.0;
	for(int i = 0; i < n; i++)
		output[i][0] /= scale;
	af::MultiLayerPerceptron nn;
	std::vector<int> n_layers;
	n_layers.push_back(20);
	n_layers.push_back(5);
	//n_layers.push_back(3);
	std::tr1::shared_ptr<af::ActivationFunction> afunc(new af::Sigmoid());
	nn.init(3,n_layers,1,afunc,1000,0.5);
	nn.learning(input,output);
	for(int i = 0; i < (int)input.size(); i++)
		std::cout << nn.run(input[i])[0] * scale - output[i][0] * scale << std::endl;

}
void test_xor()
{
	//xor function 
	std::vector<std::vector<double>> input(4);
	std::vector<double> output(4);
	for(int i = 0; i < (int)input.size(); i++)
	{
		input[i].resize(2);		
	}
	input[0][0] = 0; input[0][1] = 0;
	input[1][0] = 0; input[1][1] = 1;
	input[2][0] = 1; input[2][1] = 0;
	input[3][0] = 1; input[3][1] = 1;

	output[0] = 0; output[1] = 1; output[2] = 1; output[3] = 1;

	if(false)
	{
		af::SinglePerceptron nn;
		std::tr1::shared_ptr<af::ActivationFunction> afunc(new af::Sigmoid());
		nn.init(2,afunc,100000,0.1);
		nn.learning(input,output);
		std::cout << "single perceptron" << std::endl;
		for(int i = 0; i < 4; i++)
			std::cout << input[i][0] << " " << input[i][1] << " " << nn.run(input[i]) << std::endl;
	}
	if(false)
	{
		af::LayerPerceptron nn;
		std::tr1::shared_ptr<af::ActivationFunction> afunc(new af::Sigmoid());
		nn.init(2,1,afunc,10000,0.1);
		std::vector<std::vector<double>> output1;
		std::vector<double> temp1;
		std::vector<double> temp2;
		temp1.push_back(0); temp2.push_back(1);			
		output1.push_back(temp1); output1.push_back(temp2); output1.push_back(temp2); output1.push_back(temp1);
		nn.learning(input,output1);
		std::cout << "single layer perceptron" << std::endl;
		for(int i = 0; i < 4; i++)
			std::cout << input[i][0] << " " << input[i][1] << " " <<nn.run(input[i])[0] << std::endl;
	}
	//if(false)
	{
		af::MultiLayerPerceptron nn;
		std::vector<int> n_layers;
		n_layers.push_back(5);		
		std::tr1::shared_ptr<af::ActivationFunction> afunc(new af::Sigmoid());
		nn.init(2,n_layers,1,afunc,100000,0.2);		
		std::vector<std::vector<double>> output1;
		std::vector<double> temp1, temp2;		
		temp1.push_back(0); temp2.push_back(1);			
		output1.push_back(temp1); output1.push_back(temp2); output1.push_back(temp2); output1.push_back(temp1);
		nn.learning(input,output1);
		std::cout << "multi layer perceptron" << std::endl;
		for(int i = 0; i < 4; i++)
			std::cout << input[i][0] << " " << input[i][1] << " " << nn.run(input[i])[0] << std::endl;
	}
}

