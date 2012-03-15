#include <iostream>
#include <fstream>
#include <vector>
#include "../NeuralNetwork/NeuralNetwork.h"

void test_xor();
void test_multilayer();

void main()
{	
	

	test_xor();
	//test_multilayer();

	getchar();
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
	af::MultiLayerPerceptron nn;
	std::vector<int> n_layers;
	n_layers.push_back(20);
	nn.init(3,n_layers,1,100);
	nn.learning(input,output);
	for(int i = 0; i < (int)input.size(); i++)
		std::cout << nn.run(input[i])[0] - output[i][0] << std::endl;

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

	output[0] = 0; output[1] = 1; output[2] = 1; output[3] = 0;

	if(false)
	{
		af::SinglePerceptron nn;
		nn.init(2,100000);
		nn.learning(input,output);
		std::cout << "single perceptron" << std::endl;
		for(int i = 0; i < 4; i++)
			std::cout << input[i][0] << " " << input[i][1] << " " << nn.run(input[i]) << std::endl;
	}
	if(false)
	{
		af::LayerPerceptron nn;
		nn.init(2,1,10000);
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
		n_layers.push_back(3);		
		nn.init(2,n_layers,1,100000);
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

