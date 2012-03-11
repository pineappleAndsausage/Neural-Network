#include <iostream>
#include <fstream>
#include <vector>
#include "../NeuralNetwork/NeuralNetwork.h"

void main()
{	
	//or function 
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

	
	af::NeuralNetwork nn;
	nn.init(2,100);
	nn.learning(input,output);
	for(int i = 0; i < 4; i++)
		std::cout << nn.run(input[i]) << std::endl;

	getchar();
}
