/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2021 Tomislav Maric, TU Darmstadt
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Description
    Utility functions data exchange between libtorch and OpenFOAM. 

SourceFiles
    torchFunctions.

\*---------------------------------------------------------------------------*/

#include <torch/torch.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>

class CustomDataset : public torch::data::Dataset<CustomDataset> 
{
private:
    std::vector<std::string> inputFiles;
    std::vector<std::string> outputFiles;
    int inputSize;
    int outputSize;

public:

    CustomDataset(const std::string& inputDir, const std::string& outputDir, int numFiles, int input, int output)
    {
    	inputSize = input;
    	outputSize = output;
    	
        for (int i = 1; i <= numFiles; ++i) 
        {
            inputFiles.push_back(inputDir + "/input_" + std::to_string(i));
            outputFiles.push_back(outputDir + "/joint_" + std::to_string(i));
        }
    }


    torch::Tensor loadInput(const std::string& filename) 
    {
        std::ifstream file(filename);
        std::vector<float> data(inputSize); 
        if (file.is_open()) 
        {
            for (int i = 0; i < inputSize; ++i) 
            {
                file >> data[i];
            }
        }

        return torch::from_blob(data.data(), {inputSize,1}, torch::kFloat).clone();
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
    loadOutput(const std::string& filename) 
    {
        std::ifstream file(filename);
        std::vector<float> labels(outputSize);   
        std::vector<float> Ux(outputSize);   
        std::vector<float> Uy(outputSize);   
        std::vector<float> col2(outputSize);    
        std::vector<float> col3(outputSize);  
        std::vector<float> inf1(outputSize);   
        std::vector<float> inf2(outputSize);      
        if (file.is_open()) 
        {
            for (int i = 0; i < outputSize; ++i) 
            {
                file >> labels[i] >> Ux[i] >> Uy[i] >> col2[i] >> col3[i]>> inf1[i] >> inf2[i];
            }
        }

        auto labelTensor = torch::from_blob(labels.data(), {outputSize,1}, torch::kFloat).clone();
        auto UxTensor = torch::from_blob(Ux.data(), {outputSize,1}, torch::kFloat).clone();
        auto UyTensor = torch::from_blob(Uy.data(), {outputSize,1}, torch::kFloat).clone();
        auto col2Tensor = torch::from_blob(col2.data(), {outputSize,1}, torch::kFloat).clone();
        auto col3Tensor = torch::from_blob(col3.data(), {outputSize,1}, torch::kFloat).clone();
        auto inf1Tensor = torch::from_blob(inf1.data(), {outputSize,1}, torch::kFloat).clone();
        auto inf2Tensor = torch::from_blob(inf2.data(), {outputSize,1}, torch::kFloat).clone();
        return {labelTensor, UxTensor, UyTensor, col2Tensor, col3Tensor, inf1Tensor, inf2Tensor};
    } 



    torch::data::Example<> get(size_t index) override 
    {
        auto inputTensor = loadInput(inputFiles[index]);
        auto [labelTensor, UxTensor, UyTensor, col2Tensor, col3Tensor, inf1Tensor, inf2Tensor] 
             = loadOutput(outputFiles[index]);

        auto combinedTarget = torch::cat({labelTensor, UxTensor, UyTensor, col2Tensor, col3Tensor, inf1Tensor, inf2Tensor}, 
        /*dim=*/1);
        return {inputTensor, combinedTarget};
    }


    torch::optional<size_t> size() const override 
    {
        return inputFiles.size();
    }
};
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


// ************************************************************************* //
