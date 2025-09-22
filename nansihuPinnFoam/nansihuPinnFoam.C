/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2022 Tomislav Maric, TU Darmstadt 
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

Application
    2dPinnFoam

Description

\*---------------------------------------------------------------------------*/


#include <torch/torch.h>
#include <torch/serialize.h>
#include "ATen/Functions.h"
#include "ATen/core/interned_strings.h"
#include "torch/nn/modules/activation.h"
#include "torch/optim/lbfgs.h"
#include "torch/optim/rmsprop.h"

#include <algorithm>
#include <random> 
#include <numeric>
#include <cmath>
#include <filesystem>
#include <iostream>

// OpenFOAM 
#include "fvCFD.H"

// libtorch-OpenFOAM data transfer
#include "torchFunctions.C"  
#include "fileNameGenerator.H" 
#include "dataset.H"
#include <sys/stat.h>


using namespace Foam;
using namespace torch::indexing;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

class PINNModel
:
    public torch::nn::Module
{
    private:
        torch::nn::Linear inputLayer1;
        //torch::nn::Linear inputLayer2;
        torch::nn::Linear hiddenLayer1;
        torch::nn::Linear hiddenLayer2;
        //torch::nn::Linear hiddenLayer3;        
        torch::nn::Linear outputLayer;
    public:
    
    PINNModel(label inputSize, label hiddenSize, label outputSize)
        :
          inputLayer1(torch::nn::Linear(inputSize, hiddenSize)), 
          //inputLayer2(torch::nn::Linear(1, hiddenSize)),  // Second input layer for time
          hiddenLayer1(torch::nn::Linear(hiddenSize, 2*hiddenSize)),
          hiddenLayer2(torch::nn::Linear(2*hiddenSize, 4*hiddenSize)),
          //hiddenLayer3(torch::nn::Linear(2*hiddenSize, hiddenSize)),      
          outputLayer(torch::nn::Linear(4*hiddenSize, 3*outputSize)) 
    {
    

        /*register_module(const std::string& name, torch::nn::Module& module);*/
        register_module("inputLayer1",inputLayer1);
        //register_module("inputLayer2",inputLayer2);
        register_module("hiddenLayer1",hiddenLayer1);
        register_module("hiddenLayer2",hiddenLayer2);
        //register_module("hiddenLayer3",hiddenLayer3);
        register_module("outputLayer",outputLayer); 
        
        // Initialize the weights
        torch::nn::init::xavier_uniform_(inputLayer1->weight);
        //torch::nn::init::xavier_uniform_(inputLayer2->weight);
        torch::nn::init::xavier_uniform_(hiddenLayer1->weight);
        torch::nn::init::xavier_uniform_(hiddenLayer2->weight);
        //torch::nn::init::xavier_uniform_(hiddenLayer3->weight); 
        torch::nn::init::xavier_uniform_(outputLayer->weight);
        
        // Initialize the biases to non-zero values
/*        torch::nn::init::zeros_(inputLayer1->bias);
        torch::nn::init::zeros_(inputLayer2->bias);
        torch::nn::init::zeros_(hiddenLayer1->bias);
        torch::nn::init::zeros_(hiddenLayer2->bias);
        torch::nn::init::zeros_(hiddenLayer3->bias);        
        torch::nn::init::zeros_(outputLayer->bias);   */
        
        torch::nn::init::normal_(inputLayer1->bias, 0, 0.1);       
        //torch::nn::init::normal_(inputLayer2->bias, 0, 0.1);
        torch::nn::init::normal_(hiddenLayer1->bias, 0, 0.1);
        torch::nn::init::normal_(hiddenLayer2->bias, 0, 0.1);
       // torch::nn::init::normal_(hiddenLayer3->bias, 0, 0.1);     
        torch::nn::init::normal_(outputLayer->bias, 0, 0.1); 
    }        
   
    torch::Tensor forward(torch::Tensor x)//, torch::Tensor t)
    {
        // Apply the first input layer to spatial data
        x = torch::tanh(inputLayer1(x)); 
        // Apply the second input layer to temporal data
        //t = torch::tanh(inputLayer2(t));
        // Concatenate the spatial and temporal features
        //torch::Tensor combinedInput = torch::cat({x, t}, /*dim=*/1);
        // Apply the first hidden layer
        x = torch::tanh(hiddenLayer1(x));
        // Apply the second hidden layer
        x = torch::tanh(hiddenLayer2(x));
       // x = torch::tanh(hiddenLayer3(x));
        // Apply the second hidden layer
//        combinedInput = torch::tanh(hiddenLayer3(combinedInput));        
        // Apply the output layer
        return outputLayer(x);
    }  
        
    
    // Function to define and return the Adam optimizer 
    torch::optim::Adam getAdamOptimizer(scalar optimizerStep)
    {
        return torch::optim::Adam
        (
            parameters(),
            torch::optim::AdamOptions(optimizerStep)
        );
    }
    
    // Function to define and return the BFGS optimizer  
    torch::optim::LBFGS getLBFGSOptimizer(scalar learningRate)
    {
        // Define L-BFGS options
        torch::optim::LBFGSOptions options;
        options.lr(learningRate);                   // Learning rate
        options.max_iter(2000);             // Maximum number of iterations
        options.max_eval(2500);             // Maximum number of function evaluations
        options.tolerance_grad(1e-08);       // Gradient tolerance
        options.tolerance_change(1e-10);     // Change tolerance
        options.history_size(200);           // History size
        options.line_search_fn("strong_wolfe");  // Line search function    
    
        return torch::optim::LBFGS
        (
            parameters(),
            options
        );
    }  
    
    // Function to define and return the MSprop optimizer    
    torch::optim::RMSprop getRMSpropOptimizer(scalar learningRate)
    {
        return torch::optim::RMSprop
        (
            parameters(),
            torch::optim::RMSpropOptions(learningRate)
                .alpha(0.99)
                .eps(1.0e-8)
                .weight_decay(0.0)
        );
    }       
 
    void clipGradNorm(scalar maxNorm)
    {
        for (auto & module : children())
        {
            for(auto & parameter : module -> parameters())
            {
                if(parameter.grad().defined())
                {
                   torch::nn::utils::clip_grad_norm_(parameter,maxNorm);
               }
           }
       }
    }

    
};


int main(int argc, char *argv[])
{
   
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    
    #include "createFields.H"
    
    std::ofstream writeExeData("exeData.txt");
    
    PINNModel nn(inputSize, hiddenSize, outputSize);
    
    nn.to(torch::kFloat);
    
//    torch::load(nn, file_name)

    // Set up PINN models, optimizers, and training parameters    
    torch::optim::Adam optimizer = nn.getAdamOptimizer(optimizerStep);
    auto dataset = CustomDataset(inputDir, outputDir, numFiles, inputSize, outputSize)
                   .map(torch::data::transforms::Stack<>());
                   
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>
    (
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(minibatch)
    );
    
    //scalar decayRate = 0.95;
    
    //scalar lambda = 0.000;
    
    // computing pde loss
    
    string bedElevationFileName = "bedElevation/bedElevation";
    std::ifstream bedElevationFile(bedElevationFileName);
    std::vector<float> data(outputSize); 
    if (bedElevationFile.is_open()) 
    {
        for (int i = 0; i < outputSize; ++i) 
        {
            bedElevationFile >> data[i];
        }
    }
   torch::Tensor bedElevation = torch::from_blob(data.data(), {1,outputSize}, torch::kFloat).clone();

    
    auto loss = [&](auto inputs, auto combinedTargets)
    {
        
        auto labels = combinedTargets.index({torch::indexing::Slice(),torch::indexing::Slice(), 0});   
        labels = (labels-stage_min)/(stage_max-stage_min); 
             
        auto Ux = combinedTargets.index({torch::indexing::Slice(),torch::indexing::Slice(), 1});
        Ux = (Ux-U_min)/(U_max-U_min);
        
        //std::cout<<"Ux"<<Ux.sizes()<<std::endl;
        
        auto Uy = combinedTargets.index({torch::indexing::Slice(),torch::indexing::Slice(), 2});
        Uy = (Uy-U_min)/(U_max-U_min);
        
        auto forwardTimeValue = combinedTargets.index({torch::indexing::Slice(),torch::indexing::Slice(), 3});
        auto laterTimeValue = combinedTargets.index({torch::indexing::Slice(),torch::indexing::Slice(), 4});
        auto currentInflow = combinedTargets.index({torch::indexing::Slice(),torch::indexing::Slice(), 5});//.unsqueeze(1);
        auto laterInflow = combinedTargets.index({torch::indexing::Slice(),torch::indexing::Slice(), 6});

	const DimensionedField<double, volMesh>& volumes = mesh.V();

	const double* volume_data = volumes.begin();


	std::vector<float> volume_data_float(volumes.size());
	std::transform(volume_data, volume_data + volumes.size(), volume_data_float.begin(), [](double value) 
	{
    		return static_cast<float>(value);
	});


	torch::Tensor volume_tensor = torch::from_blob(volume_data_float.data(), {volumes.size()}, torch::kFloat).clone();
	inputs=(inputs-stage_min)/(stage_max-stage_min);
        inputs = inputs.view({-1,inputSize});

  
        torch::Tensor predict = nn.forward(inputs);
        predict = predict.view({predict.size(0), outputSize, 3});


        torch::Tensor labelPredict = predict.index({torch::indexing::Slice(),torch::indexing::Slice(), 0});
        torch::Tensor UxPredict = predict.index({torch::indexing::Slice(),torch::indexing::Slice(), 1});
        torch::Tensor UyPredict = predict.index({torch::indexing::Slice(),torch::indexing::Slice(), 2});

        /*double alpha = -0.05;
        torch::Tensor error = -labelPredict + labels;
        torch::Tensor labelLoss = torch::mean(torch::exp(alpha * error) - alpha * error - 1);*/
        
        torch::Tensor labelLoss = torch::mse_loss(labelPredict, labels);
        torch::Tensor UxLoss = torch::mse_loss(UxPredict, Ux);
        torch::Tensor UyLoss = torch::mse_loss(UyPredict, Uy);


        //torch::Tensor variance = predict - labels;
        //torch::Tensor varianceLoss = torch::var(variance);
        
        torch::Tensor bedElevation_expand = bedElevation.repeat({labelPredict.size(0), 1});
    
    /*    
        auto loss1 = torch::relu(torch::sum(volume_tensor*(labelPredict-(bedElevation_expand-32)), 1) 
                   - torch::sum(volume_tensor*(forwardTimeValue-(bedElevation_expand-32)) , 1)
                   - deltaT*torch::mean(currentInflow, 1)) /totalArea;


        auto loss2 = torch::relu(torch::sum(volume_tensor*(laterTimeValue-(bedElevation_expand-32)), 1) 
                   - torch::sum(volume_tensor*(labelPredict-(bedElevation_expand-32)), 1)
                   - deltaT*torch::mean(laterInflow, 1))/totalArea;
    */
                   
        auto loss1 = torch::relu(torch::sum(volume_tensor*((labelPredict*(stage_max-stage_min)+stage_min)-bedElevation_expand), 1) 
                   - torch::sum(volume_tensor*(forwardTimeValue-bedElevation_expand) , 1)
                   - deltaT*torch::mean(currentInflow, /*dim=*/1)) /totalArea;


        auto loss2 = torch::relu(torch::sum(volume_tensor*(laterTimeValue-bedElevation_expand), /*dim=*/1) 
                   - torch::sum(volume_tensor*((labelPredict*(stage_max-stage_min)+stage_min)-bedElevation_expand), 1)
                   - deltaT*torch::mean(laterInflow, /*dim=*/1))/totalArea;

        auto massLoss1 = torch::mse_loss(loss1, torch::zeros_like(loss1));
        auto massLoss2 = torch::mse_loss(loss2, torch::zeros_like(loss2));
        auto massLoss = massLoss1 + massLoss2;
        
        //std::cout<<loss1<<"\t"<<loss1.sizes()<<std::endl;
        //std::cout<<loss2<<"\t"<<loss2.sizes()<<std::endl;
        
	torch::Tensor totalMassLoss = massLoss.sum();
	torch::Tensor totalLabelLoss = labelLoss.sum();
	//torch::Tensor totalUxLoss = UxLoss.sum();
	//torch::Tensor totalUyLoss = UyLoss.sum();
        
        float weight1 = 1.0;
        float weight2 = 1.0;
        
        torch::Tensor totalLoss = weight1*totalMassLoss + weight2*totalLabelLoss; //+ totalUxLoss + totalUyLoss;
        
        std::cout << "mass loss: " << totalMassLoss.item<float>() << "\t"
                  << "label loss: " << totalLabelLoss.item<float>() << "\t"
                  //<< "Ux loss: " << totalUxLoss.item<float>() << "\t"
                  //<< "Uy loss: " << totalUyLoss.item<float>() << "\t"
                  << "total loss: " << totalLoss.item<float>()
                  << std::endl<<std::endl;
        writeExeData << "mass loss: " << totalMassLoss.item<float>() << "\t"
                     << "label loss: " << totalLabelLoss.item<float>() << "\t"
                     //<< "Ux loss: " << totalUxLoss.item<float>() << "\t"
                    // << "Uy loss: " << totalUyLoss.item<float>() << "\t"
                     << "total loss: " << totalLoss.item<float>()
                     << std::endl<<std::endl;
                  
        return totalLoss;    
    };  

    
    auto saveModelParameters = [&](const PINNModel& nn, const std::string& filename)
    {
    	torch::serialize::OutputArchive archive;
    	nn.save(archive);  
    	archive.save_to(filename);
    	std::cout << "Model parameters saved to " << filename << std::endl;         
    };   
    
    auto loadModelParameters = [&](PINNModel& nn, const std::string& filename)
    {
    	torch::serialize::InputArchive archive;
    	archive.load_from(filename);
   	nn.load(archive); 
    	std::cout << "Model parameters loaded from " << filename << std::endl;      
    };   
    
    if(!std::filesystem::exists("bestModel")) 
    {    

    
    if(std::filesystem::exists("tempModel"))
    {
        loadModelParameters(nn, "tempModel");
        maxIterations = 0 ;
    }

    //Training loop
    for(std::size_t epoch = 1; epoch <= maxIterations; ++epoch) 
    {
    
    	size_t batch_index = 1;
        for (auto& batch : *data_loader) 
      {
        optimizer.zero_grad();
        
        auto inputs = batch.data;
        auto combinedTargets = batch.target;
        
        std::cout << "Epoch: " << epoch << ", Batch: " << batch_index++ << std::endl;
        writeExeData<< "Epoch: " << epoch << ", Batch: " << batch_index << std::endl;   
        torch::Tensor totalLoss = loss(inputs, combinedTargets);
        totalLoss.backward();  
        optimizer.step();       

      }
      
      if(epoch % 1000 == 0)
      {
 	    saveModelParameters(nn, "tempModel");     
      }

    }
    
    // Train the network using LBFGS optimizer
    torch::optim::LBFGS optimizer1 = nn.getLBFGSOptimizer(BFGS_learningRate); 

    

    //Training loop
    for(std::size_t epoch = 1; epoch <= BFGS_maxIterations; ++epoch) 
    {
    	size_t batch_index = 0;
        for (auto& batch : *data_loader) 
      {
        auto inputs = batch.data;
        auto combinedTargets = batch.target;
        
        std::cout << "Epoch: " << epoch << ", Batch: " << batch_index++ << std::endl;
        writeExeData<< "Epoch: " << epoch << ", Batch: " << batch_index << std::endl; 
	 auto closure = [&]() 
	 
	 {

            optimizer1.zero_grad();
            

            torch::Tensor totalLoss = loss(inputs, combinedTargets);
            

            totalLoss.backward();
            
            return totalLoss;
        };


        optimizer1.step(closure);

       }
           std::string fileName = "BFGStempModel" + std::to_string(epoch);
       
 	    saveModelParameters(nn, fileName); 
                    
    }
    
    saveModelParameters(nn, "bestModel");
    }  
    
//    output();
    loadModelParameters(nn, "bestModel");
    //  - Reinterpret OpenFOAM's vectorField as vector* array 
    for (int i = 1; i <= numFiles_predict; i++) 
   {
        runTime.setTime( startTime + (i-1)*86400+(inputSize/inputElementary-1)*deltaT, 0);
        std::string predict_input_file = predictDir + "/input_" + std::to_string(i);

        std::ifstream file(predict_input_file);
        std::vector<float> data(inputSize);  
        if (file.is_open()) 
        {
            for (int i = 0; i < inputSize; ++i) 
            {
                file >> data[i];
            }
        }
       auto input_tensor = torch::from_blob(data.data(), {inputSize,1}, torch::kFloat).clone(); 
       input_tensor=(input_tensor-stage_min)/(stage_max-stage_min);
       input_tensor = input_tensor.view({-1,inputSize});
       auto predict_tensor = nn.forward(input_tensor);
       predict_tensor = predict_tensor.view({outputSize, 3});
       auto label_predict = predict_tensor.index({torch::indexing::Slice(), 0});
       
       auto Ux_predict = predict_tensor.index({torch::indexing::Slice(), 1});
       auto Uy_predict = predict_tensor.index({torch::indexing::Slice(), 2});

       label_predict = label_predict.squeeze(); 
       Ux_predict = Ux_predict.squeeze(); 
       Uy_predict = Uy_predict.squeeze(); 

       label_predict = label_predict.to(torch::kDouble);
       Ux_predict = Ux_predict.to(torch::kDouble);
       Uy_predict = Uy_predict.to(torch::kDouble);

     forAll(stage_nn, cellI) 
      {
        //stage_nn[cellI] = label_predict[cellI].item<double>()+ 32 ;
        stage_nn[cellI] = label_predict[cellI].item<double>()*(stage_max-stage_min)+ stage_min ;
      }
  
     forAll(Ux_nn, cellI) 
      {
        Ux_nn[cellI] = Ux_predict[cellI].item<double>()*(U_max-U_min)+U_min;
      }
      
     forAll(Uy_nn, cellI) 
      {
        Uy_nn[cellI] = Uy_predict[cellI].item<double>()*(U_max-U_min)+U_min;
      }
      
    volScalarField stage_simulation 
    (
        IOobject
        (
            "stage",
            runTime.timeName(), 
            mesh, 
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );
      stage_nn.correctBoundaryConditions();
      Ux_nn.correctBoundaryConditions();
      Uy_nn.correctBoundaryConditions();
      stage_residual = mag(stage_simulation - stage_nn);//mag(stage_simulation - stage_nn)/mag(stage_simulation)*100;
      stage_residual.correctBoundaryConditions();


      stage_nn.write();
      Ux_nn.write();
      Uy_nn.write();
      stage_residual.write();
   }

    Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
        << "  ClockTime = " << runTime.elapsedClockTime() << " s"
        << nl << endl;
            
    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
