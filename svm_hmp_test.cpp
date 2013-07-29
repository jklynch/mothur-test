//
//  svm_integration_test.cpp
//
//  Created by Joshua Lynch on 7/8/2013.
//  Copyright (c) 2013 Schloss Lab. All rights reserved.
//

#include "gtest/gtest.h"

#include "mothur/mothurout.h"
#include "mothur/groupmap.h"
#include "mothur/inputdata.h"
#include "mothur/classifysvmsharedcommand.h"

#include "mothur/svm.hpp"


MothurOut* MothurOut::_uniqueInstance = 0;


TEST(OneVsOneMultiClassSvmTrainer, HmpData) {
    LabeledObservationVector labeledObservationVector;

    std::cout << "testing hmp data" << std::endl;
    std::string sharedFilePath = "~/gsoc2013/data/Stool.0.03.subsample.0.03.filter.shared";
    std::string designFilePath = "~/gsoc2013/data/Stool.0.03.subsample.0.03.filter.mix.design";

    ClassifySvmSharedCommand::readSharedAndDesignFiles(sharedFilePath, designFilePath, labeledObservationVector);

    EXPECT_EQ(596, labeledObservationVector.size());

    OneVsOneMultiClassSvmTrainer t(labeledObservationVector);
    EXPECT_EQ(3, t.getLabelSet().size());
    EXPECT_EQ(3, t.getLabelPairSet().size());

    ParameterRangeMap linearParameterRangeMap;
    linearParameterRangeMap[SmoTrainer::MapKey_C] = SmoTrainer::defaultCRange;
    linearParameterRangeMap[LinearKernelFunction::MapKey_Constant] = LinearKernelFunction::defaultConstantRange;

    ParameterRangeMap rbfParameterRangeMap;
    rbfParameterRangeMap[SmoTrainer::MapKey_C] = SmoTrainer::defaultCRange;
    rbfParameterRangeMap[RbfKernelFunction::MapKey_Gamma] = RbfKernelFunction::defaultGammaRange;

    ParameterRangeMap polynomialParameterRangeMap;
    polynomialParameterRangeMap[SmoTrainer::MapKey_C] = SmoTrainer::defaultCRange;
    polynomialParameterRangeMap[PolynomialKernelFunction::MapKey_Constant] = PolynomialKernelFunction::defaultConstantRange;
    polynomialParameterRangeMap[PolynomialKernelFunction::MapKey_Degree] = PolynomialKernelFunction::defaultDegreeRange;

    KernelParameterRangeMap kernelParameterRangeMap;
    kernelParameterRangeMap[LinearKernelFunction::MapKey] = linearParameterRangeMap;
    kernelParameterRangeMap[RbfKernelFunction::MapKey] = rbfParameterRangeMap;
    kernelParameterRangeMap[PolynomialKernelFunction::MapKey] = polynomialParameterRangeMap;

    MultiClassSVM* s = t.train(kernelParameterRangeMap);

    delete s;

    for (LabeledObservationVector::iterator i = labeledObservationVector.begin(); i != labeledObservationVector.end(); i++) {
        delete i->second;
    }
}
