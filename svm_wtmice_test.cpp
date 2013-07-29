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


TEST(OneVsOneMultiClassSvmTrainer, WtMiceData) {
    LabeledObservationVector labeledObservationVector;

    std::string sharedFilePath = "~/gsoc2013/data/WTmiceonly_final.shared";
    std::string designFilePath = "~/gsoc2013/data/WTmiceonly_final.design";

    ClassifySvmSharedCommand::readSharedAndDesignFiles(sharedFilePath, designFilePath, labeledObservationVector);

    EXPECT_EQ(113, labeledObservationVector.size());

    OneVsOneMultiClassSvmTrainer t(labeledObservationVector);
    EXPECT_EQ(4, t.getLabelSet().size());
    EXPECT_EQ(6, t.getLabelPairSet().size());

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

    for (LabeledObservationVector::iterator i = labeledObservationVector.begin(); i != labeledObservationVector.end(); i++) {
        delete i->second;
    }
}
