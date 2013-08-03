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

    KernelParameterRangeMap kernelParameterRangeMap;
    getDefaultKernelParameterRangeMap(kernelParameterRangeMap);

    MultiClassSVM* s = t.train(kernelParameterRangeMap);
    std::cout << "in the WTmice test - done training" << std::endl;
    delete s;

    for (LabeledObservationVector::iterator i = labeledObservationVector.begin(); i != labeledObservationVector.end(); i++) {
        delete i->second;
    }
}
