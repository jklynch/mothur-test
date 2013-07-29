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


TEST(OneVsOneMultiClassSvmTrainer, FisherIrisData) {
    LabeledObservationVector labeledObservationVector;

    ClassifySvmSharedCommand::readSharedAndDesignFiles("iris.shared", "iris.design", labeledObservationVector);

    EXPECT_EQ(150, labeledObservationVector.size());
    EXPECT_EQ("setosa", labeledObservationVector[0].first);
    EXPECT_EQ("versicolor", labeledObservationVector[50].first);
    EXPECT_EQ("virginica", labeledObservationVector[100].first);

    KernelParameterRangeMap kernelParameterRangeMap;
    getDefaultKernelParameterRangeMap(kernelParameterRangeMap);

    OneVsOneMultiClassSvmTrainer t(labeledObservationVector);
    EXPECT_EQ(50, t.getLabeledObservationVectorForLabel("setosa").size());
    EXPECT_EQ(50, t.getLabeledObservationVectorForLabel("versicolor").size());
    EXPECT_EQ(50, t.getLabeledObservationVectorForLabel("virginica").size());

    EXPECT_EQ(3, t.getLabelPairSet().size());

    std::cout << "test:  train" << std::endl;
    MultiClassSVM* s = t.train(kernelParameterRangeMap);
    std::cout << "test:  delete s" << std::endl;
    delete s;

    for (LabeledObservationVector::iterator i = labeledObservationVector.begin(); i != labeledObservationVector.end(); i++) {
        delete i->second;
    }
}
