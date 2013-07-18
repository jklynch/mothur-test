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
    EXPECT_EQ("0", labeledObservationVector[0].first);   // setosa
    EXPECT_EQ("1", labeledObservationVector[50].first);  // versicolor
    EXPECT_EQ("2", labeledObservationVector[100].first); // virginica

    OneVsOneMultiClassSvmTrainer t(labeledObservationVector);
    EXPECT_EQ(50, t.getLabeledObservationVectorForLabel("0").size());
    EXPECT_EQ(50, t.getLabeledObservationVectorForLabel("1").size());
    EXPECT_EQ(50, t.getLabeledObservationVectorForLabel("2").size());

    EXPECT_EQ(3, t.getLabelPairSet().size());

    std::cout << "test:  train" << std::endl;
    MultiClassSVM* s = t.train();
    std::cout << "test:  delete s" << std::endl;
    delete s;

    int n = 0;
    for (LabeledObservationVector::iterator i = labeledObservationVector.begin(); i != labeledObservationVector.end(); i++) {
        std::cout << "test: delete labeled observation vector " << n << " with label " << i->first << std::endl;
        delete i->second;
    }
}
