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


// is this an integration test?? probably not
TEST(ClassifySvmSharedCommand, readSharedAndDesignFiles) {
    LabeledObservationVector labeledObservationVector;
    ClassifySvmSharedCommand::readSharedAndDesignFiles("test.shared", "test.design", labeledObservationVector);
    EXPECT_EQ(2, labeledObservationVector.size());
}


TEST(OneVsOneMultiClassSvmTrainer, FisherIrisData) {
    //LabelVector labelVector;
    //ObservationVector observationVector;
    LabeledObservationVector labeledObservationVector;

    ClassifySvmSharedCommand::readSharedAndDesignFiles("iris.shared", "iris.design", labeledObservationVector);

    EXPECT_EQ(150, labeledObservationVector.size());
    EXPECT_EQ("setosa", labeledObservationVector[0].first);
    EXPECT_EQ("versicolor", labeledObservationVector[50].first);
    EXPECT_EQ("virginica", labeledObservationVector[100].first);

    //OneVsOneMultiClassSvmTrainer t(observationVector, labelVector);
    OneVsOneMultiClassSvmTrainer t(labeledObservationVector);
    //EXPECT_EQ(50, t.getLabeledObservations()["setosa"]->size());
    //EXPECT_EQ(50, t.getLabeledObservations()["versicolor"]->size());
    //EXPECT_EQ(50, t.getLabeledObservations()["virginica"]->size());

    //EXPECT_EQ(3, t.getLabelPairSet().size());

    //MultiClassSVM* s = t.train();

    //delete s;

    for ( int j = 0; j < labeledObservationVector.size(); j++ ) {
        delete labeledObservationVector[j].second;
    }
}

void print(int z) {
    std::cout << "z=" << z << std::endl;
}
