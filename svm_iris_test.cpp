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


class FisherIrisDataFixture : public testing::Test {
public:
    MothurOut* m = MothurOut::getInstance();

    LabeledObservationVector labeledObservationVector;
    FeatureVector featureVector;

    SvmDataset* svmDataset;

    ExternalSvmTrainingInterruption externalInterruption;

    OneVsOneMultiClassSvmTrainer* trainer;

    virtual void SetUp() {
        ClassifySvmSharedCommand classifySvmSharedCommand;

        labeledObservationVector.clear();
        featureVector.clear();

        classifySvmSharedCommand.readSharedAndDesignFiles("iris.shared", "iris.design", labeledObservationVector, featureVector);

        svmDataset = new SvmDataset(labeledObservationVector, featureVector);

        int evaluationFoldCount = 3;
        int trainFoldCount = 5;
        trainer = new OneVsOneMultiClassSvmTrainer(*svmDataset, evaluationFoldCount, trainFoldCount, externalInterruption);
    }

    virtual void TearDown() {
        delete trainer;
        delete svmDataset;
    }
};

TEST_F(FisherIrisDataFixture, OneVsOneMultiClassSvmTrainer) {
    EXPECT_EQ(150, labeledObservationVector.size());
    EXPECT_EQ("setosa", labeledObservationVector[0].first);
    EXPECT_EQ("versicolor", labeledObservationVector[50].first);
    EXPECT_EQ("virginica", labeledObservationVector[100].first);

    KernelParameterRangeMap kernelParameterRangeMap;
    getDefaultKernelParameterRangeMap(kernelParameterRangeMap);

    EXPECT_EQ(50, trainer->getLabeledObservationVectorForLabel("setosa").size());
    EXPECT_EQ(50, trainer->getLabeledObservationVectorForLabel("versicolor").size());
    EXPECT_EQ(50, trainer->getLabeledObservationVectorForLabel("virginica").size());

    EXPECT_EQ(3, trainer->getLabelPairSet().size());

    std::cout << "test:  train" << std::endl;
    MultiClassSVM* s = trainer->train(kernelParameterRangeMap);
    std::cout << "test:  delete s" << std::endl;
    delete s;

}

TEST_F(FisherIrisDataFixture, SvmRfe) {
    SvmRfe svmRfe;
    FeatureList orderedFeatureList = svmRfe.getOrderedFeatureList(*svmDataset, *trainer, LinearKernelFunction::defaultConstantRange, SmoTrainer::defaultCRange);

    std::cout << "ordered features:" << std::endl;
    for (FeatureList::iterator i = orderedFeatureList.begin(); i != orderedFeatureList.end(); i++) {
        std::cout << i->getFeatureLabel() << std::endl;
    }

}
