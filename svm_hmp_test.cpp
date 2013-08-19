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


class HmpDataFixture : public testing::Test {
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

        std::cout << "testing hmp data" << std::endl;
        std::string sharedFilePath = "~/gsoc2013/data/Stool.0.03.subsample.0.03.filter.shared";
        std::string designFilePath = "~/gsoc2013/data/Stool.0.03.subsample.0.03.filter.mix.design";
        classifySvmSharedCommand.readSharedAndDesignFiles(sharedFilePath, designFilePath, labeledObservationVector, featureVector);

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
TEST_F(HmpDataFixture, OneVsOneMultiClassSvmTrainer) {
    EXPECT_EQ(596, labeledObservationVector.size());

    EXPECT_EQ(3, trainer->getLabelSet().size());
    EXPECT_EQ(3, trainer->getLabelPairSet().size());

    KernelParameterRangeMap kernelParameterRangeMap;
    getDefaultKernelParameterRangeMap(kernelParameterRangeMap);

    MultiClassSVM* s = trainer->train(kernelParameterRangeMap);
    std::cout << "in the HMP test - done training" << std::endl;

    delete s;
}


TEST_F(HmpDataFixture, SvmRfe) {
    SvmRfe svmRfe;
    FeatureList orderedFeatureList = svmRfe.getOrderedFeatureList(*svmDataset, *trainer, LinearKernelFunction::defaultConstantRange, SmoTrainer::defaultCRange);

    int n = 0;
    std::cout << "ordered features:" << std::endl;
    for (FeatureList::iterator i = orderedFeatureList.begin(); i != orderedFeatureList.end(); i++) {
        std::cout << i->getFeatureLabel() << std::endl;
        n++;
        if (n > 20) break;
    }

}
