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

        OutputFilter outputFilter(OutputFilter::DEBUG);

        std::cout << "testing hmp data" << std::endl;
        std::string sharedFilePath = "~/gsoc2013/data/Stool.0.03.subsample.0.03.filter.shared";
        std::string designFilePath = "~/gsoc2013/data/Stool.0.03.subsample.0.03.filter.mix.design";
        classifySvmSharedCommand.readSharedAndDesignFiles(sharedFilePath, designFilePath, labeledObservationVector, featureVector);
        transformZeroOne(labeledObservationVector);
        svmDataset = new SvmDataset(labeledObservationVector, featureVector);

        int evaluationFoldCount = 3;
        int trainFoldCount = 5;
        trainer = new OneVsOneMultiClassSvmTrainer(*svmDataset, evaluationFoldCount, trainFoldCount, externalInterruption, outputFilter);

    }

    virtual void TearDown() {
        delete trainer;
        delete svmDataset;
    }
};

TEST_F(HmpDataFixture, OneVsOneMultiClassSvmTrainerTransformZeroOne) {
    //transformZeroOne(labeledObservationVector);
    EXPECT_EQ(596, labeledObservationVector.size());

    EXPECT_EQ(3, trainer->getLabelSet().size());
    EXPECT_EQ(3, trainer->getLabelPairSet().size());

    KernelParameterRangeMap kernelParameterRangeMap;
    getDefaultKernelParameterRangeMap(kernelParameterRangeMap);

    MultiClassSVM* s = trainer->train(kernelParameterRangeMap);
    std::cout << "in the HMP test - done training" << std::endl;

    delete s;
}

TEST_F(HmpDataFixture, OneVsOneMultiClassSvmTrainerTransformZeroMeanUnitVariance) {
    transformZeroMeanUnitVariance(labeledObservationVector);
    EXPECT_EQ(596, labeledObservationVector.size());

    EXPECT_EQ(3, trainer->getLabelSet().size());
    EXPECT_EQ(3, trainer->getLabelPairSet().size());

    KernelParameterRangeMap kernelParameterRangeMap;
    getDefaultKernelParameterRangeMap(kernelParameterRangeMap);

    MultiClassSVM* s = trainer->train(kernelParameterRangeMap);
    std::cout << "in the HMP test - done training" << std::endl;

    delete s;
}

// using more constants does make a difference in results
TEST_F(HmpDataFixture, SvmRfe) {
    SvmRfe svmRfe;
    double constantRange[] = {0.0};
    ParameterRange linearConstantRange(constantRange, constantRange + 1);
    RankedFeatureList orderedFeatureList = svmRfe.getOrderedFeatureList(*svmDataset, *trainer, linearConstantRange, SmoTrainer::defaultCRange);

    int n = 0;
    std::cout << "ordered features:" << std::endl;
    for (RankedFeatureList::iterator i = orderedFeatureList.begin(); i != orderedFeatureList.end(); i++) {
        std::cout << i->getFeature().getFeatureLabel() << std::endl;
        n++;
        if (n > 20) break;
    }

}
