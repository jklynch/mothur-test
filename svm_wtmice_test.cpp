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


class MouseDataFixture : public testing::Test {
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

        std::cout << "testing wtmiceonly data" << std::endl;
        std::string sharedFilePath = "~/gsoc2013/data/WTmiceonly_final.shared";
        std::string designFilePath = "~/gsoc2013/data/WTmiceonly_final.design";
        classifySvmSharedCommand.readSharedAndDesignFiles(sharedFilePath, designFilePath, labeledObservationVector, featureVector);

        svmDataset = new SvmDataset(labeledObservationVector, featureVector);

        OutputFilter outputFilter(OutputFilter::QUIET);

        int evaluationFoldCount = 3;
        int trainFoldCount = 5;
        trainer = new OneVsOneMultiClassSvmTrainer(*svmDataset, evaluationFoldCount, trainFoldCount, externalInterruption, outputFilter);
    }

    virtual void TearDown() {
        delete trainer;
        delete svmDataset;
    }
};
/*
TEST(OneVsOneMultiClassSvmTrainer, WtMiceData) {
    MothurOut* m = MothurOut::getInstance();
    ClassifySvmSharedCommand classifySvmSharedCommand;

    LabeledObservationVector labeledObservationVector;
    FeatureVector featureVector;

    SvmDataset svmDataset(labeledObservationVector, featureVector);
    ExternalSvmTrainingInterruption externalInterruption;

    std::cout << "testing wtmiceonly data" << std::endl;
    std::string sharedFilePath = "~/gsoc2013/data/WTmiceonly_final.shared";
    std::string designFilePath = "~/gsoc2013/data/WTmiceonly_final.design";

    classifySvmSharedCommand.readSharedAndDesignFiles(sharedFilePath, designFilePath, labeledObservationVector, featureVector);

    EXPECT_EQ(113, labeledObservationVector.size());

    int evaluationFoldCount = 3;
    int trainFoldCount = 5;
    OneVsOneMultiClassSvmTrainer t(svmDataset, evaluationFoldCount, trainFoldCount, externalInterruption);
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
*/

TEST_F(MouseDataFixture, OneVsOneMultiClassSvmTrainerZeroOne) {
    transformZeroOne(labeledObservationVector);
    EXPECT_EQ(113, labeledObservationVector.size());

    EXPECT_EQ(4, trainer->getLabelSet().size());
    EXPECT_EQ(6, trainer->getLabelPairSet().size());

    KernelParameterRangeMap kernelParameterRangeMap;
    getDefaultKernelParameterRangeMap(kernelParameterRangeMap);

    MultiClassSVM* s = trainer->train(kernelParameterRangeMap);
    std::cout << "in the WTmice test - done training" << std::endl;

    delete s;
}


TEST_F(MouseDataFixture, OneVsOneMultiClassSvmTrainerZeroMeanUnitVariance) {
    transformZeroOne(labeledObservationVector);
    EXPECT_EQ(113, labeledObservationVector.size());

    EXPECT_EQ(4, trainer->getLabelSet().size());
    EXPECT_EQ(6, trainer->getLabelPairSet().size());

    KernelParameterRangeMap kernelParameterRangeMap;
    getDefaultKernelParameterRangeMap(kernelParameterRangeMap);

    MultiClassSVM* s = trainer->train(kernelParameterRangeMap);
    std::cout << "in the WTmice test - done training" << std::endl;

    delete s;
}

// SmoTrainer C does not seem to be important here
// LinearKernelFunction constant range does not seem to be important here
TEST_F(MouseDataFixture, SvmRfe) {
    transformZeroOne(labeledObservationVector);
    SvmRfe svmRfe;
    double constantRangeList[] = {0.0};
    ParameterRange linearConstantRange(constantRangeList, constantRangeList + 1);
    double smoCRangeList[] = {10.0, 1.0, 0.1};
    ParameterRange smoCRange(smoCRangeList, smoCRangeList + 3);
    RankedFeatureList orderedFeatureList = svmRfe.getOrderedFeatureList(*svmDataset, *trainer, linearConstantRange, smoCRange);

    int n = 0;
    std::cout << "ordered features:" << std::endl;
    for (RankedFeatureList::iterator i = orderedFeatureList.begin(); i != orderedFeatureList.end(); i++) {
        std::cout << i->getFeature().getFeatureLabel() << std::endl;
        n++;
        if (n > 20) break;
    }
}
