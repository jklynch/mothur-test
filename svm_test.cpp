//
//  svm_test.cpp
//
//  Created by Joshua Lynch on 6/19/2013.
//  Copyright (c) 2013 Schloss Lab. All rights reserved.
//

#include "gtest/gtest.h"

#include "mothur/mothurout.h"
#include "mothur/inputdata.h"
#include "mothur/svm.hpp"


MothurOut* MothurOut::_uniqueInstance = 0;


TEST(SvmTest, Construct) {
    SVM svm();
}

TEST(SvmTest, ReadData) {
    InputData input("test.shared", "sharedfile");
    vector<SharedRAbundVector*> lookup = input.getSharedRAbundVectors();
    int numObservations = lookup.size();
    EXPECT_EQ(2, numObservations);
    int numFeatures = lookup[0]->getNumBins();
    EXPECT_EQ(5, numFeatures);

    // convert OTU frequency counts to double
    // in general we will also want to normalize the counts
    //std::vector<std::vector<double> > observations(numObservations, std::vector<double> (numFeatures, 0.0));
}

TEST(FeatureVector, Construct) {
    FeatureVector fv(2);
    EXPECT_EQ(2, fv.size());
    fv[0] = 0.0;
    fv[1] = 0.0;
    EXPECT_EQ(2, fv.size());
}

TEST(OneVsOneMultiClassSvmTrainer, GetLabelSet) {
    LabelVector labelVector;
    labelVector.push_back("label_0");
    labelVector.push_back("label_2");
    labelVector.push_back("label_1");
    labelVector.push_back("label_0");
    labelVector.push_back("label_2");
    labelVector.push_back("label_1");
    LabelSet expectedLabelSet;
    expectedLabelSet.insert(labelVector.begin(), labelVector.end());
    //LabelSet labelSet;

    FeatureVector featureVector;
    ObservationVector observations;
    observations.push_back(&featureVector);
    observations.push_back(&featureVector);
    observations.push_back(&featureVector);
    observations.push_back(&featureVector);
    observations.push_back(&featureVector);
    observations.push_back(&featureVector);
    OneVsOneMultiClassSvmTrainer t(observations, labelVector);

    const LabelSet& labelSet = t.getLabelSet();

    EXPECT_EQ(expectedLabelSet.size(), labelSet.size());
    EXPECT_EQ(expectedLabelSet, labelSet);
}
