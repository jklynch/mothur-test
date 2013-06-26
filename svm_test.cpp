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

TEST(SmoTrainer, AssignNumericLabels) {
    LabelVector labelVector;
    labelVector.push_back("label_0");
    labelVector.push_back("label_2");
    labelVector.push_back("label_0");
    labelVector.push_back("label_2");

    std::vector<double> y(4);
    SmoTrainer t;
    t.assignNumericLabels(y, labelVector);

    EXPECT_EQ(4, y.size());
    EXPECT_EQ(-1.0, y[0]);
    EXPECT_EQ(+1.0, y[1]);
    EXPECT_EQ(-1.0, y[2]);
    EXPECT_EQ(+1.0, y[3]);
}

// here we are testing for an exception if SmoTrainer.assignNumericLabels()
// gets more than two labels or fewer than two labels
TEST(SmoTrainer, MoreThanTwoLabels) {
    LabelVector oneLabelVector;
    oneLabelVector.push_back("label_0");
    oneLabelVector.push_back("label_0");
    oneLabelVector.push_back("label_0");

    LabelVector threeLabelsVector;
    threeLabelsVector.push_back("label_0");
    threeLabelsVector.push_back("label_1");
    threeLabelsVector.push_back("label_2");

    std::vector<double> y(3);
    SmoTrainer t;
    EXPECT_THROW(t.assignNumericLabels(y, oneLabelVector), std::exception);
    EXPECT_THROW(t.assignNumericLabels(y, threeLabelsVector), std::exception);
}

// test SmoTrainer on four data points
//
// 1 * *
// 0 * *
//   012
TEST(SmoTrainer, Train) {
    LabelVector labelVector;
    labelVector.push_back("label_0");
    labelVector.push_back("label_2");
    labelVector.push_back("label_0");
    labelVector.push_back("label_2");

    FeatureVector x_01(2);
    x_01[0] = 0.0;
    x_01[1] = 0.0;
    FeatureVector x_02(2);
    x_02[0] = 0.0;
    x_02[1] = 1.0;
    FeatureVector x_21(2);
    x_21[0] = 2.0;
    x_21[1] = 0.0;
    FeatureVector x_22(2);
    x_22[0] = 2.0;
    x_22[1] = 1.0;
    ObservationVector observationVector(4);
    observationVector[0] = &x_01;
    observationVector[1] = &x_21;
    observationVector[2] = &x_02;
    observationVector[3] = &x_22;

    SmoTrainer t;
    SVM* svm = t.train(observationVector, labelVector);
    delete svm;
}


TEST(SmoTrainer, ElementwiseMultiply) {
    std::vector<double> a(2, 2.0);
    std::vector<double> b(2, 0.5);
    std::vector<double> c(2);
    SmoTrainer t;
    t.elementwise_multiply(a, b, c);
    EXPECT_EQ(1.0, c[0]);
    EXPECT_EQ(1.0, c[1]);
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
