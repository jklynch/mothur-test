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
// 1 + *
// 0 + *
//   012
TEST(SmoTrainer, Train) {
    LabelVector labelVector;
    labelVector.push_back("blue");
    labelVector.push_back("blue");
    labelVector.push_back("blue");
    labelVector.push_back("blue");
    labelVector.push_back("green");
    labelVector.push_back("green");
    labelVector.push_back("green");
    labelVector.push_back("green");

    //
    // blue data points
    //
    FeatureVector x_blue_0(2);
    x_blue_0[0] = 1.0;
    x_blue_0[1] = 3.0;
    FeatureVector x_blue_1(2);
    x_blue_1[0] = 2.0;
    x_blue_1[1] = 5.0;
    FeatureVector x_blue_2(2);
    x_blue_2[0] = 3.0;
    x_blue_2[1] = 8.0;
    FeatureVector x_blue_3(2);
    x_blue_3[0] = 6.0;
    x_blue_3[1] = 4.0;

    //
    // green data points
    //
    FeatureVector x_green_0(2);
    x_green_0[0] = 6.0;
    x_green_0[1] = 7.0;
    FeatureVector x_green_1(2);
    x_green_1[0] = 7.0;
    x_green_1[1] = 8.0;
    FeatureVector x_green_2(2);
    x_green_2[0] = 8.0;
    x_green_2[1] = 4.0;
    FeatureVector x_green_3(2);
    x_green_3[0] = 3.0;
    x_green_3[1] = 6.0;
    ObservationVector observationVector;
    observationVector.push_back(&x_blue_0);
    observationVector.push_back(&x_blue_1);
    observationVector.push_back(&x_blue_2);
    observationVector.push_back(&x_blue_3);
    observationVector.push_back(&x_green_0);
    observationVector.push_back(&x_green_1);
    observationVector.push_back(&x_green_2);
    observationVector.push_back(&x_green_3);
    std::cout << "observation count: " << observationVector.size() << std::endl;
    std::cout << "feature count: " << observationVector[0]->size() << std::endl;

    // online method for mean and variance
    for ( FeatureVector::size_type feature = 0; feature < observationVector[0]->size(); feature++ ) {
        double n = 0.0;
        double mean = 0.0;
        double M2 = 0.0;
        for ( ObservationVector::size_type observation = 0; observation < observationVector.size(); observation++ ) {
            n += 1.0;
            double x = observationVector[observation]->at(feature);
            double delta = x - mean;
            mean += delta / n;
            M2 += delta * (x - mean);
        }
        double variance = M2 / (n - 1.0);
        double standardDeviation = sqrt(variance);
        std::cout << "mean of feature " << feature << " is " << mean << std::endl;
        std::cout << "std of feature " << feature << " is " << standardDeviation << std::endl;
        // normalize the feature
        for ( ObservationVector::size_type observation = 0; observation < observationVector.size(); observation++ ) {
            observationVector[observation]->at(feature) = (observationVector[observation]->at(feature) - mean ) / standardDeviation;
        }
    }

    for ( ObservationVector::size_type i = 0; i < observationVector.size(); i++ ) {
        std::cout << "i = " << i;
        for ( FeatureVector::size_type j = 0; j < observationVector[0]->size(); j++ ) {
            std::cout << " " << observationVector[i]->at(j);
        }
        std::cout << std::endl;
    }


    SmoTrainer t;
    SVM* svm = t.train(observationVector, labelVector);

    //EXPECT_EQ(-1, svm->discriminant(x_01));
    //EXPECT_EQ(1, svm->discriminant(x_02));
    //EXPECT_EQ(1, svm->discriminant(x_21));
    //EXPECT_EQ(1, svm->discriminant(x_21));

    //delete svm;
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
