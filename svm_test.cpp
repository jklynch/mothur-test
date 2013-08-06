//
//  svm_test.cpp
//
//  Created by Joshua Lynch on 6/19/2013.
//  Copyright (c) 2013 Schloss Lab. All rights reserved.
//

#include "gtest/gtest.h"

#include "mothur/mothurout.h"
#include "mothur/groupmap.h"
#include "mothur/inputdata.h"

#include "mothur/classifysvmsharedcommand.h"
#include "mothur/svm.hpp"


MothurOut* MothurOut::_uniqueInstance = 0;

// I'm not sure the behavior of this test is correct
TEST(ClassifySvmSharedCommand, readSharedAndDesignFiles) {
    LabeledObservationVector labeledObservationVector;
    ClassifySvmSharedCommand::readSharedAndDesignFiles("test.shared", "test.design", labeledObservationVector);
    EXPECT_EQ(2, labeledObservationVector.size());
    for (LabeledObservationVector::iterator i = labeledObservationVector.begin(); i != labeledObservationVector.end(); i++) {
        delete i->second;
    }
}

TEST(LabelPair, BuildLabelPair) {
    Label blueLabel = "blue";
    Label greenLabel = "green";
    LabelPair labelPair = buildLabelPair(blueLabel, greenLabel);

    EXPECT_EQ(2, labelPair.size());
    EXPECT_EQ("blue", labelPair[0]);
    EXPECT_EQ("green", labelPair[1]);

    LabelPair::iterator i = labelPair.begin();
    EXPECT_EQ("blue", *i);
    i++;
    EXPECT_EQ("green", *i);
    i++;
    EXPECT_EQ(labelPair.end(), i);
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

TEST(Observation, Construct) {
    Observation observation(2);
    EXPECT_EQ(2, observation.size());
    observation[0] = 0.0;
    observation[1] = 0.0;
    EXPECT_EQ(2, observation.size());
}

TEST(SmoTrainer, AssignNumericLabels) {
    Observation observation;
    LabeledObservationVector labelVector;
    //labelVector.push_back(std::make_pair("label_0", &observation));
    //labelVector.push_back(std::make_pair("label_2", &observation));
    //labelVector.push_back(std::make_pair("label_0", &observation));
    //labelVector.push_back(std::make_pair("label_2", &observation));
    labelVector.push_back(LabeledObservation(0, "label_0", &observation));
    labelVector.push_back(LabeledObservation(1, "label_2", &observation));
    labelVector.push_back(LabeledObservation(2, "label_0", &observation));
    labelVector.push_back(LabeledObservation(3, "label_2", &observation));

    std::vector<double> y(4);
    NumericClassToLabel discriminantToLabel;
    discriminantToLabel[-1] = "label_0";
    discriminantToLabel[+1] = "label_1";

    SmoTrainer t;
    t.assignNumericLabels(y, labelVector, discriminantToLabel);

    EXPECT_EQ(4, y.size());
    EXPECT_EQ(-1.0, y[0]);
    EXPECT_EQ(+1.0, y[1]);
    EXPECT_EQ(-1.0, y[2]);
    EXPECT_EQ(+1.0, y[3]);
}

// here we are testing for an exception if SmoTrainer.assignNumericLabels()
// gets more than two labels or fewer than two labels
TEST(SmoTrainer, MoreThanTwoLabels) {
    Observation fv;
    LabeledObservationVector oneLabelVector;
    //oneLabelVector.push_back(std::make_pair("label_0", &fv));
    //oneLabelVector.push_back(std::make_pair("label_0", &fv));
    //oneLabelVector.push_back(std::make_pair("label_0", &fv));
    oneLabelVector.push_back(LabeledObservation(0, "label_0", &fv));
    oneLabelVector.push_back(LabeledObservation(1, "label_0", &fv));
    oneLabelVector.push_back(LabeledObservation(2, "label_0", &fv));

    LabeledObservationVector threeLabelsVector;
    //threeLabelsVector.push_back(std::make_pair("label_0", &fv));
    //threeLabelsVector.push_back(std::make_pair("label_1", &fv));
    //threeLabelsVector.push_back(std::make_pair("label_2", &fv));
    threeLabelsVector.push_back(LabeledObservation(0, "label_0", &fv));
    threeLabelsVector.push_back(LabeledObservation(1, "label_1", &fv));
    threeLabelsVector.push_back(LabeledObservation(2, "label_2", &fv));

    std::vector<double> y(3);
    NumericClassToLabel discriminantToLabel;
    discriminantToLabel[-1] = "label_0";
    discriminantToLabel[+1] = "label_1";
    SmoTrainer t;
    EXPECT_THROW(t.assignNumericLabels(y, oneLabelVector, discriminantToLabel), std::exception);
    EXPECT_THROW(t.assignNumericLabels(y, threeLabelsVector, discriminantToLabel), std::exception);
}


// this class is a testing fixture
// with 8 data points in two classes
class EightPointDataset : public testing::Test {
public:
    LabeledObservationVector observationVector;
    Observation x_blue_0;
    Observation x_blue_1;
    Observation x_blue_2;
    Observation x_blue_3;
    Observation x_green_0;
    Observation x_green_1;
    Observation x_green_2;
    Observation x_green_3;

    virtual void SetUp() {
        //
        // blue data points
        //
        x_blue_0.push_back(1.0);
        x_blue_0.push_back(3.0);

        x_blue_1.push_back(2.0);
        x_blue_1.push_back(5.0);

        x_blue_2.push_back(3.0);
        x_blue_2.push_back(8.0);

        x_blue_3.push_back(6.0);
        x_blue_3.push_back(4.0);

        //
        // green data points
        //
        x_green_0.push_back(6.0);
        x_green_0.push_back(7.0);

        x_green_1.push_back(7.0);
        x_green_1.push_back(8.0);

        x_green_2.push_back(8.0);
        x_green_2.push_back(4.0);

        x_green_3.push_back(3.0);
        x_green_3.push_back(6.0);

        observationVector.push_back(LabeledObservation(0, "blue", &x_blue_0));
        observationVector.push_back(LabeledObservation(1, "blue", &x_blue_1));
        observationVector.push_back(LabeledObservation(2, "blue", &x_blue_2));
        observationVector.push_back(LabeledObservation(3, "blue", &x_blue_3));
        observationVector.push_back(LabeledObservation(4, "green", &x_green_0));
        observationVector.push_back(LabeledObservation(5, "green", &x_green_1));
        observationVector.push_back(LabeledObservation(6, "green", &x_green_2));
        observationVector.push_back(LabeledObservation(7, "green", &x_green_3));
    }

    virtual void TearDown() {

    }
};

// test SmoTrainer on eight data points
TEST_F(EightPointDataset, SmoTrainerTrain) {
    OneVsOneMultiClassSvmTrainer::standardizeObservations(observationVector);
    for ( ObservationVector::size_type i = 0; i < observationVector.size(); i++ ) {
        std::cout << "i = " << i;
        for ( Observation::size_type j = 0; j < observationVector[0].second->size(); j++ ) {
            std::cout << " " << observationVector[i].second->at(j);
        }
        std::cout << std::endl;
    }


    LinearKernelFunction linearKernelFunction;
    SmoTrainer t;
    SVM* svm = t.train(&linearKernelFunction, observationVector);

    EXPECT_EQ(-1, svm->discriminant(x_blue_0));
    EXPECT_EQ( 1, svm->discriminant(x_green_0));
    EXPECT_EQ(-1, svm->discriminant(x_blue_1));
    EXPECT_EQ( 1, svm->discriminant(x_green_1));

    EXPECT_EQ("blue", svm->classify(x_blue_0));
    EXPECT_EQ("green", svm->classify(x_green_0));

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

TEST(ParameterSetBuilder, Construct) {
    ParameterRangeMap m;
    m["a"].push_back(1);
    m["a"].push_back(2);
    m["a"].push_back(3);
    m["b"].push_back(1);
    m["b"].push_back(2);
    m["b"].push_back(3);
    m["c"].push_back(1);
    m["c"].push_back(2);
    m["c"].push_back(3);

    ParameterSetBuilder p(m);
    ParameterMapVector::const_iterator i = p.getParameterSetList().begin();
    for (; i != p.getParameterSetList().end(); i++) {
        ParameterMap pmap = *i;
        for ( ParameterMap::iterator j = pmap.begin(); j != pmap.end(); j++ ) {
            std::cout << j->first << " " << j->second << std::endl;
        }
    }
}

// use the eight point dataset fixture
TEST(KFoldLabeledObservationsDivider, TwoFoldTest) {
    Observation x1;
    Observation x2;
    Observation x3;
    Observation x4;

    LabeledObservationVector X;
    //X.push_back(make_pair("blue", &x1));
    //X.push_back(make_pair("green", &x2));
    //X.push_back(make_pair("blue", &x3));
    //X.push_back(make_pair("green", &x4));
    X.push_back(LabeledObservation(0, "blue", &x1));
    X.push_back(LabeledObservation(1, "green", &x2));
    X.push_back(LabeledObservation(2, "blue", &x3));
    X.push_back(LabeledObservation(3, "green", &x4));

    // we know the order the labeled observations
    // will be returned in training and test data
    // sets for each cross validation fold so we can
    // test that
    KFoldLabeledObservationsDivider d(2, X);

    // if end() is called before start() do we want an exception?
    EXPECT_EQ(true, d.end());
    EXPECT_EQ(2, d.getFoldNumber());

    d.start();

    EXPECT_EQ(false, d.end());
    EXPECT_EQ(0, d.getFoldNumber());

    EXPECT_EQ(2, d.getTrainingData().size());
    EXPECT_EQ("blue", d.getTrainingData()[0].first);
    EXPECT_EQ(&x3, d.getTrainingData()[0].second);
    EXPECT_EQ("green", d.getTrainingData()[1].first);
    EXPECT_EQ(&x4, d.getTrainingData()[1].second);

    EXPECT_EQ(2, d.getTestingData().size());
    EXPECT_EQ("blue", d.getTestingData()[0].first);
    EXPECT_EQ(&x1, d.getTestingData()[0].second);
    EXPECT_EQ("green", d.getTestingData()[1].first);
    EXPECT_EQ(&x2, d.getTestingData()[1].second);

    EXPECT_EQ(false, d.end());

    d.next();

    EXPECT_EQ(1, d.getFoldNumber());

    EXPECT_EQ(2, d.getTrainingData().size());
    EXPECT_EQ("blue", d.getTrainingData()[0].first);
    EXPECT_EQ(&x1, d.getTrainingData()[0].second);
    EXPECT_EQ("green", d.getTrainingData()[1].first);
    EXPECT_EQ(&x2, d.getTrainingData()[1].second);

    EXPECT_EQ(2, d.getTestingData().size());
    EXPECT_EQ("blue", d.getTestingData()[0].first);
    EXPECT_EQ(&x3, d.getTestingData()[0].second);
    EXPECT_EQ("green", d.getTestingData()[1].first);
    EXPECT_EQ(&x4, d.getTestingData()[1].second);

    d.next();

    EXPECT_EQ(true, d.end());
}

TEST(KFoldLabeledObservationsDivider, TwoFoldLoopTest) {
    Observation x1;
    Observation x2;
    Observation x3;
    Observation x4;

    LabeledObservationVector X;
    X.push_back(LabeledObservation(0, "blue", &x1));
    X.push_back(LabeledObservation(1, "green", &x2));
    X.push_back(LabeledObservation(2, "blue", &x3));
    X.push_back(LabeledObservation(3, "green", &x4));

    // we know the order the labeled observations
    // will be returned in training and test data
    // sets for each cross validation fold so we can
    // test that
    KFoldLabeledObservationsDivider d(2, X);

    // if end() is called before start() do we want an exception?
    EXPECT_EQ(true, d.end());

    int i = 0;
    for (d.start(); !d.end(); d.next()) {
        EXPECT_EQ(i, d.getFoldNumber());
        EXPECT_EQ(2, d.getTrainingData().size());
        EXPECT_EQ(2, d.getTestingData().size());
        i++;
    }
    EXPECT_EQ(2, i);
    EXPECT_EQ(2, d.getFoldNumber());

}

TEST(KFoldLabeledObservationsDivider, ThreeFoldLoopTest) {
    Observation x1;
    Observation x2;
    Observation x3;
    Observation x4;
    Observation x5;
    Observation x6;

    LabeledObservationVector X;
    //X.push_back(make_pair("blue", &x1));
    //X.push_back(make_pair("green", &x2));
    //X.push_back(make_pair("blue", &x3));
    //X.push_back(make_pair("green", &x4));
    //X.push_back(make_pair("blue", &x5));
    //X.push_back(make_pair("green", &x6));
    X.push_back(LabeledObservation(0, "blue", &x1));
    X.push_back(LabeledObservation(1, "green", &x2));
    X.push_back(LabeledObservation(2, "blue", &x3));
    X.push_back(LabeledObservation(3, "green", &x4));
    X.push_back(LabeledObservation(4, "blue", &x5));
    X.push_back(LabeledObservation(5, "green", &x6));

    // we know the order the labeled observations
    // will be returned in training and test data
    // sets for each cross validation fold so we can
    // test that
    KFoldLabeledObservationsDivider d(3, X);

    // if end() is called before start() do we want an exception?
    EXPECT_EQ(true, d.end());

    int i = 0;
    for (d.start(); !d.end(); d.next()) {
        EXPECT_EQ(i, d.getFoldNumber());
        EXPECT_EQ(4, d.getTrainingData().size());
        EXPECT_EQ(2, d.getTestingData().size());
        i++;
        std::cout << "fold " << i << " fold number " << d.getFoldNumber() << std::endl;
    }
    EXPECT_EQ(3, i);
    EXPECT_EQ(3, d.getFoldNumber());

}

TEST(OneVsOneMultiClassSvmTrainer, buildLabelSet) {
    Observation x1;
    Observation x2;
    Observation x3;

    LabeledObservationVector X;
    X.push_back(LabeledObservation(0, "blue", &x1));
    X.push_back(LabeledObservation(1, "green", &x2));
    X.push_back(LabeledObservation(2, "blue", &x3));

    LabelSet labelSet;

    buildLabelSet(labelSet, X);

    EXPECT_EQ(2, labelSet.size());
    EXPECT_EQ(1, labelSet.count("blue"));
    EXPECT_EQ(1, labelSet.count("green"));
}

// Build a label-to-labeled-observation-vector from a list
// of four observations with three different labels.
TEST(OneVsOneMultiClassSvmTrainer, buildLabelToLabeledObservationVector) {
    Observation x1;
    Observation x2;
    Observation x3;
    Observation x4;

    LabeledObservationVector X;
    X.push_back(LabeledObservation(0, "blue", &x1));
    X.push_back(LabeledObservation(1, "green", &x2));
    X.push_back(LabeledObservation(2, "blue", &x3));
    X.push_back(LabeledObservation(3, "red", &x4));

    LabelToLabeledObservationVector labelToLabeledObservationVector;
    buildLabelToLabeledObservationVector(labelToLabeledObservationVector, X);

    EXPECT_EQ(2, labelToLabeledObservationVector["blue"].size());
    EXPECT_EQ("blue", labelToLabeledObservationVector["blue"][0].first);
    EXPECT_EQ(&x1, labelToLabeledObservationVector["blue"][0].second);
    EXPECT_EQ("blue", labelToLabeledObservationVector["blue"][1].first);
    EXPECT_EQ(&x3, labelToLabeledObservationVector["blue"][1].second);

    EXPECT_EQ(1, labelToLabeledObservationVector["green"].size());
    EXPECT_EQ("green", labelToLabeledObservationVector["green"][0].first);
    EXPECT_EQ(&x2, labelToLabeledObservationVector["green"][0].second);

    EXPECT_EQ(1, labelToLabeledObservationVector["red"].size());
    EXPECT_EQ("red", labelToLabeledObservationVector["red"][0].first);
    EXPECT_EQ(&x4, labelToLabeledObservationVector["red"][0].second);
}

// Build a label pair set with one pair and another with three pairs.
TEST(OneVsOneMultiClassSvmTrainer, buildLabelPairSet) {
    Observation x1;
    Observation x2;
    Observation x3;
    Observation x4;

    LabeledObservationVector X;
    X.push_back(LabeledObservation(0, "blue", &x1));
    X.push_back(LabeledObservation(1, "green", &x2));
    X.push_back(LabeledObservation(2, "blue", &x3));

    LabelPairSet onePairLabelPairSet;

    OneVsOneMultiClassSvmTrainer::buildLabelPairSet(onePairLabelPairSet, X);

    EXPECT_EQ(1, onePairLabelPairSet.size());
    EXPECT_EQ(1, onePairLabelPairSet.count(buildLabelPair("blue","green")));

    X.push_back(LabeledObservation(3, "red", &x4));

    LabelPairSet threePairsLabelPairSet;

    OneVsOneMultiClassSvmTrainer::buildLabelPairSet(threePairsLabelPairSet, X);

    EXPECT_EQ(3, threePairsLabelPairSet.size());
    EXPECT_EQ(1, threePairsLabelPairSet.count(buildLabelPair("blue","green")));
    EXPECT_EQ(1, threePairsLabelPairSet.count(buildLabelPair("blue","red")));
    EXPECT_EQ(1, threePairsLabelPairSet.count(buildLabelPair("green","red")));
}

TEST(OneVsOneMultiClassSvmTrainer, appendTrainingAndTestingData) {
    //static void appendTrainingAndTestingData(Label, const ObservationVector&, ObservationVector&, LabelVector&, ObservationVector&, LabelVector&);
}

TEST(OneVsOneMultiClassSvmTrainer, standardizeObservations) {
    //static void standardizeObservations(const ObservationVector&);
}

TEST(OneVsOneMultiClassSvmTrainer, GetLabelSet) {
    LabeledObservationVector labeledObservationVector;
    ClassifySvmSharedCommand::readSharedAndDesignFiles("test.shared", "test.design", labeledObservationVector);
    EXPECT_EQ(2, labeledObservationVector.size());
    OneVsOneMultiClassSvmTrainer t(labeledObservationVector);

    const LabelSet& labelSet = t.getLabelSet();

    EXPECT_EQ(2, labelSet.size());
    EXPECT_EQ(1, labelSet.count("a"));
    EXPECT_EQ(1, labelSet.count("b"));
}


TEST_F(EightPointDataset, SvmRfe) {
    KernelParameterRangeMap kernelParameterRangeMap;
    getDefaultKernelParameterRangeMap(kernelParameterRangeMap);

    SvmRfe svmRfe;

    //FeatureLabelVector orderedFeatureList = svmRfe.getOrderedFeatureList(observationList, kernelParameterRangeMap);
}
