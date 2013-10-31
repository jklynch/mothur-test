//
//  svm_test.cpp
//
//  Created by Joshua Lynch on 6/19/2013.
//  Copyright (c) 2013 Schloss Lab. All rights reserved.
//

//
//  Run tests by name using command line option:
//      ./svm_test --gtest_filter=EightPointDataset.SmoTrainerTrain
//

#include "gtest/gtest.h"

#include "mothur/mothurout.h"
#include "mothur/groupmap.h"
#include "mothur/inputdata.h"

#include "mothur/classifysvmsharedcommand.h"
#include "mothur/svm.hpp"


MothurOut* MothurOut::_uniqueInstance = 0;


// can we instantiate ClassifySvmSharedCommand?
TEST(ClassifySvmSharedCommand, Instantiate) {
    ClassifySvmSharedCommand c("shared=x,design=y");
    EXPECT_EQ(4, c.getKernelParameterRangeMap().size());
}

TEST(ClassifySvmSharedCommand, KernelOption) {
    ClassifySvmSharedCommand c1("shared=x,design=y,kernel=linear,linearconstant=0.1");
    EXPECT_EQ(1, c1.getKernelParameterRangeMap().size());
    EXPECT_EQ(1, c1.getKernelParameterRangeMap().find("linear")->second.find("constant")->second.size());
    EXPECT_EQ(8, c1.getKernelParameterRangeMap().find("linear")->second.find("smoc")->second.size());

    ClassifySvmSharedCommand c2("shared=x,design=y,kernel=rbf");
    EXPECT_EQ(1, c2.getKernelParameterRangeMap().size());

    ClassifySvmSharedCommand c3("shared=x,design=y,kernel=polynomial");
    EXPECT_EQ(1, c3.getKernelParameterRangeMap().size());

    ClassifySvmSharedCommand c4("shared=x,design=y,kernel=linear-rbf");
    EXPECT_EQ(2, c4.getKernelParameterRangeMap().size());

    ClassifySvmSharedCommand c5("shared=x,design=y,kernel=linear-polynomial");
    EXPECT_EQ(2, c5.getKernelParameterRangeMap().size());
}

// can we read a shared file in a test?
// this is a test of my understanding rather than of code
TEST(ClassifySvmSharedCommand, ReadSharedFile) {
    MothurOut* m = MothurOut::getInstance();
    InputData input("test.shared", "sharedfile");
    vector<SharedRAbundVector*> lookup = input.getSharedRAbundVectors();

    int numObservations = lookup.size();
    EXPECT_EQ(2, numObservations);
    int numFeatures = lookup[0]->getNumBins();
    EXPECT_EQ(5, numFeatures);
    EXPECT_EQ("OTU_01", m->currentBinLabels[0]);
    EXPECT_EQ("OTU_02", m->currentBinLabels[1]);
}

// I'm not sure the behavior of this test is correct
TEST(ClassifySvmSharedCommand, ReadSharedAndDesignFiles) {
    MothurOut* m = MothurOut::getInstance();
    ClassifySvmSharedCommand classifySvmSharedCommand;

    LabeledObservationVector labeledObservationVector;
    FeatureVector featureVector;

    classifySvmSharedCommand.readSharedAndDesignFiles("test.shared", "test.design", labeledObservationVector, featureVector);

    EXPECT_EQ(2, labeledObservationVector.size());
    EXPECT_EQ(0, featureVector[0].getFeatureIndex());
    EXPECT_EQ(m->currentBinLabels[0], featureVector[0].getFeatureLabel());
    for (LabeledObservationVector::iterator i = labeledObservationVector.begin(); i != labeledObservationVector.end(); i++) {
        delete i->second;
    }
}

// test the behavior of the LabelPair typedef
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

TEST(Observation, Construct) {
    Observation observation(2);
    EXPECT_EQ(2, observation.size());
    observation[0] = 0.0;
    observation[1] = 0.0;
    EXPECT_EQ(2, observation.size());
}

// a little test for the OutputFilter class
TEST(OutputFilter, AllOptions) {
    OutputFilter quietFilter(OutputFilter::QUIET);
    OutputFilter infoFilter(OutputFilter::INFO);
    OutputFilter debugFilter(OutputFilter::DEBUG);
    OutputFilter traceFilter(OutputFilter::TRACE);

    EXPECT_EQ(false, quietFilter.info());
    EXPECT_EQ(false, quietFilter.debug());
    EXPECT_EQ(false, quietFilter.trace());

    EXPECT_EQ(true, infoFilter.info());
    EXPECT_EQ(false, infoFilter.debug());
    EXPECT_EQ(false, infoFilter.trace());

    EXPECT_EQ(true, debugFilter.info());
    EXPECT_EQ(true, debugFilter.debug());
    EXPECT_EQ(false, debugFilter.trace());

    EXPECT_EQ(true, traceFilter.info());
    EXPECT_EQ(true, traceFilter.debug());
    EXPECT_EQ(true, traceFilter.trace());
}


// this test puts together a trivial set of labeled observations
// in order to test assignment of numeric classes (+1.0 and -1.0)
TEST(SmoTrainer, AssignNumericLabels) {
    Observation observation;
    LabeledObservationVector labelVector;

    labelVector.push_back(LabeledObservation(0, "label_0", &observation));
    labelVector.push_back(LabeledObservation(1, "label_2", &observation));
    labelVector.push_back(LabeledObservation(2, "label_0", &observation));
    labelVector.push_back(LabeledObservation(3, "label_2", &observation));

    std::vector<double> y;
    NumericClassToLabel discriminantToLabel;

    ExternalSvmTrainingInterruption externalInterruption;

    OutputFilter outputFilter(OutputFilter::INFO);
    SmoTrainer t(externalInterruption, outputFilter);
    t.assignNumericLabels(y, labelVector, discriminantToLabel);

    EXPECT_EQ(4, y.size());
    EXPECT_EQ(-1.0, y[0]);
    EXPECT_EQ(+1.0, y[1]);
    EXPECT_EQ(-1.0, y[2]);
    EXPECT_EQ(+1.0, y[3]);
    EXPECT_EQ("label_0", discriminantToLabel[-1.0]);
    EXPECT_EQ("label_2", discriminantToLabel[+1.0]);
}

// here we are testing for an exception if SmoTrainer.assignNumericLabels()
// gets more than two labels or fewer than two labels
TEST(SmoTrainer, MoreThanTwoLabels) {
    Observation fv;
    LabeledObservationVector oneLabelVector;

    oneLabelVector.push_back(LabeledObservation(0, "label_0", &fv));
    oneLabelVector.push_back(LabeledObservation(1, "label_0", &fv));
    oneLabelVector.push_back(LabeledObservation(2, "label_0", &fv));

    LabeledObservationVector threeLabelsVector;

    threeLabelsVector.push_back(LabeledObservation(0, "label_0", &fv));
    threeLabelsVector.push_back(LabeledObservation(1, "label_1", &fv));
    threeLabelsVector.push_back(LabeledObservation(2, "label_2", &fv));

    std::vector<double> y;
    NumericClassToLabel discriminantToLabel;

    ExternalSvmTrainingInterruption externalInterruption;
    OutputFilter outputFilter(OutputFilter::INFO);

    SmoTrainer t(externalInterruption, outputFilter);
    EXPECT_THROW(t.assignNumericLabels(y, oneLabelVector, discriminantToLabel), SmoTrainerException);
    EXPECT_THROW(t.assignNumericLabels(y, threeLabelsVector, discriminantToLabel), SmoTrainerException);
}


// this class is a testing fixture
// with 8 data points in two classes, 'blue' and 'green'
// this data is not perfectly separable, as can be seen below:
//
//  8           B2          g1
//
//  7                    g0
//
//  6           g3
//
//  5        B1
//
//  4                    B3    g2
//
//  3     B0
//
//  2
//
//  1
//
//  0
//     0  1  2  3  4  5  6  7  8  9
//
//  We should expect data points B3 and g3 to be misclassified
//  even in the best case.
//
class EightPointLabeledObservationVector : public testing::Test {
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
    Feature featureA;
    Feature featureB;
    FeatureVector featureVector;

    EightPointLabeledObservationVector() : featureA(0, "feature A"), featureB(1, "feature B") {}


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

        featureVector.push_back(featureA);
        featureVector.push_back(featureB);
    }
};


class EightPointDataset : public EightPointLabeledObservationVector {
public:
    SvmDataset* svmDataset;
    virtual void SetUp() {
        EightPointLabeledObservationVector::SetUp();
        transformZeroOne(observationVector);

        svmDataset = new SvmDataset(observationVector, featureVector);
    }

    virtual void TearDown() {
        delete svmDataset;
    }
};

TEST(StdThreshold, StdThreshold) {
    Observation x_blue_0;
    Observation x_blue_1;
    Observation x_blue_2;
    Observation x_blue_3;

    // feature B has low standard deviation

    x_blue_0.push_back(0.0);
    x_blue_0.push_back(3.0);
    x_blue_0.push_back(0.0);
    x_blue_0.push_back(3.0);
    x_blue_0.push_back(0.0);

    x_blue_1.push_back(0.0);
    x_blue_1.push_back(2.0);
    x_blue_1.push_back(0.0);
    x_blue_1.push_back(2.0);
    x_blue_1.push_back(0.0);

    x_blue_2.push_back(0.0);
    x_blue_2.push_back(1.0);
    x_blue_2.push_back(0.0);
    x_blue_2.push_back(1.0);
    x_blue_2.push_back(0.0);

    x_blue_3.push_back(0.0);
    x_blue_3.push_back(0.0);
    x_blue_3.push_back(0.0);
    x_blue_3.push_back(0.0);
    x_blue_3.push_back(0.0);

    LabeledObservationVector observationVector;
    observationVector.push_back(LabeledObservation(0, "blue", &x_blue_0));
    observationVector.push_back(LabeledObservation(1, "blue", &x_blue_1));
    observationVector.push_back(LabeledObservation(2, "blue", &x_blue_2));
    observationVector.push_back(LabeledObservation(3, "blue", &x_blue_3));

    Feature featureA(0, "feature A");
    Feature featureB(1, "feature B");
    Feature featureC(2, "feature C");
    Feature featureD(3, "feature D");
    Feature featureE(4, "feature E");
    FeatureVector featureVector;
    featureVector.push_back(featureA);
    featureVector.push_back(featureB);
    featureVector.push_back(featureC);
    featureVector.push_back(featureD);
    featureVector.push_back(featureE);

    EXPECT_EQ(4, observationVector.size());
    EXPECT_EQ(5, observationVector[0].second->size());
    FeatureVector removedFeatureList = applyStdThreshold(0.5, observationVector, featureVector);
    EXPECT_EQ(2, observationVector[0].second->size());
    EXPECT_EQ("feature B", featureVector.at(0).getFeatureLabel());
    EXPECT_EQ(0, featureVector.at(0).getFeatureIndex());

    EXPECT_EQ("feature D", featureVector.at(1).getFeatureLabel());
    EXPECT_EQ(1, featureVector.at(1).getFeatureIndex());

    EXPECT_EQ(3, removedFeatureList.size());

    EXPECT_EQ("feature A", removedFeatureList.at(0).getFeatureLabel());
    EXPECT_EQ(-1, removedFeatureList.at(0).getFeatureIndex());
    EXPECT_EQ("feature C", removedFeatureList.at(1).getFeatureLabel());
    EXPECT_EQ("feature E", removedFeatureList.at(2).getFeatureLabel());


}

TEST_F(EightPointLabeledObservationVector, MinMax) {
    EXPECT_EQ(1.0, getMinimumFeatureValueForObservation(featureA.getFeatureIndex(), observationVector));
    EXPECT_EQ(8.0, getMaximumFeatureValueForObservation(featureA.getFeatureIndex(), observationVector));

    EXPECT_EQ(3.0, getMinimumFeatureValueForObservation(featureB.getFeatureIndex(), observationVector));
    EXPECT_EQ(8.0, getMaximumFeatureValueForObservation(featureB.getFeatureIndex(), observationVector));
}

TEST_F(EightPointLabeledObservationVector, TransformZeroOne) {
    transformZeroOne(observationVector);
    EXPECT_EQ(0.0, getMinimumFeatureValueForObservation(featureA.getFeatureIndex(), observationVector));
    EXPECT_EQ(1.0, getMaximumFeatureValueForObservation(featureA.getFeatureIndex(), observationVector));

    EXPECT_EQ(0.0, getMinimumFeatureValueForObservation(featureB.getFeatureIndex(), observationVector));
    EXPECT_EQ(1.0, getMaximumFeatureValueForObservation(featureB.getFeatureIndex(), observationVector));
}

TEST_F(EightPointLabeledObservationVector, KernelFunctionCache) {
    KernelFunctionFactory kernelFunctionFactory(observationVector);
    KernelFunction& linearKernelFunction = kernelFunctionFactory.getKernelFunctionForKey(
        LinearKernelFunction::MapKey
    );

    ParameterMap m;
    m.insert(std::make_pair(LinearKernelFunction::MapKey_Constant, 1.0));
    linearKernelFunction.setParameters(m);

    KernelFunctionCache linearKernelFunctionCache(linearKernelFunction, observationVector);

    std::cout << "checking row 0" << std::endl;
    EXPECT_EQ(true, linearKernelFunction.rowNotCached(0));
    EXPECT_EQ(true, linearKernelFunctionCache.rowNotCached(0));

    std::cout << "checking row 1" << std::endl;
    EXPECT_EQ(true, linearKernelFunction.rowNotCached(1));
    EXPECT_EQ(true, linearKernelFunctionCache.rowNotCached(1));

    std::cout << "checking value 0,0" << std::endl;
    EXPECT_EQ(10.0, linearKernelFunction.calculateParameterFreeSimilarity(observationVector[0], observationVector[0]));
    EXPECT_EQ(11.0, linearKernelFunction.similarity(observationVector[0], observationVector[0]));
    EXPECT_EQ(11.0, linearKernelFunctionCache.similarity(observationVector[0], observationVector[0]));

    std::cout << "checking value 0,1" << std::endl;
    EXPECT_EQ(17.0, linearKernelFunction.calculateParameterFreeSimilarity(observationVector[0], observationVector[1]));
    EXPECT_EQ(18.0, linearKernelFunction.similarity(observationVector[0], observationVector[1]));
    EXPECT_EQ(18.0, linearKernelFunctionCache.similarity(observationVector[0], observationVector[1]));

    EXPECT_EQ(false, linearKernelFunction.rowNotCached(0));
    EXPECT_EQ(false, linearKernelFunctionCache.rowNotCached(0));

    EXPECT_EQ(true, linearKernelFunctionCache.rowNotCached(1));
}


// test SmoTrainer on eight data points
TEST_F(EightPointDataset, SmoTrainerTrain) {
    transformZeroMeanUnitVariance(observationVector);
    for ( ObservationVector::size_type i = 0; i < observationVector.size(); i++ ) {
        std::cout << "i = " << i;
        for ( Observation::size_type j = 0; j < observationVector[0].second->size(); j++ ) {
            std::cout << " " << observationVector[i].second->at(j);
        }
        std::cout << std::endl;
    }

    LinearKernelFunction linearKernelFunction(observationVector);
    KernelFunctionCache linearKernelFunctionCache(linearKernelFunction, observationVector);

    ExternalSvmTrainingInterruption externalInterruption;
    OutputFilter outputFilter(OutputFilter::INFO);

    SmoTrainer t(externalInterruption, outputFilter);
    // the default C is 1.0 which results in misclassification of x_blue_1
    t.setC(0.1);
    SVM* svm = t.train(linearKernelFunctionCache, observationVector);

    EXPECT_EQ(-1, svm->discriminant(x_blue_0));
    EXPECT_EQ( 1, svm->discriminant(x_green_0));

    EXPECT_EQ("blue", svm->classify(x_blue_0));
    EXPECT_EQ("green", svm->classify(x_green_0));

    EXPECT_EQ(-1, svm->discriminant(x_blue_1));
    EXPECT_EQ( 1, svm->discriminant(x_green_1));
    EXPECT_EQ(-1, svm->discriminant(x_blue_2));
    EXPECT_EQ( 1, svm->discriminant(x_green_2));

    // we expect x_blue_3 and x_green_3 to be mis-classified
    EXPECT_EQ( 1, svm->discriminant(x_blue_3));
    EXPECT_EQ(-1, svm->discriminant(x_green_3));

    EXPECT_EQ("green", svm->classify(x_blue_3));
    EXPECT_EQ("blue", svm->classify(x_green_3));

    delete svm;
}


// test SmoTrainer on eight data points with RBF kernel
TEST_F(EightPointDataset, SmoTrainerRbfKernelTrain) {
    // zero mean, unit variance works better than zero to one for rbf on this dataset
    transformZeroMeanUnitVariance(observationVector);
    for ( ObservationVector::size_type i = 0; i < observationVector.size(); i++ ) {
        std::cout << "i = " << i;
        for ( Observation::size_type j = 0; j < observationVector[0].second->size(); j++ ) {
            std::cout << " " << observationVector[i].second->at(j);
        }
        std::cout << std::endl;
    }

    RbfKernelFunction kernelFunction(observationVector);
    KernelFunctionCache kernelFunctionCache(kernelFunction, observationVector);

    ExternalSvmTrainingInterruption externalInterruption;
    OutputFilter outputFilter(OutputFilter::INFO);

    SmoTrainer t(externalInterruption, outputFilter);
    SVM* svm = t.train(kernelFunctionCache, observationVector);

    EXPECT_EQ(-1, svm->discriminant(x_blue_0));
    EXPECT_EQ( 1, svm->discriminant(x_green_0));
    EXPECT_EQ(-1, svm->discriminant(x_blue_1));
    EXPECT_EQ( 1, svm->discriminant(x_green_1));

    EXPECT_EQ("blue", svm->classify(x_blue_0));
    EXPECT_EQ("green", svm->classify(x_green_0));

    delete svm;
}


// test SmoTrainer on eight data points with polynomial kernel
TEST_F(EightPointDataset, SmoTrainerPolynomialKernelTrain) {
    for ( ObservationVector::size_type i = 0; i < observationVector.size(); i++ ) {
        std::cout << "i = " << i;
        for ( Observation::size_type j = 0; j < observationVector[0].second->size(); j++ ) {
            std::cout << " " << observationVector[i].second->at(j);
        }
        std::cout << std::endl;
    }

    RbfKernelFunction kernelFunction(observationVector);
    KernelFunctionCache kernelFunctionCache(kernelFunction, observationVector);

    ExternalSvmTrainingInterruption externalInterruption;
    OutputFilter outputFilter(1);
    SmoTrainer t(externalInterruption, outputFilter);
    SVM* svm = t.train(kernelFunctionCache, observationVector);

    EXPECT_EQ(-1, svm->discriminant(x_blue_0));
    EXPECT_EQ( 1, svm->discriminant(x_green_0));
    EXPECT_EQ(-1, svm->discriminant(x_blue_1));
    EXPECT_EQ( 1, svm->discriminant(x_green_1));

    EXPECT_EQ("blue", svm->classify(x_blue_0));
    EXPECT_EQ("green", svm->classify(x_green_0));

    delete svm;
}


// test SmoTrainer on eight data points with sigmoid kernel
TEST_F(EightPointDataset, SmoTrainerSigmoidKernelTrain) {
    for ( ObservationVector::size_type i = 0; i < observationVector.size(); i++ ) {
        std::cout << "i = " << i;
        for ( Observation::size_type j = 0; j < observationVector[0].second->size(); j++ ) {
            std::cout << " " << observationVector[i].second->at(j);
        }
        std::cout << std::endl;
    }

    SigmoidKernelFunction kernelFunction(observationVector);
    KernelFunctionCache kernelFunctionCache(kernelFunction, observationVector);

    ExternalSvmTrainingInterruption externalInterruption;

    OutputFilter outputFilter(OutputFilter::INFO);

    SmoTrainer t(externalInterruption, outputFilter);
    SVM* svm = t.train(kernelFunctionCache, observationVector);

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

    ExternalSvmTrainingInterruption externalInterruption;

    OutputFilter outputFilter(2);

    SmoTrainer t(externalInterruption, outputFilter);
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
    //m["b"].push_back(3);
    m["c"].push_back(1);
    //m["c"].push_back(2);
    //m["c"].push_back(3);

    int parameterSetCount = 0;

    // ParameterSetBuilder should construct these sets:
    //    {a:1, b:1, c:1}
    //    {a:1, b:2, c:1}
    //    {a:2, b:1, c:1}
    //    {a:2, b:2, c:1}
    //    {a:3, b:1, c:1}
    //    {a:3, b:2, c:1}
    ParameterSetBuilder p(m);
    ParameterMapVector::const_iterator i = p.getParameterSetList().begin();
    for (; i != p.getParameterSetList().end(); i++) {
        ParameterMap pmap = *i;
        for ( ParameterMap::iterator j = pmap.begin(); j != pmap.end(); j++ ) {
            //std::cout << j->first << " " << j->second << std::endl;
        }
        parameterSetCount++;
    }

    EXPECT_EQ(6, parameterSetCount);
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

TEST(OneVsOneMultiClassSvmTrainer, GetLabelSet) {
    MothurOut* m = MothurOut::getInstance();
    ClassifySvmSharedCommand classifySvmSharedCommand;

    LabeledObservationVector labeledObservationVector;
    FeatureVector featureVector;

    ExternalSvmTrainingInterruption externalInterruption;

    classifySvmSharedCommand.readSharedAndDesignFiles("test.shared", "test.design", labeledObservationVector, featureVector);
    SvmDataset svmDataset(labeledObservationVector, featureVector);
    EXPECT_EQ(2, labeledObservationVector.size());

    OutputFilter outputFilter(OutputFilter::INFO);

    int evaluationFoldCount = 3;
    int trainFoldCount = 5;
    OneVsOneMultiClassSvmTrainer t(svmDataset, evaluationFoldCount, trainFoldCount, externalInterruption, outputFilter);

    const LabelSet& labelSet = t.getLabelSet();

    EXPECT_EQ(2, labelSet.size());
    EXPECT_EQ(1, labelSet.count("a"));
    EXPECT_EQ(1, labelSet.count("b"));
}

class TestExternalSvmTrainingInterruption : public ExternalSvmTrainingInterruption {
public:
    bool interruptTraining() { std::cout << "ha!" << std::endl;  return true; }
};

TEST_F(EightPointDataset, OneVsOneMultiClassSvmTrainer_ExternalSvmTrainingInterruption) {
    MothurOut* m = MothurOut::getInstance();
    ClassifySvmSharedCommand classifySvmSharedCommand;

    LabeledObservationVector labeledObservationVector;
    FeatureVector featureVector;

    int evaluationFoldCount = 2;
    int trainFoldCount = 2;
    TestExternalSvmTrainingInterruption testExternalInterruption;

    OutputFilter outputFilter(OutputFilter::INFO);

    OneVsOneMultiClassSvmTrainer t(*svmDataset, evaluationFoldCount, trainFoldCount, testExternalInterruption, outputFilter);

    KernelParameterRangeMap kernelParameterRangeMap;
    getDefaultKernelParameterRangeMap(kernelParameterRangeMap);

    EXPECT_THROW(t.train(kernelParameterRangeMap), SvmTrainingInterruptedException);
}
