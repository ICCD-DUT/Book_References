/*
Copyright (c) 2016, TU Dresden
Copyright (c) 2017, Heidelberg University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the TU Dresden, Heidelberg University nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TU DRESDEN OR HEIDELBERG UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include <iostream>
#include <fstream>

#include "properties.h"
#include "thread_rand.h"
#include "util.h"
#include "stop_watch.h"
#include "dataset.h"

#include "lua_calls.h"
#include "cnn.h"

int main(int argc, const char* argv[])
{
    // read parameters
    GlobalProperties* gp = GlobalProperties::getInstance();
    gp->parseConfig();
    gp->parseCmdLine(argc, argv);

    int objHyps = gp->tP.ransacIterations;
    int inlierThreshold2D = gp->tP.ransacInlierThreshold;
    int refSteps = gp->tP.ransacRefinementIterations;  
    
    std::string baseScriptRGB = gp->tP.objScript;
    std::string baseScriptObj = gp->tP.scoreScript;
    std::string modelFileRGB = gp->tP.objModel;

    // setup data and torch
    std::cout << std::endl << BLUETEXT("Loading test set ...") << std::endl;
    jp::Dataset testDataset = jp::Dataset(gp->tP.testFolder);

    // lua and models
    std::ofstream testFile;
    testFile.open(gp->tP.testResultsFile+"ransac_test_loss_"+baseScriptRGB+"_rdraw"+intToString(gp->tP.randomDraw)+"_"+gp->tP.sessionString+".csv"); // contains evaluation information for the whole test sequence
    
    if(gp->tinP.ifContinue){     
        for (unsigned lin_i=gp->tinP.trainingRoundsStart; lin_i<gp->tinP.trainingRounds;lin_i+=gp->tP.testStep){
            std::cout << "Loading script: " << baseScriptObj << std::endl;
            lua_State* stateObj = luaL_newstate();
            luaL_openlibs(stateObj);
            execute(baseScriptObj.c_str(), stateObj);
            setGPU(gp->tP.GPUNo,stateObj);
            setStoreCounter(gp->tinP.trainingStoreCounter,stateObj);
            loadScore(inlierThreshold2D, gp->getCNNOutputDimX(), gp->getCNNOutputDimY(), stateObj);  

            std::cout << "Loading script: " << baseScriptRGB << std::endl;
            lua_State* stateRGB = luaL_newstate();
            luaL_openlibs(stateRGB);
            execute(baseScriptRGB.c_str(), stateRGB); 
            setGPU(gp->tP.GPUNo,stateRGB);
            setStoreCounter(gp->tinP.trainingStoreCounter,stateRGB);
            loadModel(modelFileRGB+intToString(lin_i), gp->getCNNInputDimX(), gp->getCNNInputDimY(), gp->getCNNOutputDimX(), gp->getCNNOutputDimY(), stateRGB);
            
            cv::Mat camMat = gp->getCamMat();
            
            
            setEvaluate(stateRGB);
            setEvaluate(stateObj);
            
            double avgCorrect = 0;
            double avgCorrect6 = 0;
            double avgCorrect7 = 0;
            double avgCorrect8 = 0;
            double avgCorrect9 = 0;
            double avgCorrect10 = 0;
            
            std::vector<double> expLosses;
            std::vector<double> sfEntropies;
            std::vector<double> rotErrs;
            std::vector<double> tErrs;
            
            std::ofstream testErrFile;

            testErrFile.open(intToString(lin_i)+"ransac_test_errors_"+baseScriptRGB+"_rdraw"+intToString(gp->tP.randomDraw)+"_"+gp->tP.sessionString+".txt"); // contains evaluation information for each test image
            for(unsigned i = 0; i < testDataset.size(); i+= gp->dP.imageSubSample)
            {
                std::cout << YELLOWTEXT("Processing test image " << i << " of " << testDataset.size()) << "." << std::endl;

                // load test image
                jp::img_bgr_t testRGB;
                testDataset.getBGR(i, testRGB);

                jp::cv_trans_t hypGT;
                testDataset.getPose(i, hypGT);

                std::cout << BLUETEXT("Predicting object coordinates.") << std::endl;

                cv::Mat_<cv::Point2i> sampling;
                std::vector<cv::Mat_<cv::Vec3f>> imgMaps;
                jp::img_coord_t estObj = getCoordImg(testRGB, sampling, imgMaps, false, stateRGB);

                // process frame (same function used in training, hence most of the variables below are not used here), see method documentation for parameter explanation
                std::vector<jp::cv_trans_t> refHyps;
                std::vector<double> sfScores;
                std::vector<std::vector<cv::Point2i>> sampledPoints;
                std::vector<double> losses;
                std::vector<cv::Mat_<int>> inlierMaps;
                double tErr;
                double rotErr;
                int hypIdx;

                double expectedLoss;
                double sfEntropy;
                bool correct;
                bool correct6;
                bool correct7;
                bool correct8;
                bool correct9;
                bool correct10;

                processImage(
                    hypGT,
                    stateObj,
                    objHyps,
                    camMat,
                    inlierThreshold2D,
                    refSteps,
                    expectedLoss,
                    sfEntropy,
                    correct,
                    correct6,
                    correct7,
                    correct8,
                    correct9,
                    correct10,
                    refHyps,
                    sfScores,
                    estObj,
                    sampling,
                    sampledPoints,
                    losses,
                    inlierMaps,
                    tErr,
                    rotErr,
                    hypIdx,
                    false);

                avgCorrect += correct;
                avgCorrect6 += correct6;
                avgCorrect7 += correct7;
                avgCorrect8 += correct8;
                avgCorrect9 += correct9;
                avgCorrect10 += correct10;
                std::cout << BLUETEXT(avgCorrect<<"---"<<avgCorrect/2000<<"%") << std::endl;

                //avgCorrect += correct;

                // invert pose to get camera pose (we estimated the scene pose)
                jp::cv_trans_t invHyp = getInvHyp(refHyps[hypIdx]);

                testErrFile
                    << expectedLoss << " "      // 0  - expected loss over the hypothesis pool
                    << sfEntropy << " "         // 1  - entropy of the hypothesis score distribution
                    << losses[hypIdx] << " "    // 2  - loss of the selected hypothesis
                    << tErr << " "              // 3  - translational error in m
                    << rotErr << " "            // 4  - rotational error in deg
                    << invHyp.first.at<double>(0, 0) << " "     // 5  - selected pose, rotation (1st component of Rodriguez vector)
                    << invHyp.first.at<double>(1, 0) << " "     // 6  - selected pose, rotation (2nd component of Rodriguez vector)
                    << invHyp.first.at<double>(2, 0) << " "     // 7  - selected pose, rotation (3th component of Rodriguez vector)
                    << invHyp.second.at<double>(0, 0) << " "    // 8  - selected pose, translation in m (x)
                    << invHyp.second.at<double>(0, 1) << " "    // 9  - selected pose, translation in m (y)
                    << invHyp.second.at<double>(0, 2) << " "    // 10 - selected pose, translation in m (z)
                    << std::endl;

                expLosses.push_back(expectedLoss);
                sfEntropies.push_back(sfEntropy);
                tErrs.push_back(tErr);
                rotErrs.push_back(rotErr);
            }
            // mean and stddev of loss
            std::vector<double> lossMean;
            std::vector<double> lossStdDev;
            cv::meanStdDev(expLosses, lossMean, lossStdDev);
            
            // mean and stddev of score entropy
            std::vector<double> entropyMean;
            std::vector<double> entropyStdDev;
            cv::meanStdDev(sfEntropies, entropyMean, entropyStdDev);
            
            avgCorrect /= testDataset.size() / gp->dP.imageSubSample;
            avgCorrect6 /= testDataset.size() / gp->dP.imageSubSample;
            avgCorrect7 /= testDataset.size() / gp->dP.imageSubSample;
            avgCorrect8 /= testDataset.size() / gp->dP.imageSubSample;
            avgCorrect9 /= testDataset.size() / gp->dP.imageSubSample;
            avgCorrect10 /= testDataset.size() / gp->dP.imageSubSample;
            
            // median of rotational and translational errors
            std::sort(rotErrs.begin(), rotErrs.end());
            std::sort(tErrs.begin(), tErrs.end());
            
            double medianRotErr = rotErrs[rotErrs.size() / 2];
            double medianTErr = tErrs[tErrs.size() / 2];
            
            std::cout << "-----------------------------------------------------------" << std::endl;
            std::cout << BLUETEXT("Avg. test loss: " << lossMean[0] << ", accuracy: " << avgCorrect * 100 << "%") << std::endl;
            std::cout << BLUETEXT("accuracy: " << avgCorrect6 * 100 << "%") << std::endl;
            std::cout << BLUETEXT("accuracy: " << avgCorrect7 * 100 << "%") << std::endl;
            std::cout << BLUETEXT("accuracy: " << avgCorrect8 * 100 << "%") << std::endl;
            std::cout << BLUETEXT("accuracy: " << avgCorrect9 * 100 << "%") << std::endl;
            std::cout << BLUETEXT("accuracy: " << avgCorrect10 * 100 << "%") << std::endl;
            std::cout << "Median Rot. Error: " << medianRotErr << "deg, Median T. Error: " << medianTErr * 100 << "cm." << std::endl;

            testFile
                    << intToString(lin_i) << ","            // 0 - percentage of correct poses
                    << avgCorrect * 100 << "%"<< ","            // 0 - percentage of correct poses
                    << avgCorrect6 * 100 << "%"<< ","            // 0 - percentage of correct poses
                    << avgCorrect7 * 100 << "%"<< ","            // 0 - percentage of correct poses
                    << avgCorrect8 * 100 << "%"<< ","            // 0 - percentage of correct poses
                    << avgCorrect9 * 100 << "%"<< ","            // 0 - percentage of correct poses
                    << avgCorrect10 * 100 << "%"<< ","            // 0 - percentage of correct poses
                    << lossMean[0] << ","           // 1 - mean loss of selected hypotheses
                    << lossStdDev[0] << ","         // 2 - standard deviation of losses of selected hypotheses
                    << entropyMean[0] << ","        // 3 - mean of the score distribution entropy
                    << entropyStdDev[0] << ","      // 4 - standard deviation of the score distribution entropy
                    << medianRotErr << ","          // 5 - median rotational error of selected hypotheses
                    << medianTErr                   // 6 - median translational error (in m) of selected hypotheses
                    << std::endl;
            testErrFile.close();
            lua_close(stateRGB);
            lua_close(stateObj);
        }
    }
    else{
        std::cout << "Loading script: " << baseScriptObj << std::endl;
        lua_State* stateObj = luaL_newstate();
        luaL_openlibs(stateObj);
        execute(baseScriptObj.c_str(), stateObj);
        setGPU(gp->tP.GPUNo,stateObj);
        setStoreCounter(gp->tinP.trainingStoreCounter,stateObj);
        loadScore(inlierThreshold2D, gp->getCNNOutputDimX(), gp->getCNNOutputDimY(), stateObj);

        std::cout << "Loading script: " << baseScriptRGB << std::endl;
        lua_State* stateRGB = luaL_newstate();
        luaL_openlibs(stateRGB);
        execute(baseScriptRGB.c_str(), stateRGB);
        setGPU(gp->tP.GPUNo,stateRGB);
        setStoreCounter(gp->tinP.trainingStoreCounter,stateRGB);

        loadModel(modelFileRGB, gp->getCNNInputDimX(), gp->getCNNInputDimY(), gp->getCNNOutputDimX(), gp->getCNNOutputDimY(), stateRGB);
    
        cv::Mat camMat = gp->getCamMat();
        
        
        setEvaluate(stateRGB);
        setEvaluate(stateObj);
        
        double avgCorrect = 0;
        double avgCorrect6 = 0;
        double avgCorrect7 = 0;
        double avgCorrect8 = 0;
        double avgCorrect9 = 0;
        double avgCorrect10 = 0;
        
        std::vector<double> expLosses;
        std::vector<double> sfEntropies;
        std::vector<double> rotErrs;
        std::vector<double> tErrs;
        
        std::ofstream testErrFile;
        testErrFile.open("ransac_test_errors_"+baseScriptRGB+"_rdraw"+intToString(gp->tP.randomDraw)+"_"+gp->tP.sessionString+".txt"); // contains evaluation information for each test image
        for(unsigned i = 0; i < testDataset.size(); i+= gp->dP.imageSubSample)
        {
            std::cout << YELLOWTEXT("Processing test image " << i << " of " << testDataset.size()) << "." << std::endl;

            // load test image
            jp::img_bgr_t testRGB;
            testDataset.getBGR(i, testRGB);

            jp::cv_trans_t hypGT;
            testDataset.getPose(i, hypGT);

            std::cout << BLUETEXT("Predicting object coordinates.") << std::endl;

            cv::Mat_<cv::Point2i> sampling;
            std::vector<cv::Mat_<cv::Vec3f>> imgMaps;
            jp::img_coord_t estObj = getCoordImg(testRGB, sampling, imgMaps, false, stateRGB);

            // process frame (same function used in training, hence most of the variables below are not used here), see method documentation for parameter explanation
            std::vector<jp::cv_trans_t> refHyps;
            std::vector<double> sfScores;
            std::vector<std::vector<cv::Point2i>> sampledPoints;
            std::vector<double> losses;
            std::vector<cv::Mat_<int>> inlierMaps;
            double tErr;
            double rotErr;
            int hypIdx;

            double expectedLoss;
            double sfEntropy;
            bool correct;
            bool correct6;
            bool correct7;
            bool correct8;
            bool correct9;
            bool correct10;

            processImage(
                hypGT,
                stateObj,
                objHyps,
                camMat,
                inlierThreshold2D,
                refSteps,
                expectedLoss,
                sfEntropy,
                correct,
                correct6,
                correct7,
                correct8,
                correct9,
                correct10,
                refHyps,
                sfScores,
                estObj,
                sampling,
                sampledPoints,
                losses,
                inlierMaps,
                tErr,
                rotErr,
                hypIdx,
                false);

            avgCorrect += correct;
            avgCorrect6 += correct6;
            avgCorrect7 += correct7;
            avgCorrect8 += correct8;
            avgCorrect9 += correct9;
            avgCorrect10 += correct10;
            std::cout << BLUETEXT(avgCorrect<<"---"<<avgCorrect/2000<<"%") << std::endl;

            //avgCorrect += correct;

            // invert pose to get camera pose (we estimated the scene pose)
            jp::cv_trans_t invHyp = getInvHyp(refHyps[hypIdx]);

            testErrFile
                << expectedLoss << " "      // 0  - expected loss over the hypothesis pool
                << sfEntropy << " "         // 1  - entropy of the hypothesis score distribution
                << losses[hypIdx] << " "    // 2  - loss of the selected hypothesis
                << tErr << " "              // 3  - translational error in m
                << rotErr << " "            // 4  - rotational error in deg
                << invHyp.first.at<double>(0, 0) << " "     // 5  - selected pose, rotation (1st component of Rodriguez vector)
                << invHyp.first.at<double>(1, 0) << " "     // 6  - selected pose, rotation (2nd component of Rodriguez vector)
                << invHyp.first.at<double>(2, 0) << " "     // 7  - selected pose, rotation (3th component of Rodriguez vector)
                << invHyp.second.at<double>(0, 0) << " "    // 8  - selected pose, translation in m (x)
                << invHyp.second.at<double>(0, 1) << " "    // 9  - selected pose, translation in m (y)
                << invHyp.second.at<double>(0, 2) << " "    // 10 - selected pose, translation in m (z)
                << std::endl;

            expLosses.push_back(expectedLoss);
            sfEntropies.push_back(sfEntropy);
            tErrs.push_back(tErr);
            rotErrs.push_back(rotErr);
        }
        // mean and stddev of loss
        std::vector<double> lossMean;
        std::vector<double> lossStdDev;
        cv::meanStdDev(expLosses, lossMean, lossStdDev);
        
        // mean and stddev of score entropy
        std::vector<double> entropyMean;
        std::vector<double> entropyStdDev;
        cv::meanStdDev(sfEntropies, entropyMean, entropyStdDev);
        
        avgCorrect /= testDataset.size() / gp->dP.imageSubSample;
        avgCorrect6 /= testDataset.size() / gp->dP.imageSubSample;
        avgCorrect7 /= testDataset.size() / gp->dP.imageSubSample;
        avgCorrect8 /= testDataset.size() / gp->dP.imageSubSample;
        avgCorrect9 /= testDataset.size() / gp->dP.imageSubSample;
        avgCorrect10 /= testDataset.size() / gp->dP.imageSubSample;
        
        // median of rotational and translational errors
        std::sort(rotErrs.begin(), rotErrs.end());
        std::sort(tErrs.begin(), tErrs.end());
        
        double medianRotErr = rotErrs[rotErrs.size() / 2];
        double medianTErr = tErrs[tErrs.size() / 2];
        
        std::cout << "-----------------------------------------------------------" << std::endl;
        std::cout << BLUETEXT("Avg. test loss: " << lossMean[0] << ", accuracy: " << avgCorrect * 100 << "%") << std::endl;
        std::cout << BLUETEXT("accuracy: " << avgCorrect6 * 100 << "%") << std::endl;
        std::cout << BLUETEXT("accuracy: " << avgCorrect7 * 100 << "%") << std::endl;
        std::cout << BLUETEXT("accuracy: " << avgCorrect8 * 100 << "%") << std::endl;
        std::cout << BLUETEXT("accuracy: " << avgCorrect9 * 100 << "%") << std::endl;
        std::cout << BLUETEXT("accuracy: " << avgCorrect10 * 100 << "%") << std::endl;
        std::cout << "Median Rot. Error: " << medianRotErr << "deg, Median T. Error: " << medianTErr * 100 << "cm." << std::endl;

        testFile
                << avgCorrect * 100 << "%"<< ","            // 0 - percentage of correct poses
                << avgCorrect6 * 100 << "%"<< ","            // 0 - percentage of correct poses
                << avgCorrect7 * 100 << "%"<< ","            // 0 - percentage of correct poses
                << avgCorrect8 * 100 << "%"<< ","            // 0 - percentage of correct poses
                << avgCorrect9 * 100 << "%"<< ","            // 0 - percentage of correct poses
                << avgCorrect10 * 100 << "%"<< ","            // 0 - percentage of correct poses
                << lossMean[0] << ","           // 1 - mean loss of selected hypotheses
                << lossStdDev[0] << ","         // 2 - standard deviation of losses of selected hypotheses
                << entropyMean[0] << ","        // 3 - mean of the score distribution entropy
                << entropyStdDev[0] << ","      // 4 - standard deviation of the score distribution entropy
                << medianRotErr << ","          // 5 - median rotational error of selected hypotheses
                << medianTErr                   // 6 - median translational error (in m) of selected hypotheses
                << std::endl;
        testErrFile.close();
        //testFile.close();  
        lua_close(stateRGB);
        lua_close(stateObj);
    }
    testFile.close();
    return 0;    
}
