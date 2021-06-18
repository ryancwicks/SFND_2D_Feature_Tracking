/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <deque>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

bool calculateKeypoints ( std::deque<DataFrame>& data_buffer,
                      const std::string & detector_type,
                      const std::string & descriptor_type,
                      const std::string & matcher_type, 
                      const std::string & selector_type,
                      bool visualize) {
    /* DETECT IMAGE KEYPOINTS */
    // extract 2D keypoints from current image
    vector<cv::KeyPoint> keypoints; // create empty feature list for current image
    Detector detector (detector_type);
    detector.visualizeDetector(false);
    try {
        (data_buffer.end()-1)->keypoint_run_time = detector.detect(keypoints, (data_buffer.end() - 1)->cameraImg);
    } catch(...) {
        std::cerr <<"Keypoint detection with " << detector_type << " failed." << std::endl;
        (data_buffer.end()-1)->keypoint_run_time = -1.0;
        return false;
    }
    
    // only keep keypoints on the preceding vehicle
    bool bFocusOnVehicle = true;
    cv::Rect vehicleRect(535, 180, 180, 150);
    if (bFocusOnVehicle)
    {
        for (auto it = keypoints.begin(); it != keypoints.end(); ) {
            if (vehicleRect.contains(it->pt)) {
                ++it;
            } else {
                keypoints.erase (it);
            }
        }
    }
    
    // optional : limit number of keypoints (helpful for debugging and learning)
    bool bLimitKpts = false;
    if (bLimitKpts)
    {
        int maxKeypoints = 50;
        if (detector_type.compare("SHITOMASI") == 0)
        { // there is no response info, so keep the first 50 as they are sorted in descending quality order
            keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
        }
        cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
        //cout << " NOTE: Keypoints have been limited!" << endl;
    }
    
    // push keypoints and descriptor for current frame to end of data buffer
    (data_buffer.end() - 1)->keypoints = keypoints;
    //cout << "#2 : DETECT KEYPOINTS done" << endl;
    /* EXTRACT KEYPOINT DESCRIPTORS */
    cv::Mat descriptors;
    Descriptor descriptor (descriptor_type);
    try{
        (data_buffer.end()-1)->descriptor_run_time = descriptor.describe((data_buffer.end() - 1)->keypoints, (data_buffer.end() - 1)->cameraImg, descriptors);
    } catch (...) {
        std::cerr << "Descriptor step failed for the combination of " << detector_type << " keypoints and " << descriptor_type << " descriptors." << std::endl;
        (data_buffer.end()-1)->descriptor_run_time = -1.0;
        return false;
    }
    // push descriptors for current frame to end of data buffer
    (data_buffer.end() - 1)->descriptors = descriptors;
    //cout << "#3 : EXTRACT DESCRIPTORS done" << endl;
    if (data_buffer.size() > 1) // wait until at least two images have been processed
    {
        /* MATCH KEYPOINT DESCRIPTORS */
        vector<cv::DMatch> matches;
        Matcher matcher (matcher_type, selector_type, descriptor.featureType());
        try {
            matcher.matchDescriptors((data_buffer.end() - 2)->keypoints, (data_buffer.end() - 1)->keypoints,
                                 (data_buffer.end() - 2)->descriptors, (data_buffer.end() - 1)->descriptors,
                                 matches);
        } catch (...) {
            std::cerr << matcher_type << " matching failed with " << selector_type << " selection and " << descriptor_type << " descriptors." << std::endl;
            return false;
        }

        // store matches in current data frame
        (data_buffer.end() - 1)->kptMatches = matches;
        //cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;
        // visualize matches between current and previous image
        
        if (visualize)
        {
            cv::Mat matchImg = ((data_buffer.end() - 1)->cameraImg).clone();
            cv::drawMatches((data_buffer.end() - 2)->cameraImg, (data_buffer.end() - 2)->keypoints,
                            (data_buffer.end() - 1)->cameraImg, (data_buffer.end() - 1)->keypoints,
                            matches, matchImg,
                            cv::Scalar::all(-1), cv::Scalar::all(-1),
                            vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            string windowName = "Matching keypoints between two camera images";
            cv::namedWindow(windowName, 7);
            cv::imshow(windowName, matchImg);
            cout << "Press key to continue to next image" << endl;
            cv::waitKey(0); // wait for key to be pressed
        }
    }

    return true;
}

struct Summary {
    std::string keypoint_type;
    std::string descriptor_type;
    std::vector <std::vector<cv::KeyPoint>> keypoints;
    std::vector <double> keypoint_times;
    std::vector <double> descriptor_times;

    friend std::ostream& operator<<(std::ostream& os, const Summary& summary);
};

std::ostream& operator<<(std::ostream& os, const Summary& summary) {
    //keypoint type, detector_type, mean # keypoints, mean keypoint time (s), mean descriptor time (s)
    double mean_keypoint_time = 0.0;
    double mean_descriptor_time = 0.0;
    int keypoint_average = 0;
    if (summary.keypoint_times.size() != 0) {
        for (auto t : summary.keypoint_times) {
            mean_keypoint_time += t;
        }
        mean_keypoint_time /= summary.keypoint_times.size();
    }
    if (summary.descriptor_times.size() != 0){
        for (auto t : summary.descriptor_times) {
            mean_descriptor_time += t;
        }
        mean_descriptor_time /= summary.descriptor_times.size();
    }
    if (summary.keypoints.size() != 0) {
        for (const auto & k : summary.keypoints) {
            keypoint_average += k.size();
        }
        keypoint_average /= summary.keypoints.size();
    }
    os << summary.keypoint_type << ", " << summary.descriptor_type << ", " << keypoint_average << ", "  
       << mean_keypoint_time << ", " << mean_descriptor_time << std::endl;
    return os;
}



/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    const string dataPath = "../";

    // camera
    const string imgBasePath = dataPath + "images/";
    const string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    const string imgFileType = ".png";
    const int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    const int imgEndIndex = 9;   // last file index to load
    const int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    const int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    deque<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    const std::vector<std::string> keypoint_types {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"}; 
    const std::vector<std::string> descriptor_types {"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"};

    //string detectorType = "SIFT"; //SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
    //string descriptorType = "SIFT"; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
    //SIFT features and ORB detector runs out of memory.
    //SIFT and AKAZE also fails. (AKAZE doesn't handle mixing keypoints and descriptors well.)
    string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
    string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN

    std::vector <Summary> summaries;

    for (auto detectorType : keypoint_types) {
        for (auto descriptorType : descriptor_types) {
            /* MAIN LOOP OVER ALL IMAGES */
            Summary summary;
            summary.keypoint_type = detectorType;
            summary.descriptor_type = descriptorType;
            for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
            {
                /* LOAD IMAGE INTO BUFFER */

                // assemble filenames for current index
                ostringstream imgNumber;
                imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
                string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

                // load image from file and convert to grayscale
                cv::Mat img, imgGray;
                img = cv::imread(imgFullFilename);
                cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

                // push image into data frame buffer
                DataFrame frame;
                frame.cameraImg = imgGray;
                dataBuffer.push_back(frame);
                if (dataBuffer.size() > dataBufferSize) {
                    //This would be a good point to save the data before it is dropped.
                    dataBuffer.pop_front();
                }
                //cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

                if (calculateKeypoints(dataBuffer, detectorType, descriptorType, matcherType, selectorType, bVis)) {

                    summary.keypoints.push_back ((dataBuffer.end()-1)->keypoints);
                    summary.keypoint_times.push_back ((dataBuffer.end()-1)->keypoint_run_time);
                    summary.descriptor_times.push_back ((dataBuffer.end()-1)->descriptor_run_time);
                }

            } // eof loop over all images

            summaries.push_back (summary);
        }
    }

    std::cout << "keypoint type, detector_type, mean # keypoints, mean keypoint time (s), mean descriptor time (s)" << std::endl;
    for (const auto &summary : summaries) {
        std::cout << summary;
    }
    return 0;
}