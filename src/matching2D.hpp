#ifndef matching2D_hpp
#define matching2D_hpp

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"

class Detector {
public:

    /**
     * Constructor for Detector, defaults to Shi-Tomasi detector if input isn't valid.
     * @param detector_type string [in]
     */
    Detector (const std::string & detector_type);
    virtual ~Detector() {};

    /**
     * Turn on or off keypoint visualization for debugging.
     * @param bool [in] Should we visualize the keypoints.
     */
    void visualizeDetector(bool visualize) {
        m_visualize = visualize;
    };

    /**
     * Detect the corners with the chosen detector.
     * @param vector of keypoints [out] vector of detected keypoints returned by reference.
     * @param Mat [in] Input image to analyze as a CV::Mat. If this image is not grayscale, 
     *                 it will be converted in place, modifying the input.
     * @return double time to run in ms.
     */
    double detect (std::vector<cv::KeyPoint> &keypoints, cv::Mat &img) const;

private:
    Detector(){};

    enum DetectorType {
        SHITOMASI,
        HARRIS,
        FAST, 
        BRISK, 
        ORB, 
        AKAZE, 
        SIFT
    };

    /** 
     * Various detection algorithms
     * @param vector of keypoints [out] vector of detected keypoints returned by reference.
     * @param Mat [in] Input image to analyze as a CV::Mat. If this image is not grayscale, 
     *                 it will be converted in place, modifying the input.
     */
    void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool is_harris=false) const;
    void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img) const;
    void detKeypointsFAST(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img) const;
    void detKeypointsBRISK(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img) const;
    void detKeypointsORB(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img) const;
    void detKeypointsAKAZE(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img) const;
    void detKeypointsSIFT(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img) const;

    /**
     * Function to get the name of the detector as a string.
     */
    std::string getName() const;

    //Type of detector to use
    DetectorType m_type;

    //Should the detections be visualized
    bool m_visualize;
};

class Descriptor {
public:
    /**
     * Constructor for the descriptor class. Defaults to BRISK if an invalid name is supplied.
     * @param string name of the descriptor.
     */
    Descriptor (const std::string & descriptor_name);
    virtual ~Descriptor() {};

    /**
     * Describe the keypoints with the given descriptor.
     * @param vector of keypoints [in] vector of detected keypoints.
     * @param Mat [in] Input image to analyze as a CV::Mat. If this image is not grayscale, 
     *                 it will be converted in place, modifying the input.
     * @param Mat [out] descriptors for each keypoint returned by reference.
     * @param double time in ms to run the computation.
     */
    double describe (std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat & descriptors) const;

    enum FeatureType {
        BINARY,
        HOG
    };

    FeatureType featureType() {
        if (m_type == DescriptorType::SIFT) {
            return FeatureType::HOG;
        } 
        return FeatureType::BINARY;
    };
private:
    Descriptor() {};

    enum DescriptorType {
        BRISK,
        BRIEF, 
        ORB, 
        FREAK, 
        AKAZE,
        SIFT
    };

    /**
     * Return the descriptor name.
     * @return string name of the descriptor
     */
    std::string getName() const;

    DescriptorType m_type;
};

class Matcher {
public:
    /**
     * Set up a Matcher class. 
     * @param string [in] matcher type string (BRUTE_FORCE of FLANN)
     * @param string [in] selector selector type.
     * @param feature_type [in] Descriptor feature type (taken from the descriptor class)
     */
    Matcher (const std::string & matcher, const std::string & selector, Descriptor::FeatureType feature_type);
    virtual ~Matcher(){};

    /**
     * Method that actually runs the matcher and the selector.
     * @param keypoints_1 [in] Vector of input keypoints from image 1
     * @param keypoints_2 [in] Vector of input keypoints from image 2
     * @param descriptors_1 [in] matrix of descriptors for image 1. If using FLANN, binary descriptors will be modified in place into floats.
     * @param descriptors_2 [in] matrix of descriptors for image 2. If using FLANN, binary descriptors will be modified in place into floats.
     * @param matches [out] output vector of matches.
     */
    void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, 
                          std::vector<cv::KeyPoint> &kPtsRef, 
                          cv::Mat &descSource, 
                          cv::Mat &descRef,
                          std::vector<cv::DMatch> &matches);

private:
    Matcher() {};

    enum MatcherType {
        BRUTE_FORCE,
        FLANN
    };

    enum SelectorType {
        NEAREST_NEIGHBOUR,
        K_NEAREST_NEIGHBOUR
    };

    MatcherType m_match_type;
    SelectorType m_selector_type;
    Descriptor::FeatureType m_feature_type;

    static const int c_k_NN;
    static const float c_dist_ratio;
};


#endif /* matching2D_hpp */
