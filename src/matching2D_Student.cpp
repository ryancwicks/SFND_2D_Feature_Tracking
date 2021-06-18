#include <numeric>
#include "matching2D.hpp"
#include <algorithm>

using namespace std;

Detector::Detector (const std::string & detector_type) :
    m_visualize(false) {
    std::string detector_upper (detector_type);
    std::transform(detector_upper.begin(), detector_upper.end(),detector_upper.begin(), ::toupper);
    if (detector_upper == "SHITOMASI" || detector_upper == "SHI-TOMASI") {
        m_type = SHITOMASI;
    } else if (detector_upper == "HARRIS") {
        m_type = HARRIS;
    } else if (detector_upper == "FAST") {
        m_type = FAST;
    } else if (detector_upper == "BRISK") {
        m_type = BRISK;
    } else if (detector_upper == "ORB") {
        m_type = ORB;
    } else if (detector_upper == "AKAZE") {
        m_type = AKAZE;
    } else if (detector_upper == "SIFT") {
        m_type = SIFT;
    } else {
        std::cerr << "Invalid detector selected: " << detector_type << std::endl << "SIFT will be used instead." << std::endl;
        m_type = SIFT;
    }
}

double Detector::detect(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img) const {
    
    keypoints.clear();
    if (img.channels() != 1) { //double check we are actually getting a grayscale image.
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    }
    double t = (double)cv::getTickCount();

    switch (m_type) {
        case SHITOMASI:
            detKeypointsShiTomasi(keypoints, img);
            break;
        case HARRIS:
            detKeypointsHarris(keypoints, img);
            break;
        case FAST:
            detKeypointsFAST(keypoints, img);
            break; 
        case BRISK:
            detKeypointsBRISK(keypoints, img);
            break; 
        case ORB:
            detKeypointsORB(keypoints, img);
            break;
        case AKAZE:
            detKeypointsAKAZE(keypoints, img);
            break; 
        case SIFT:
            detKeypointsSIFT(keypoints, img);
            break;
        default:
            std::cerr << "Invalid detector, this shouldn't be possible.";
            break;
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    std::string detector_name = getName();
    //cout << detector_name << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (m_visualize)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detector_name + " Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

    return t;
}

std::string Detector::getName() const {
    switch (m_type) {
        case SHITOMASI:
            return "Shi-Thomas";
        case HARRIS:
            return "Harris";
        case FAST:
            return "FAST";
        case BRISK:
            return "BRISK";
        case ORB:
            return "ORB";
        case AKAZE:
            return "AKAZE";
        case SIFT:
            return "SIFT";
        default:
            std::cerr << "Invalid detector, this shouldn't be possible.";
            return "";
    }
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void Detector::detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool is_harris) const
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, is_harris, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    
}

void Detector::detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img) const {
    detKeypointsShiTomasi(keypoints, img, true);
}

void Detector::detKeypointsFAST(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img) const {
    const int threshold=10; 
    const bool nonmaxSuppression=true; 
    const auto type=cv::FastFeatureDetector::TYPE_9_16;
    auto detector = cv::FastFeatureDetector::create(threshold, nonmaxSuppression, type);
    detector->detect(img, keypoints);
}

void Detector::detKeypointsBRISK(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img) const {
    const int thresh=30; 
    const int octaves=3; 
    const float patternScale=1.0f;
    auto detector = cv::BRISK::create(thresh, octaves, patternScale);
    detector->detect(img, keypoints);
}

void Detector::detKeypointsORB(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img) const {
    const int nfeatures=500; 
    const float scaleFactor=1.2f; 
    const int nlevels=8; 
    const int edgeThreshold=31; 
    const int firstLevel=0;
    const int WTA_K=2; 
    const auto scoreType=cv::ORB::HARRIS_SCORE; 
    const int patchSize=31; 
    const int fastThreshold=20;
    auto detector = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
    detector->detect (img, keypoints);
}

void Detector::detKeypointsAKAZE(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img) const {
    const auto descriptor_type=cv::AKAZE::DESCRIPTOR_MLDB;
    const int descriptor_size=0;
    const int descriptor_channels=3;
    const float threshold=0.001f;
    const int nOctaves=4;
    const int nOctaveLayers=4;
    const auto diffusivity=cv::KAZE::DIFF_PM_G2;
    auto detector = cv::AKAZE::create (descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves, nOctaveLayers, diffusivity);
    detector->detect(img, keypoints);
}

void Detector::detKeypointsSIFT(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img) const {
    const int nfeatures=0; 
    const int nOctaveLayers=3; 
    const double contrastThreshold=0.04; 
    const double edgeThreshold=10; 
    const double sigma=1.6;
    auto detector = cv::SIFT::create (nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    detector->detect(img, keypoints);
}

Descriptor::Descriptor (const std::string & descriptor_name) {
    std::string descriptor_upper (descriptor_name);
    std::transform(descriptor_upper.begin(), descriptor_upper.end(),descriptor_upper.begin(), ::toupper);
    if (descriptor_upper == "BRISK") {
        m_type = BRISK;
    } else if (descriptor_upper == "BRIEF") {
        m_type = BRIEF;
    } else if (descriptor_upper == "ORB") {
        m_type = ORB;
    } else if (descriptor_upper == "FREAK") {
        m_type = FREAK;
    } else if (descriptor_upper == "AKAZE") {
        m_type = AKAZE;
    } else if (descriptor_upper == "SIFT") {
        m_type = SIFT;
    } else {
        std::cerr << "Invalid descriptor selected: " << descriptor_name << std::endl << "SIFT will be used instead." << std::endl;
        m_type = BRISK;
    }

}

double Descriptor::describe(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat & descriptors) const {

    cv::Ptr<cv::DescriptorExtractor> extractor (nullptr);
    
    double t = (double)cv::getTickCount();
    switch (m_type) {
        case BRISK : {
            int threshold = 30;        // FAST/AGAST detection threshold score.
            int octaves = 3;           // detection octaves (use 0 to do single scale)
            float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
            extractor = cv::BRISK::create(threshold, octaves, patternScale);
            break;
        }
        case BRIEF: {
            const int bytes=32;
            const bool use_orientation=false;
            extractor = cv::xfeatures2d::BriefDescriptorExtractor::create (bytes, use_orientation);
            break;
        }
        case ORB: {
            const int nfeatures=500; 
            const float scaleFactor=1.2f; 
            const int nlevels=8; 
            const int edgeThreshold=31; 
            const int firstLevel=0;
            const int WTA_K=2; 
            const auto scoreType=cv::ORB::HARRIS_SCORE; 
            const int patchSize=31; 
            const int fastThreshold=20;
            extractor = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
            break;
        }
        case FREAK: {
            const bool orientationNormalized=true;
            const bool scaleNormalized=true;
            const float patternScale=22.0f;
            const int nOctaves=4;
            extractor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale, nOctaves);
            break;
        }
        case AKAZE: {
            const auto descriptor_type=cv::AKAZE::DESCRIPTOR_MLDB;
            const int descriptor_size=0;
            const int descriptor_channels=3;
            const float threshold=0.001f;
            const int nOctaves=4;
            const int nOctaveLayers=4;
            const auto diffusivity=cv::KAZE::DIFF_PM_G2;
            extractor = cv::AKAZE::create (descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves, nOctaveLayers, diffusivity);
            break;
        }
        case SIFT: {
            const int nfeatures=0; 
            const int nOctaveLayers=3; 
            const double contrastThreshold=0.04; 
            const double edgeThreshold=10; 
            const double sigma=1.6;
            extractor = cv::SIFT::create ();//nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
            break;
        }
        default: {
            std::cerr << "Invalid detector, this shouldn't be possible.";
            break;
        }
    }

    if (extractor != nullptr) {
        extractor->compute(img, keypoints, descriptors);
    } else {
        std::cerr << "Failed to initialize the descriptor " << getName() << std::endl;
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    //cout << getName() << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;

    return t;
}

std::string Descriptor::getName() const {
    switch (m_type) {
        case BRISK :
            return "BRISK";
        case BRIEF:
            return "BRIEF";
        case ORB:
            return "ORB";
        case FREAK:
            return "FREAK";
        case AKAZE:
            return "AKAZE";
        case SIFT:
            return "SIFT";
        default:
            std::cerr << "Invalid detector, this shouldn't be possible.";
            return "";
    }
}


const int Matcher::c_k_NN = 2;
const float Matcher::c_dist_ratio = 0.8f;

Matcher::Matcher (const std::string & matcher, const std::string & selector, Descriptor::FeatureType feature_type) :
    m_feature_type(feature_type) {

    std::string matcher_upper (matcher);
    std::transform(matcher_upper.begin(), matcher_upper.end(), matcher_upper.begin(), ::toupper);
    std::string selector_upper (selector);
    std::transform(selector_upper.begin(), selector_upper.end(), selector_upper.begin(), ::toupper);
    
    if (matcher_upper == "BRUTE_FORCE" || matcher_upper == "BRUTEFORCE" || matcher_upper == "BRUTE FORCE" ||
        matcher_upper == "MAT_BF") {
        m_match_type = BRUTE_FORCE;
    } else {
        m_match_type = FLANN;
    }

    if (selector_upper == "NEAREST_NEIGHBOUR" || selector_upper == "NEARESTNEIGHBOUR" || selector_upper == "NEAREST NEIGHBOUR" ||
        selector_upper == "NEAREST_NEIGHBOR" || selector_upper == "NEARESTNEIGHBOR" || selector_upper == "NEAREST NEIGHBOR" ||
        selector_upper == "SEL_NN") {
        m_selector_type = NEAREST_NEIGHBOUR;
    } else {
        m_selector_type = K_NEAREST_NEIGHBOUR;
    }
}


// Find best matches for keypoints in two camera images based on several matching methods
void Matcher::matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, 
                               std::vector<cv::KeyPoint> &kPtsRef, 
                               cv::Mat &descSource, 
                               cv::Mat &descRef,
                               std::vector<cv::DMatch> &matches) {
    matches.clear();
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    int normType;
    if (m_feature_type == Descriptor::FeatureType::BINARY) {
        normType = cv::NORM_HAMMING;
    } else {
        normType = cv::NORM_L2;
    }

    if (m_match_type == BRUTE_FORCE) {
        matcher = cv::BFMatcher::create(normType, crossCheck);
    } else {
        matcher = cv::FlannBasedMatcher::create();
        if (descSource.type() != CV_32F) {
            descSource.convertTo(descSource, CV_32F); 
        }
        if (descRef.type() != CV_32F) {
            descRef.convertTo(descRef, CV_32F); 
        }
    }
    
    // perform matching task
    if (m_selector_type == NEAREST_NEIGHBOUR) { 
        // nearest neighbor (best match)
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    } else { 
        // k nearest neighbors (k=2)
        std::vector<std::vector<cv::DMatch>> k_matches;
        matcher->knnMatch(descSource, descRef, k_matches, c_k_NN );
        //Perform the descriptor distance ratio test on the 2 nearest neighbours
        for ( const auto & k_match : k_matches ) {
            if (k_match[0].distance/k_match[1].distance <= c_dist_ratio) {
                matches.push_back (k_match[0]);
            }
        }
    }
}
