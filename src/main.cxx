
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <regex>
#include <set>
#include <string>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

#include "openvo/Tracker.hpp"

namespace fs = boost::filesystem;
namespace po = boost::program_options;

constexpr int ESC_KEY = 27;

/**
 * Represents the passed command-line arguments.
 */
struct ProgramOptions
{
	ProgramOptions(int argc, char ** argv)
		: mBaseDir{}
		, mLeftFramesDir{}
		, mRightFramesDir{}
		, mCalibrationFile{}
		, mStart{}
		, mEnd{}
		, mStep{}
	{
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h", "produce help message")

			("base,b",
			 po::value<fs::path>(&mBaseDir),
			 "acts as the base directory for other parameters")

			("left,l",
			 po::value<fs::path>(&mLeftFramesDir)->default_value("left")->implicit_value("image_0"),
			 "folder containing trip images")

			("right,r",
			 po::value<fs::path>(&mRightFramesDir)->default_value("right")->implicit_value("image_1"),
			 "folder containing depth images per frame")

			("calibration,c",
			 po::value<fs::path>(&mCalibrationFile)->default_value("calib.txt"),
			 "calibration file")

			("start",
			 po::value<int>(&mStart)->default_value(0),
			 "start at frame")
			("step",
			 po::value<int>(&mStep)->default_value(1)->implicit_value(2),
			 "frame skipping, step size over frames")
			("end",
			 po::value<int>(&mEnd)->default_value(std::numeric_limits<int>::max()), // NOTE: just keep it in mind
			 "end at frame")

			 ("verbose",
			 "enable console output");

		po::positional_options_description p;
		p.add("base", -1);

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);

		if (vm.count("help")) {
			std::cout << desc << std::endl;
			std::exit(EXIT_SUCCESS);
		}

		// hook: any action that should be taken,
		// like ("compression", value<int>()->default_value(10), "compression level")
		po::notify(vm);

		if (not mBaseDir.empty()) {
			mLeftFramesDir = this->extend(mBaseDir, mLeftFramesDir);
			mRightFramesDir = this->extend(mBaseDir, mRightFramesDir);
			mCalibrationFile = this->extend(mBaseDir, mCalibrationFile);
		}

		if (not fs::is_directory(mLeftFramesDir)) {
			throw std::invalid_argument("dir given in --left does not exist");
		}
		if (not fs::is_directory(mRightFramesDir)) {
			throw std::invalid_argument("dir given in --right does not exist");
		}
		if (not fs::is_regular_file(mCalibrationFile)) {
			throw std::invalid_argument("file given in --calibration does not exist");
		}
	}

	/**
	 * @brief Concatenates @a root and @a child path, if @a child is not empty and relative, otherwise returns @a child.
	 *
	 * @param root The root path.
	 * @param child The child path.
	 * @return root / child
	 */
	[[nodiscard]] fs::path extend(fs::path const & root, fs::path const & child) const
	{
		if (child.is_relative() and not child.empty()) {
			return root / child;
		}
		return child;
	}

	fs::path mBaseDir;
	fs::path mLeftFramesDir;
	fs::path mRightFramesDir;
	fs::path mCalibrationFile;

	int mStart;
	int mStep;
	int mEnd;
};

struct Calibration
{
	float_t fx;
	float_t fy;

	float_t cx;
	float_t cy;

	// tx = -fx * B
	// B = tx / (-fx)
	float_t tx;
};

/**
 * @brief Checks if given path @a p is hidden.
 *
 * @param p File path.
 * @return True if path is hidden, otherwise false.
 */
bool isHidden(fs::path const & p)
{
	auto const name = p.filename();
	return name != ".." && name != "." && name.string()[0] == '.';
}

/**
 * @brief Checks if given path @a p is a regular file.
 *
 * @param p File path.
 * @return True if @p p is a regular file, otherwise false.
 */
bool isRegular(fs::path const & p)
{
	return fs::is_regular_file(p);
}

/**
 * @brief Collects regular and not hidden files of @a dir.
 *
 * @param dir Directory.
 * @return A collection of file paths.
 */
std::vector<std::string> collectFiles(fs::path const & dir) {
	std::vector<std::string> collection;
	for (auto const & f : boost::make_iterator_range(fs::directory_iterator(dir), {})) {
		if (isRegular(f) and not isHidden(f)) {
			collection.emplace_back(f.path().string());
		}
	}
	return collection;
}

/**
 * Loads a whole file in a string.
 *
 * @param file File path.
 * @return The contant of the file.
 */
std::string loadFileContent(fs::path const & file)
{
	std::ifstream t{file.string()};
	std::string str;

	t.seekg(0, std::ios::end);
	auto const pos = t.tellg();
	if (0 > pos) {
		throw std::runtime_error{"Stream seek fails at the content of '" + file.string() + "'"};
	}
	str.reserve(static_cast<std::size_t>(pos));
	t.seekg(0, std::ios::beg);

	str.assign(std::istreambuf_iterator<char>(t), std::istreambuf_iterator<char>());
	boost::trim(str);

	return str;
}
/**
 * @brief Reads all numeric values from a file and fills a vector with them.
 *
 * @tparam T Arithmetic type.
 * @param file File path.
 * @return A vector of numeric values.
 */
template <typename T>
std::vector<T> loadNumerics(fs::path const & file)
{
	static_assert(std::is_arithmetic<T>::value, "Not an arithmetic type.");

	std::vector<T> numerics;
	std::string content = loadFileContent(file);
	std::regex const regex{R"(\s|\n|\r|\t)"};
	std::sregex_token_iterator it{std::begin(content), std::end(content), regex, -1};
	std::vector<std::string> words{it, {}};
	for (auto && w : words) {
		try {
			T const val = boost::lexical_cast<T>(w);
			numerics.emplace_back(val);
		}
		catch (boost::bad_lexical_cast const &) {
			std::cout << "[WARN] Caught bad lexical cast on '" << w << "'." << std::endl;
		}
	}
	return numerics;
}

/**
 * Loads (right camera) calibration.
 *
 * @param file File path.
 * @return Calibration.
 */
Calibration loadCalibration(fs::path const & file)
{
	auto && numerics = loadNumerics<float_t>(file);

	std::size_t const base{12L}; // skip left camera calibration
	float_t const fx = numerics.at(base + 0);
	float_t const cx = numerics.at(base + 2);
	float_t const tx = numerics.at(base + 3);
	float_t const fy = numerics.at(base + 5);
	float_t const cy = numerics.at(base + 6);

	return {fx, fy, cx, cy, tx};
}

int main(int argc, char ** argv)
{
	std::unique_ptr<ProgramOptions> opts;
	try {
		opts = std::make_unique<ProgramOptions>(argc, argv);
	} catch (po::invalid_option_value const & e) {
		std::cerr << "[ERROR]: " << e.what() << std::endl;
		return EXIT_FAILURE;
	} catch (po::invalid_command_line_syntax const & e) {
		std::cerr << "[ERROR]: " << e.what() << std::endl;
		return EXIT_FAILURE;
	} catch (po::unknown_option const & e) {
		std::cerr << "[ERROR]: " << e.what() << std::endl;
		return EXIT_FAILURE;
	} catch (std::invalid_argument const & e) {
		std::cerr << "[ERROR]: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	//
	// preparation
	//

	// collect left & right frame paths
	std::vector<std::string> leftFramePaths = collectFiles(opts->mLeftFramesDir);
	std::vector<std::string> rightFramePaths = collectFiles(opts->mRightFramesDir);

	// sort both path collections
	std::sort(std::begin(leftFramePaths), std::end(leftFramePaths));
	std::sort(std::begin(rightFramePaths), std::end(rightFramePaths));

	// TODO: left & right consistency check (size & name)

	// makes sure the loop ends
	opts->mEnd = static_cast<int>(std::size(leftFramePaths));

	// load the calibration of right camera
	Calibration calib = loadCalibration(opts->mCalibrationFile);

	//
	// create stereo matcher
	//

	// 64 seems to be a good sweet spot -> depth range ~ (6m - 6000m], we will skip infinite depths
	cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(64, 11);
	stereo->setPreFilterCap(49);
	stereo->setPreFilterSize(65);
	stereo->setSpeckleRange(51);
	stereo->setSpeckleWindowSize(160);
	stereo->setTextureThreshold(160);
	stereo->setUniquenessRatio(32);

	//
	// parameterization
	//

	// Shi-Tomasi parameter
	int const maxCorners = 1024;
	double const qualityLevel = 0.01;
	double const minDistance = 10;
	int const blockSize = 3;
	bool const useHarrisDetector = false;
	double const k = 0.04;

	cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

	//
	// Loop
	//

	// iterate over frames and feed them into OpenVO
	for (int id = opts->mStart; id < opts->mEnd; id += opts->mStep) {

		auto const & leftPath = leftFramePaths[id];
		auto const & rightPath = rightFramePaths[id];

		// load left image
		cv::Mat leftMat = cv::imread(leftPath, cv::IMREAD_GRAYSCALE);
		if (leftMat.empty()) { // is input invalid
			std::cerr <<  "Could not open or find the left image: '" << leftPath << "'" << std::endl;
			return EXIT_FAILURE;
		}

		// load right image
		cv::Mat rightMat = cv::imread(rightPath, cv::IMREAD_GRAYSCALE);
		if (rightMat.empty()) { // is input invalid
			std::cerr <<  "Could not open or find the right image: '" << rightMat << "'" << std::endl;
			return EXIT_FAILURE;
		}

		//
		// disparity & depth map calculation
		//

		cv::Mat disparity16;
		// StereoBM compute 16-bit fixed-point disparity map (where each disparity value has 4 fractional bits).
		stereo->compute(leftMat, rightMat, disparity16);

		cv::Mat disparity32;
		disparity16.convertTo(disparity32, CV_32F, 1.f / 16.f);

		cv::Mat_<float_t> depthMap(disparity32.size());
		depthMap = -calib.tx / disparity32;

		//
		// corner detection
		//

		std::vector<cv::Point2f> corners;
		// detect corners
		cv::goodFeaturesToTrack(leftMat, corners, maxCorners, qualityLevel, minDistance, cv::noArray(), blockSize, useHarrisDetector, k);

		//
		// visualization
		//

		// apply a colormap to the computed disparity for visualization only!
		cv::Mat disparityColored, disparityScaled;
		// disparity range: [min valid/used value - max value] -> (0 - 64]
		// 16-bit fixed-point disparity map with 4 fractional bits -> shift 4 bits or * 16
		// alpha = 255 / (max - min) -> 0.249 = 255 / (64 * 16 - (1/16) * 16)
		// fix alpha keeps the colors of depths stable, calculation of min - max changes alpha -> "color flickers"
		disparity16.convertTo(disparityScaled, CV_8U, 0.249);
		cv::applyColorMap(disparityScaled, disparityColored, cv::COLORMAP_JET);

		cv::Mat leftColored; // not really, just 3 channels
		cv::cvtColor(leftMat, leftColored, cv::COLOR_GRAY2BGR);

		// overlay left image and disparity map
		cv::Mat overlay = leftColored * 0.67f + disparityColored * 0.33f;

		// mark invalid disparities (and zeros, skip infinite depths)
		cv::Mat mask = (short{0} >= disparity16);
		// copy the marked pixels from the left image into the result image
		leftColored.copyTo(overlay, mask);

		// draw all strong corners
		for (auto && pt : corners) {
			cv::circle(overlay, pt, 2, cv::Scalar{255, 255, 255}, -1, 8, 0);
			cv::circle(overlay, pt, 1, cv::Scalar{0, 0, 0}, -1, 8, 0);
		}

		cv::imshow("result", overlay);
		auto const key = cv::waitKey(1);
		if (ESC_KEY == key) { // exit when the Escape key is pressed
			id = std::numeric_limits<int>::max() - 1;
		}

	}

	cv::destroyAllWindows();

	return EXIT_SUCCESS;
}
