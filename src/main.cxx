
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

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/video/tracking.hpp>

#include "openvo/Tracker.hpp"

namespace fs = boost::filesystem;
namespace po = boost::program_options;

using Pose = Eigen::Affine3d;
using Trajectory = std::vector<Pose, Eigen::aligned_allocator<Pose>>;

constexpr int ESC_KEY = 27;
constexpr int ESC_SPACE = 32;

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
		, mGroundTruthFile{}
		, mTrajectoryFile{}
		, mHeadless{}
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

			("gt",
			 po::value<fs::path>(&mGroundTruthFile)->default_value(""),
			 "ground truth file\n"
			 "([R|t] 3x4-matrix in row major order")

			("trajectory,t",
			 po::value<fs::path>(&mTrajectoryFile)->implicit_value("ovo_result.txt"),
			 "output file for the computed trajectory")

			("headless",
			 po::bool_switch(&mHeadless)->default_value(false),
			 "run without visualization")

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

		if (not std::empty(mBaseDir)) {
			mLeftFramesDir = fs::canonical(mLeftFramesDir, mBaseDir);
			mRightFramesDir = fs::canonical(mRightFramesDir, mBaseDir);
			mCalibrationFile = fs::canonical(mCalibrationFile, mBaseDir);

			if (not std::empty(mGroundTruthFile)) {
				mGroundTruthFile = fs::canonical(mGroundTruthFile, mBaseDir);
			}

			if (not std::empty(mTrajectoryFile)) {
				mTrajectoryFile = this->extend(fs::canonical(mBaseDir), mTrajectoryFile);
			}
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

		if (not std::empty(mGroundTruthFile) and not fs::is_regular_file(mGroundTruthFile)) {
			throw std::invalid_argument("file given in --groundtruth does not exist");
		}

		if (not std::empty(mTrajectoryFile)) {
			auto parent = mTrajectoryFile.parent_path();
			if (not fs::exists(parent)) {
				fs::create_directories(parent);
			}
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
		if (child.is_relative() and not std::empty(child)) {
			return root / child;
		}
		return child;
	}

	fs::path mBaseDir;
	fs::path mLeftFramesDir;
	fs::path mRightFramesDir;
	fs::path mCalibrationFile;
	fs::path mGroundTruthFile;

	fs::path mTrajectoryFile;

	bool mHeadless;

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
 * @return True if @a p is a regular file, otherwise false.
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
 * @return Content of the file.
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

/**
 * Loads a trajectory from a @a file. The file should be in KITTI format.
 *
 * @param file Path to the trajectory file.
 * @return Trajectory, std::vector of Eigen::Affine3d.
 */
Trajectory loadTrajectory(fs::path const & file)
{
	auto && numerics = loadNumerics<float_t>(file);
	std::size_t const size = std::size(numerics);
	assert(0 == (size % 12));
	std::size_t const length = std::size(numerics) / 12L; // number of poses
	Trajectory trajectory;
	trajectory.reserve(length);

	for (std::size_t i = 0L; i < size; i += 12L) {
		// KITTI style: transform consists of 3x4 transformation matrix, stored in rows

		auto const m_00 = boost::lexical_cast<double>(numerics.at(i + 0L));
		auto const m_01 = boost::lexical_cast<double>(numerics.at(i + 1L));
		auto const m_02 = boost::lexical_cast<double>(numerics.at(i + 2L));

		auto const tx = boost::lexical_cast<double>(numerics.at(i + 3L));

		auto const m_10 = boost::lexical_cast<double>(numerics.at(i + 4L));
		auto const m_11 = boost::lexical_cast<double>(numerics.at(i + 5L));
		auto const m_12 = boost::lexical_cast<double>(numerics.at(i + 6L));

		auto const ty = boost::lexical_cast<double>(numerics.at(i + 7L));

		auto const m_20 = boost::lexical_cast<double>(numerics.at(i + 8L));
		auto const m_21 = boost::lexical_cast<double>(numerics.at(i + 9L));
		auto const m_22 = boost::lexical_cast<double>(numerics.at(i + 10L));

		auto const tz = boost::lexical_cast<double>(numerics.at(i + 11L));

		Eigen::Translation3d const translation{tx, ty, tz};
		Eigen::Matrix3d rotation;
		rotation <<
			m_00, m_01, m_02,
			m_10, m_11, m_12,
			m_20, m_21, m_22;

		Pose const pose{translation * rotation};
		trajectory.push_back(pose);
	}

	return trajectory;
}

/**
 * Saves a @a trajectory to a @a file in KITTI format.
 *
 * @param file Path to the trajectory file.
 * @param trajectory Trajectory.
 */
void saveTrajectory(fs::path const & file, Trajectory const & trajectory)
{
	// open stream
	std::ofstream s{file.string()};

	for (auto && pose : trajectory) {
		Eigen::Matrix3d r = pose.rotation();
		Eigen::Vector3d t = pose.translation();

		s <<
			r(0, 0) << ' ' << r(0, 1) << ' ' << r(0, 2) << ' ' << t.x() << ' ' <<
			r(1, 0) << ' ' << r(1, 1) << ' ' << r(1, 2) << ' ' << t.y() << ' ' <<
			r(2, 0) << ' ' << r(2, 1) << ' ' << r(2, 2) << ' ' << t.z() << '\n';
	}

	s.flush();
	s.close();
}

/**
 * Removes all elements from @a v where the value of @a b is 0.
 * Provided for convenience.
 *
 * @tparam T Type of vector @a v.
 * @param b Mask.
 * @param v Vector.
 */
template<typename T>
inline void filter(std::vector<std::uint8_t> const & b, std::vector<T> & v)
{
	assert(std::size(v) == std::size(b));
	auto it = std::begin(b);
	v.erase(std::remove_if(std::begin(v), std::end(v), [&](T) { return !static_cast<bool>(*it++); }), std::end(v));
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
	// load the reference trajectory
	Trajectory groundTruth;
	if (not std::empty(opts->mGroundTruthFile) and not opts->mHeadless) {
		groundTruth = loadTrajectory(opts->mGroundTruthFile);
	}

	//
	// parameterization
	//

	// Shi-Tomasi parameter
	int const maxCorners = 1024;
	double const qualityLevel = 0.01;
	double const minDistance = 10;
	int const cornerBlockSize = 3;
	bool const useHarrisDetector = false;
	double const k = 0.04;

	// sparse iterative optical flow parameter
	cv::Size const windowSize{31, 31};
	int const maxLevel = 3;
	cv::TermCriteria const termCriteria{cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01};
	double const minEigThreshold = 1e-4;

	// stereo block matcher
	int const numDisparities = 64; // seems to be a good sweet spot -> depth range ~ (6m - 6000m], we will skip infinite depths
	int const stereoBlockSize = 11;
	int const preFilterCap = 49;
	int const preFilterSize = 65;
	int const speckleRange = 51;
	int const speckleWindowSize = 160;
	int const textureThreshold = 160;
	int const uniquenessRatio = 32;

	//
	// create stereo matcher
	//

	cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(numDisparities, stereoBlockSize);
	stereo->setPreFilterCap(preFilterCap);
	stereo->setPreFilterSize(preFilterSize);
	stereo->setSpeckleRange(speckleRange);
	stereo->setSpeckleWindowSize(speckleWindowSize);
	stereo->setTextureThreshold(textureThreshold);
	stereo->setUniquenessRatio(uniquenessRatio);

	if (not opts->mHeadless) {
		cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
	}

	cv::Mat prevLeftImage;

	Trajectory trajectory;
	// first frame
	trajectory.push_back(Pose::Identity());

	// meter per pixel
	float trajectoryScale = 1.f;
	bool play = true;

	//
	// Loop
	//

	// iterate over frames and feed them into OpenVO
	for (int frameIdx = opts->mStart; frameIdx < opts->mEnd; frameIdx += opts->mStep) {

		auto const & leftPath = leftFramePaths[frameIdx];
		auto const & rightPath = rightFramePaths[frameIdx];

		// load left image
		cv::Mat leftImage = cv::imread(leftPath, cv::IMREAD_GRAYSCALE);
		if (leftImage.empty()) { // is input invalid
			std::cerr <<  "Could not open or find the left image: '" << leftPath << "'" << std::endl;
			return EXIT_FAILURE;
		}

		// load right image
		cv::Mat rightImage = cv::imread(rightPath, cv::IMREAD_GRAYSCALE);
		if (rightImage.empty()) { // is input invalid
			std::cerr << "Could not open or find the right image: '" << rightImage << "'" << std::endl;
			return EXIT_FAILURE;
		}

		//
		// disparity calculation
		//

		cv::Mat disparity16;
		// StereoBM compute 16-bit fixed-point disparity map (where each disparity value has 4 fractional bits).
		stereo->compute(leftImage, rightImage, disparity16);

		//
		// disparity conversion (fixed-point to floating-point)
		//

		cv::Mat disparity32;
		disparity16.convertTo(disparity32, CV_32F, 1.f / 16.f);

		//
		// depths computation
		//

		cv::Mat_<float_t> depthMap(disparity32.size());
		depthMap = -calib.tx / disparity32;

		//
		// corner detection
		//

		// convert error to color
		std::vector<cv::Point2f> detectedCorners;
		// detect corners
		cv::goodFeaturesToTrack(
				leftImage,
				detectedCorners,
				maxCorners,
				qualityLevel,
				minDistance,
				cv::noArray(),
				cornerBlockSize,
				useHarrisDetector,
				k);

		//
		// depth collection
		//

		std::vector<float_t> depths;
		depths.reserve(std::size(detectedCorners));

		for (std::size_t i = 0L; i < std::size(detectedCorners); ++i) {
			auto && pt = detectedCorners[i];

			auto const col = static_cast<int>(std::round(pt.x));
			auto const row = static_cast<int>(std::round(pt.y));

			auto depth = depthMap.at<float_t>(row, col);
			bool valid = 0.f < depth && std::isfinite(depth);

			if (valid) { // only collect valid depths
				depths.emplace_back(depth);
			} else { // remove corners without valid depth
				detectedCorners[i] = detectedCorners.back();
				detectedCorners.pop_back();
				--i;
			}
		}

		//
		// "memories"
		//

		std::vector<cv::Point2f> trackedCorners;
		if (!prevLeftImage.empty()) {

			std::vector<uchar> status;
			std::vector<float> errors;

			// track from current to previous image
			cv::calcOpticalFlowPyrLK(
					leftImage,
					prevLeftImage,
					detectedCorners,
					trackedCorners,
					status,
					errors,
					windowSize,
					maxLevel,
					termCriteria,
					0,
					minEigThreshold);

			float const ifx = 1.f / calib.fx;
			float const ify = 1.f / calib.fy;
			float const icx = -calib.cx / calib.fx;
			float const icy = -calib.cy / calib.fy;

			std::vector<cv::Point3f> landmarks;
			std::vector<cv::Point2f> observations;

			for (std::size_t i = 0L; i < std::size(detectedCorners); ++i) {
				// this corner could be tracked back
				if (status[i]) {
					float const z = depths[i]; // only collected valid depths
					cv::Point2f const & curr = detectedCorners[i];
					cv::Point2f const & prev = trackedCorners[i];

					float const cu = curr.x * ifx + icx;
					float const cv = curr.y * ify + icy;
					landmarks.emplace_back(cu * z, cv * z, z);

					float const pu = prev.x * ifx + icx;
					float const pv = prev.y * ify + icy;
					observations.emplace_back(pu, pv);
				}
			}

			cv::Matx31d rvec{};
			cv::Matx31d tvec{};
			std::vector<int> inliers;

			cv::solvePnPRansac(
					landmarks,
					observations,
					cv::Matx33f::eye(),
					cv::noArray(),
					rvec,
					tvec,
					false,
					std::size(observations),
					0.707f * ifx, // points are also "normalized"
					0.99,
					inliers,
					cv::SOLVEPNP_ITERATIVE);

			// clean up the corners (actually only for visualization)
			if (not opts->mHeadless) {
				// remove not tracked corners
				filter(status, detectedCorners);
				filter(status, trackedCorners);

				std::vector<cv::Point2f> detected;
				std::vector<cv::Point2f> tracked;

				detected.swap(detectedCorners);
				tracked.swap(trackedCorners);

				detectedCorners.reserve(std::size(inliers));
				trackedCorners.reserve(std::size(inliers));

				// keep only inliers
				for (auto const idx : inliers) {
					detectedCorners.emplace_back(detected[idx]);
					trackedCorners.emplace_back(tracked[idx]);
				}
			}

			// convert translation to Eigen
			Eigen::Vector3d t;
			cv::cv2eigen(tvec, t);

			// convert rotation to Eigen
			cv::Matx33d cvR;
			cv::Rodrigues(rvec, cvR);
			Eigen::Matrix3d R;
			cv::cv2eigen(cvR, R);

			// combine translation & rotation
			Eigen::Affine3d const delta{Eigen::Translation3d(t) * R}; // we track back -> already inverse!
			Eigen::Affine3d const pose{trajectory.back() * delta};

			trajectory.push_back(pose);
		}

		//
		// visualization
		//

		if (not opts->mHeadless) {
			// apply a colormap to the computed disparity for visualization only!
			cv::Mat disparityColored, disparityScaled;
			// disparity range: [min valid/used value - max value] -> (0 - 64]
			// 16-bit fixed-point disparity map with 4 fractional bits -> multiplied by 16 ("shift by 4 bits")
			// alpha = 255 / (max - min) -> 0.249 = 255 / (64 * 16 - (1/16) * 16)
			// fix alpha keeps the colors of depths stable, calculation of min - max changes alpha -> "color flickers"
			disparity16.convertTo(disparityScaled, CV_8U, 0.249);
			cv::applyColorMap(disparityScaled, disparityColored, cv::COLORMAP_JET);

			cv::Mat leftColored; // not really, just 3 channels
			cv::cvtColor(leftImage, leftColored, cv::COLOR_GRAY2BGR);

			// overlay left image and disparity map
			cv::Mat overlay = leftColored * 0.67f + disparityColored * 0.33f;

			// mark invalid disparities (and zeros, skip infinite depths)
			cv::Mat const mask = (short{0} >= disparity16);

			// copy the marked pixels from the left image into the result image
			leftColored.copyTo(overlay, mask);

			// draw corners
			if (!std::empty(trackedCorners)) {
				std::size_t const size = std::size(detectedCorners);

				for (std::size_t i = 0L; i < size; ++i) {
					auto const &prev = trackedCorners[i];
					auto const &curr = detectedCorners[i];

					cv::line(
							overlay,
							cv::Point{static_cast<int>(std::round(prev.x)), static_cast<int>(std::round(prev.y))},
							cv::Point{static_cast<int>(std::round(curr.x)), static_cast<int>(std::round(curr.y))},
							cv::Scalar{0, 255, 255},
							1,
							cv::LINE_8,
							0);

					cv::circle(overlay, curr, 2, cv::Scalar{255, 255, 255}, -1, cv::LINE_8, 0);
					cv::circle(overlay, curr, 1, cv::Scalar{0, 0, 0}, -1, cv::LINE_8, 0);
				}
			}

			float const scale = 1.f / trajectoryScale;
			cv::Point2f const pp{calib.cx, calib.cy};

			if (std::size(trajectory) <= std::size(groundTruth)) {
				Eigen::Affine3d const gtOrigin{groundTruth[std::size(trajectory) - 1L].inverse()};
				for (auto const &pose : groundTruth) {
					Eigen::Affine3d const transform{gtOrigin * pose};
					Eigen::Vector3f const t = transform.translation().cast<float>();
					cv::Point2f const pt{static_cast<float_t >(-t[0]), static_cast<float_t >(t[2])};
					cv::circle(overlay, pp - (pt * scale), 1, cv::Scalar{0, 255, 0}, -1, 8, 0);
				}
			}

			Eigen::Affine3d const origin{trajectory.back().inverse()};
			for (auto &&pose : trajectory) {
				Eigen::Affine3d const transform{origin * pose};
				Eigen::Vector3f const t = transform.translation().cast<float>();
				cv::Point2f const pt{-t[0], t[2]};
				cv::circle(overlay, pp - (pt * scale), 1, cv::Scalar{255, 0, 255}, -1, 8, 0);
			}

			cv::imshow("result", overlay);
			auto const key = cv::waitKey(play);
			if (ESC_KEY == key) { // exit when the Escape key is pressed
				frameIdx = std::numeric_limits<int>::max() - 1;
			} else if (ESC_SPACE == key) { // toggle pause/play
				play = !play;
			} else if (43 == key || 171 == key) { // +
				trajectoryScale /= 2.f;
			} else if (45 == key || 173 == key) { // -
				trajectoryScale *= 2.f;
			}
		}

		//
		// "remember"
		//

		prevLeftImage = std::move(leftImage);

	}

	if (not std::empty(opts->mTrajectoryFile)) {
		saveTrajectory(opts->mTrajectoryFile, trajectory);
	}

	cv::destroyAllWindows();

	return EXIT_SUCCESS;
}
