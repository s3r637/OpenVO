
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <opencv2/opencv.hpp>

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

			("start",
			 po::value<int>(&mStart)->default_value(0),
			 "start at frame")
			("step",
			 po::value<int>(&mStep)->default_value(1)->implicit_value(2),
			 "frame skipping, step size over frames")
			("end",
			 po::value<int>(&mEnd)->default_value(std::numeric_limits<int>::max()), // TODO: just keep it in mind
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
		}

		if (not fs::is_directory(mLeftFramesDir)) {
			throw std::invalid_argument("path given in --left does not exist");
		}
		if (not fs::is_directory(mRightFramesDir)) {
			throw std::invalid_argument("path given in --right does not exist");
		}
	}

	/**
	 * @brief Concatenates @a root and @a child path, if @a child is not empty and relative, otherwise returns @a child.
	 *
	 * @param root The root path.
	 * @param child The child path.
	 * @return root / child
	 */
	fs::path extend(fs::path const & root, fs::path const & child) {
		if (child.is_relative() and not child.empty()) {
			return root / child;
		}
		return child;
	}

	fs::path mBaseDir;
	fs::path mLeftFramesDir;
	fs::path mRightFramesDir;

	int mStart;
	int mStep;
	int mEnd;
};

/**
 * @brief Checks if given path @a p is hidden.
 *
 * @param p File path.
 * @return True if path is hidden, otherwise false.
 */
bool isHidden(fs::path const & p) {
	auto const name = p.filename();
	return name != ".." && name != "." && name.string()[0] == '.';
}

/**
 * @brief Checks if given path @a p is a regular file.
 *
 * @param p File path.
 * @return True if @p p is a regular file, otherwise false.
 */
bool isRegular(fs::path const & p) {
	return fs::regular_file == fs::status(p).type();
}

/**
 * @brief Collects all files of @a dir.
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

	// collect left & right frames
	std::vector<std::string> leftFramePaths = collectFiles(opts->mLeftFramesDir);
	std::vector<std::string> rightFramePaths = collectFiles(opts->mRightFramesDir);

	// sort both frame collections
	std::sort(std::begin(leftFramePaths), std::end(leftFramePaths));
	std::sort(std::begin(rightFramePaths), std::end(rightFramePaths));

	// TODO: left & right consistency check (size & name)

	opts->mEnd = static_cast<int>(std::size(leftFramePaths));

	cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

	// iterate over frames and feed them into OpenVO
	for (int id = opts->mStart; id < opts->mEnd; id += opts->mStep) {
		auto const & leftPath = leftFramePaths[id];
		auto const & rightPath = rightFramePaths[id];

		// load left image
		cv::Mat leftMat = cv::imread(leftPath, cv::IMREAD_UNCHANGED);
		if (leftMat.empty()) { // is input invalid
			std::cerr <<  "Could not open or find the left image: '" << leftPath << "'" << std::endl;
			return EXIT_FAILURE;
		}

		// load right image
		cv::Mat rightMat = cv::imread(rightPath, cv::IMREAD_UNCHANGED);
		if (rightMat.empty()) { // is input invalid
			std::cerr <<  "Could not open or find the right image: '" << rightMat << "'" << std::endl;
			return EXIT_FAILURE;
		}

		cv::Mat stereoMat = leftMat * 0.5f + rightMat * 0.5f;
		cv::imshow("result", stereoMat);
		auto const key = cv::waitKey(100);
		if (ESC_KEY == key) { // exit when the Escape key is pressed
			id = std::numeric_limits<int>::max() - 1;
		}

	}

	cv::destroyAllWindows();

	return EXIT_SUCCESS;
}
