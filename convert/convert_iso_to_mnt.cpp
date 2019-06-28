
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <vector>
#include <algorithm>
// #include <filesystem>
// namespace fs = std::filesystem;
// #include <experimental/filesystem>
// namespace fs = std::experimental::filesystem;
#include <sys/types.h>
#include <dirent.h>

template <unsigned byteN>
unsigned bigEndian(const unsigned char (&bytes)[byteN])
{
	unsigned result = 0;
	for (unsigned n = 0; n < byteN; n++)
	{
		result = (result << 8) + bytes[n];
	}
	return result;
}

// References:
// http://www.italdata-roma.com/PDF/Norme%20ISO-IEC%20Minutiae%20Data%20Format%2019794-2.pdf
// https://biolab.csr.unibo.it/fvcongoing/UI/Form/Download.aspx (ISO section, sample ZIP, ISOTemplate.pdf)
//
// Format (all numbers are big-endian)
#pragma pack(1)
struct fingerprintFile
{
	unsigned char magic[4];			  // 4B magic "FMR\0"
	unsigned char version[4];		  // 4B version (ignored, set to " 20\0"
	unsigned char total_length[4];  // 4B total length (including header)
	unsigned char trash0[2];		  // 2B rubbish (zeroed)
	unsigned char sizeX[2];			  // 2B image size in pixels X
	unsigned char sizeY[2];			  // 2B image size in pixels Y
	unsigned char ppcmX[2];			  // 2B rubbish (pixels per cm X, set to 196 = 500dpi)
	unsigned char ppcmY[2];			  // 2B rubbish (pixels per cm Y, set to 196 = 500dpi)
	unsigned char fingerN[1];		  // 1B rubbish (number of fingerprints, set to 1)
	unsigned char trash1[1];		  // 1B rubbish (zeroed)
	unsigned char fingerPos[1];	  // 1B rubbish (finger position, zeroed)
	unsigned char trash2[1];		  // 1B rubbish (zeroed)
	unsigned char fingerQuality[1]; // 1B rubbish (fingerprint quality, set to 100)
	unsigned char minutiaeCount[1]; // 1B minutia count
											  // N*6B minutiae
											  //      2B minutia position X in pixels
											  //      2B minutia position Y in pixels (upper 2b ignored, zeroed)
											  //      1B direction, compatible with SourceAFIS angles
											  //      1B quality (ignored, zeroed)
											  // 2B rubbish (extra data length, zeroed)
											  // N*1B rubbish (extra data)
};

#pragma pack(1)
struct minutiaFile
{
	unsigned char posX[2]; // 2B minutia position X in pixels
	//          2b (upper) minutia type (01 ending, 10 bifurcation, 00 other)
	unsigned char posY[2];		 // 2B minutia position Y in pixels (upper 2b ignored, zeroed)
	unsigned char direction[1]; // 1B direction, compatible with SourceAFIS angles
	unsigned char quality[1];	// 1B quality (ignored, zeroed)
};

struct minutia
{
	int posX;
	int posY;
	double direction;
	int type;
};

struct fingerprint
{
	int sizeX;
	int sizeY;
	int ppcmX;
	int ppcmY;
	int fingerN;
	int fingerPos;
	int fingerQuality;
	std::vector<minutia> minutaeVec;

	fingerprint(std::string const &filename)
	{
		std::ifstream istream(filename, std::ios::binary);
		// std::string readBytes(12);
		setContents(istream);
	}

	void readMinutiae(std::ifstream &istream, int minutiaeCount)
	{
		std::vector<char> buf(sizeof(minutiaFile));
		for (int i = 0; i < minutiaeCount; ++i)
		{
			istream.read(buf.data(), buf.size());
			minutiaFile *file = reinterpret_cast<minutiaFile *>(buf.data());
			minutia min;

			//          2b (upper) minutia type (01 ending, 10 bifurcation, 00 other)
			min.type = (file->posX[0] & 0b11000000) >> 6;
			// delete type bits
			file->posX[0] = (file->posX[0] & 0b00111111);
			min.posX = bigEndian(file->posX);
			min.posY = bigEndian(file->posY);
			min.direction = (double)bigEndian(file->direction);
			// convert angle to radians
			min.direction *= 0.0174533;
			minutaeVec.push_back(min);
		}
	}

	void setContents(std::ifstream &istream)
	{
		std::vector<char> buf(sizeof(fingerprintFile));
		istream.read(buf.data(), buf.size());

		fingerprintFile *file = reinterpret_cast<fingerprintFile *>(buf.data());

		sizeX = bigEndian(file->sizeX);
		sizeY = bigEndian(file->sizeY);
		ppcmX = bigEndian(file->ppcmX);
		ppcmY = bigEndian(file->ppcmY);
		fingerN = bigEndian(file->fingerN);
		fingerPos = bigEndian(file->fingerPos);
		fingerQuality = bigEndian(file->fingerQuality);

		int minutiaeCount = bigEndian(file->minutiaeCount);
		readMinutiae(istream, minutiaeCount);
	}
};

std::ostream &operator<<(std::ostream &os, const fingerprint &finger)
{
	os << finger.minutaeVec.size() << " " << finger.sizeX << " " << finger.sizeY << "\n";
	for (const minutia & minu : finger.minutaeVec){
		os << minu.posX << " " << minu.posY << " " << minu.direction << " " << minu.type << "\n";
	}
	return os;
}

int main(int argc, char **argv)
{
	if (argc < 2){
		std::cerr << "Usage: " << argv[0] << " dirname" << std::endl;
		return 0;
	}
	std::clog << "Opening " << argv[1] << std::endl;

	std::string dirpath(argv[1]);

	DIR *dirp = opendir(dirpath.c_str());
	struct dirent *dp;
	while ((dp = readdir(dirp)) != NULL)
	{
		std::string filename = dp->d_name;
		if (filename == "." || filename == "..")
		{
			continue;
		}
		std::cout << "Opening: " << filename << std::endl;
		// extract name / remove file extension
		std::string delimiter = ".";
		std::string name = filename.substr(0, filename.find(delimiter));
		// open file to write in plaintext
		std::ofstream outFile(name + std::string(".mnt"));
		// open fingerprint file
		fingerprint fingerfile(dirpath + "/" + filename);
		// write fingerprint to file
		outFile << name << std::endl;
		outFile << fingerfile << std::flush;
	}
	closedir(dirp);

	return 0;
}
