#include <iostream>
#include <string>
#include <fstream>
#include "converter.hpp"

using namespace std;

// The user provides an HRDi image as input and outputs a radiance image in exr format
int main(int argc, char *argv[])
{
    // Check if the user provided the correct number of arguments
    if(argc <=2)
    {
        std::cout << "Usage: " << argv[0] << " <input_image.hdr> <output_radiance_image.exr>" << std::endl;
        return 1;
    }
    
    // Check if the input file exists
    string input_file = argv[1];
    ifstream file(input_file);
    if (!file.good())
    {
        cout << "Error: The input file " << input_file << " does not exist." << std::endl;
        return 1;
    }

    // Check if the input file is an HDR image
    if (input_file.substr(input_file.find_last_of(".") + 1) != "hdr")
    {
        cout << "Error: The input file " << input_file << " is not an HDR image." << std::endl;
        return 1;
    }
    
    // Check if the output file is an EXR image
    string output_file = argv[2];
    if (output_file.substr(output_file.find_last_of(".") + 1) != "exr")
    {
        cout << "Error: The output file " << output_file << " is not an EXR image." << endl;
        return 1;
    }

    // Create a converter object and process the input image to generate the output image
    converter conv(input_file);    
    conv.process();
    conv.save(output_file);

    return 0;
}

