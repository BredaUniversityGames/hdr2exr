#include "converter.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <iostream>
#include <fstream>

using namespace std;

converter::converter(std::string input_file)
{
    // We assume that the input file exists and is an HDR image

    // Load the HDR image    
    data = stbi_loadf(input_file.c_str(), &width, &height, &channels, 0);
    if (!data)
    {
        cout << "Error: The input file " << input_file << " is not a valid HDR image." << endl;
        return;
    }
}

converter::~converter()
{
    // Free the memory allocated for the HDR image
    stbi_image_free(data);
}

bool converter::process()
{
    // We don't do anything here for now
    return true;
}

void converter::save(std::string output_file)
{
    // Save the radiance image in EXR format
    stbi_write_hdr(output_file.c_str(), width, height, channels, data);
}