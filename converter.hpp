#include <string>

class converter
{
public:
    converter(std::string input_file);
    ~converter();
    bool process();
    void save(std::string output_file);
private:
    float *data = nullptr;
    int width = -1; 
    int height = -1;
    int channels = -1;
};
