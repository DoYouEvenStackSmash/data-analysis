#include <iostream>
#include <fstream>
#include "image_generated.h" // Include the generated C++ header
#include <chrono>
// using std::chrono;

using namespace MyData; // Use the namespace from your schema

std::vector<ImageT> convertToImageTVector(const flatbuffers::Vector<flatbuffers::Offset<MyData::Image>>* images) {
    std::vector<ImageT> imageTVector;
    
    for (flatbuffers::uoffset_t i = 0; i < images->size(); ++i) {

        const Image* image = (*images)[i];
        
        ImageT imageT;
        imageT.shape = image->shape();
        imageT.data.assign(image->data()->begin(), image->data()->end());
        
        imageTVector.push_back(imageT);
    
    }
    
    return imageTVector;
}

int main(int argc, char* argv[]) {

    if (argc < 2) {
        std::cerr << "no file provided" << std::endl;
        return 1;
    }
    
    // Open the binary file
    char* filename = argv[1];

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open" << filename << std::endl;
        return 1;
    }
    auto start = std::chrono::steady_clock::now();
    // Read the binary data into a buffer
    std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // Verify and access the Data object
    const Data* data = flatbuffers::GetRoot<Data>(buffer.data());
    if (!data) {
        std::cerr << "Failed to parse Data object" << std::endl;
        return 1;
    }

    // Access and print the contents
    const flatbuffers::Vector<flatbuffers::Offset<Image>>* images = data->M();
    // const Image* image = (*images)[0];
    // ImageT* a = image->UnPack();
    // std::cout << a->shape << std::endl;
    // return 0;
    std::vector<ImageT> imageTVector = convertToImageTVector(images);
    // for (int n = 0; n < imageTVector.size(); n++)
    //     std::cout << n << std::endl;
    auto end = std::chrono::steady_clock::now();
    auto time_diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "time taken: " << time_diff << "ns\n";
    return 0;
    for (flatbuffers::uoffset_t i = 0; i < 10/*images->size()*/; ++i) {
        const Image* image = (*images)[i];
        std::cout << "Image " << i << ": Shape=" << image->shape() << ", Data Size=" << image->data()->size() << std::endl;
    }

    return 0;
}
