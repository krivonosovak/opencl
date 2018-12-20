
#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <assert.h>

#define __CL_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include "cl.hpp"


int main()
{
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {

        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        // load opencl source
        std::ifstream cl_file("scan.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
            cl_string.length() + 1));

        // create program
        cl::Program program(context, source);

        // compile opencl source
        try {
            program.build(devices);
        } catch(const cl::Error &err) {
            std::cerr
                    << "OpenCL compilation error" << std::endl
                    << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0])
                    << std::endl;
            throw err;
        }

        // create a message to send to kernel
        size_t const block_size = 4;
        size_t test_array_size = 67;
        
        std::ifstream in("input.txt");
        std::ofstream out("output.txt");
        in >> test_array_size;
        size_t old_array_size = test_array_size;
        
        if ((test_array_size % block_size) != 0) {
            test_array_size = (test_array_size / block_size + 1) * block_size;
        }
        
        std::vector<float> input(test_array_size, 0);
        std::vector<float> output(test_array_size, 0);
       
        for (int k = 0; k < old_array_size; ++k) {
                in >> input[k];
        }

        // allocate device buffer to hold message
        cl::Buffer dev_input(context, CL_MEM_READ_WRITE, sizeof(float) * test_array_size);

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * test_array_size, &input[0]);
        queue.finish();

        // load named kernel from opencl source
        
        cl::Kernel kernel_hs(program, "scan_hillis_steele");
        cl::KernelFunctor scan_hs(kernel_hs, queue, cl::NullRange, cl::NDRange(test_array_size), cl::NDRange(block_size));
        
        cl::Kernel kernel_merge(program, "merge");
        cl::KernelFunctor merge(kernel_merge, queue, cl::NullRange, cl::NDRange(test_array_size), cl::NDRange(block_size));
       
        size_t level_size = test_array_size;
        int level = 1;
        

        // recursive scan 
        while (level_size > 1 || level_size == block_size) {
            cl::Event event = scan_hs(dev_input,
                                    cl::__local(sizeof(float) * block_size), cl::__local(sizeof(float) * block_size), 
                                    level, test_array_size);
            event.wait();
            level_size = level_size / block_size + (int)(level_size % block_size != 0);
            level *= block_size;
        }
        
        level /= (block_size * block_size);
       
        // merge scans
        while (level > 0) {
            cl::Event event = merge(dev_input,
                                    cl::__local(sizeof(float) * block_size), 
                                    level, test_array_size);
            event.wait();
            level /= block_size;
        }
    

        queue.enqueueReadBuffer(dev_input, CL_TRUE, 0, sizeof(float) * test_array_size, &output[0]);
        
        out.setf(std::ios::fixed);
        for (size_t i = 0; i < old_array_size; ++i) {
            out << std::setprecision(3) << output[i] << " ";
        }
        out << std::endl;

    }
    catch (cl::Error e)
    {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}