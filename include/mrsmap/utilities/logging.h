#ifndef LOGGING_H
#define LOGGING_H

#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>

#define LOG_OUTPUT 1

#define LOG_STREAM(args) \
    if ( LOG_OUTPUT ) std::cout << std::setprecision (15) << args << std::endl;

#define OUTPUT_STREAM(args) \
    std::cout << std::setprecision (15) << args << std::endl;

#endif // LOGGING_H
