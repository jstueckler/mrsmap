#ifndef MRSSLAM_API_H
#define MRSSLAM_API_H

#ifdef _MSC_VER
// We are using a Microsoft compiler:
#ifdef mrsslam_EXPORTS
#define MRSSLAM_API __declspec(dllexport)
#else
#define MRSSLAM_API __declspec(dllimport)
#endif
#else
// Not Microsoft compiler so set empty definition:
#define MRSSLAM_API
#endif

#endif // MRSSLAM_API_H

