#ifndef MRSMAP_API_H
#define MRSMAP_API_H

#ifdef _MSC_VER
// We are using a Microsoft compiler:
#ifdef mrsmap_EXPORTS
#define MRSMAP_API __declspec(dllexport)
#else
#define MRSMAP_API __declspec(dllimport)
#endif
#else
// Not Microsoft compiler so set empty definition:
#define MRSMAP_API
#endif

#endif // MRSMAP_API_H

