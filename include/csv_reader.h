#pragma once
#include <string>
#include <vector>
#include <unordered_map>

// A minimal, dependency-free CSV reader that supports:
//  - a configurable delimiter (the Book-Crossing dataset uses ';')
//  - quoted fields (fields wrapped in "...") including embedded delimiters
//  - a header row used to build a name -> column-index lookup
class CsvReader {
public:
    // Loads the whole file into memory as rows of string fields.
    // Throws std::runtime_error if the file cannot be opened.
    static std::vector<std::vector<std::string>> readAll(const std::string& path,
                                                           char delimiter = ';');

    // Convenience: reads the file and returns (header-index map, data rows).
    // The header row (row 0) is consumed and used to build the index map;
    // `dataRows` contains every row after the header.
    static void readWithHeader(const std::string& path,
                                char delimiter,
                                std::unordered_map<std::string, size_t>& headerIndex,
                                std::vector<std::vector<std::string>>& dataRows);

private:
    static std::vector<std::string> splitLine(const std::string& line, char delimiter);
};
