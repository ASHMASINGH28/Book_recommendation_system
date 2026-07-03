#include "csv_reader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

std::vector<std::string> CsvReader::splitLine(const std::string& line, char delimiter) {
    std::vector<std::string> fields;
    std::string field;
    bool inQuotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];

        if (inQuotes) {
            if (c == '"') {
                // Escaped quote ("") inside a quoted field -> literal quote
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    field += '"';
                    ++i;
                } else {
                    inQuotes = false;
                }
            } else {
                field += c;
            }
        } else {
            if (c == '"') {
                inQuotes = true;
            } else if (c == delimiter) {
                fields.push_back(field);
                field.clear();
            } else if (c == '\r') {
                // ignore stray carriage returns
            } else {
                field += c;
            }
        }
    }
    fields.push_back(field);
    return fields;
}

std::vector<std::vector<std::string>> CsvReader::readAll(const std::string& path, char delimiter) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }

    std::vector<std::vector<std::string>> rows;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        rows.push_back(splitLine(line, delimiter));
    }
    return rows;
}

void CsvReader::readWithHeader(const std::string& path,
                                char delimiter,
                                std::unordered_map<std::string, size_t>& headerIndex,
                                std::vector<std::vector<std::string>>& dataRows) {
    auto rows = readAll(path, delimiter);
    if (rows.empty()) {
        throw std::runtime_error("File is empty: " + path);
    }

    headerIndex.clear();
    const auto& header = rows[0];
    for (size_t i = 0; i < header.size(); ++i) {
        headerIndex[header[i]] = i;
    }

    dataRows.assign(rows.begin() + 1, rows.end());
}
