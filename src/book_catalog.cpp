#include "book_catalog.h"
#include "csv_reader.h"
#include <stdexcept>

namespace {
// Returns the first header name present in `headerIndex`, or -1 if none match.
long findColumn(const std::unordered_map<std::string, size_t>& headerIndex,
                 std::initializer_list<const char*> candidates) {
    for (const char* name : candidates) {
        auto it = headerIndex.find(name);
        if (it != headerIndex.end()) return static_cast<long>(it->second);
    }
    return -1;
}
}  // namespace

void BookCatalog::load(const std::string& path) {
    std::unordered_map<std::string, size_t> headerIndex;
    std::vector<std::vector<std::string>> rows;
    CsvReader::readWithHeader(path, ';', headerIndex, rows);

    long isbnCol = findColumn(headerIndex, {"ISBN"});
    long titleCol = findColumn(headerIndex, {"Title", "Book-Title"});
    long authorCol = findColumn(headerIndex, {"Author", "Book-Author"});
    long yearCol = findColumn(headerIndex, {"Year", "Year-Of-Publication"});
    long publisherCol = findColumn(headerIndex, {"Publisher"});

    if (isbnCol < 0) {
        throw std::runtime_error("Books.csv is missing an ISBN column");
    }

    books_.clear();
    books_.reserve(rows.size());

    for (const auto& row : rows) {
        if (static_cast<size_t>(isbnCol) >= row.size()) continue;
        const std::string& isbn = row[isbnCol];
        if (isbn.empty()) continue;

        // Keep the first occurrence only (mirrors drop_duplicates(subset='ISBN')).
        if (books_.find(isbn) != books_.end()) continue;

        BookInfo info;
        info.title     = (titleCol >= 0 && static_cast<size_t>(titleCol) < row.size()) ? row[titleCol] : "";
        info.author    = (authorCol >= 0 && static_cast<size_t>(authorCol) < row.size()) ? row[authorCol] : "";
        info.year      = (yearCol >= 0 && static_cast<size_t>(yearCol) < row.size()) ? row[yearCol] : "";
        info.publisher = (publisherCol >= 0 && static_cast<size_t>(publisherCol) < row.size()) ? row[publisherCol] : "";

        books_.emplace(isbn, std::move(info));
    }
}

const BookInfo* BookCatalog::lookup(const std::string& isbn) const {
    auto it = books_.find(isbn);
    return it != books_.end() ? &it->second : nullptr;
}
