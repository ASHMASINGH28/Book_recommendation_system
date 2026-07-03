#pragma once
#include <string>
#include <unordered_map>

struct BookInfo {
    std::string title;
    std::string author;
    std::string year;
    std::string publisher;
};

class BookCatalog {
public:
    // Loads Books.csv (';'-delimited). Tolerates either the notebook's
    // simplified headers (ISBN, Title, Author, Year, Publisher) or the
    // original Book-Crossing headers (ISBN, Book-Title, Book-Author,
    // Year-Of-Publication, Publisher).
    void load(const std::string& path);

    // Returns nullptr if the ISBN is not in the catalog.
    const BookInfo* lookup(const std::string& isbn) const;

    size_t size() const { return books_.size(); }

private:
    std::unordered_map<std::string, BookInfo> books_;
};
