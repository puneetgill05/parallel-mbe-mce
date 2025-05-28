//
// Created by puneet on 28/04/25.
//

#include "Vertex.h"

#include <utility>
#include <functional>

Vertex::Vertex(string id) {
    _id = std::move(id);
}

string Vertex::getId() const {
    return _id;
}

bool Vertex::operator==(const Vertex& other) const {
    return _id == other._id;
}

bool Vertex::operator<(const Vertex& other) const {
    return _id < other._id;
}





