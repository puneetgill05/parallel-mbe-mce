//
// Created by puneet on 28/04/25.
//

#ifndef VERTEX_H
#define VERTEX_H
#include <string>

using namespace std;

class Vertex {
    string _id;
public:
    Vertex(string id);

    string getId() const;
    bool operator==(const Vertex& other) const;
    bool operator<(const Vertex &other) const;
};

namespace std {
    template <>
    struct hash<Vertex> {
        size_t operator()(const Vertex& v) const {
            return hash<string>()(v.getId());  // use the public member `id`
        }
    };
}



#endif //VERTEX_H
