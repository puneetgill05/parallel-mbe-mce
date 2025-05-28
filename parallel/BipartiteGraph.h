//
// Created by puneet on 13/05/25.
//

#ifndef BIPARTITEGRAPH_H
#define BIPARTITEGRAPH_H
#include <map>
#include <unordered_set>
#include <vector>
#include <string>
#include <oneapi/tbb/concurrent_unordered_set.h>

using namespace std;

class Vertex;

class BipartiteGraph {
    int numberOfVertices;
    int numberOfEdges;

public:
    std::map<Vertex, std::unordered_set<Vertex>> _adjacencyList;
    BipartiteGraph();
    explicit BipartiteGraph(map<string, vector<string>> inputGraph);

    static BipartiteGraph createGraph(std::map<string, vector<string>>);
    void addEdge(const Vertex &u, const Vertex &v);
    void addVertex(const Vertex &u);
    int getNumberOfVertices() const;
    int getNumberOfEdges() const;
    int degreeOf(const Vertex& v);
    int computeNumberOfVertices() const;
    int computeNumberOfEdges();
    void print();
    unordered_set<Vertex>& neighbor(const Vertex &);
    tbb::concurrent_unordered_set<Vertex>& neighbor_concurrent(Vertex);
    void readInBiEdgeList(const string &fname);
    virtual ~BipartiteGraph();
};



#endif //BIPARTITEGRAPH_H
