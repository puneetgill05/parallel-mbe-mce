//
// Created by puneet on 28/04/25.
//

#ifndef UNDIRECTEDGRAPH_H
#define UNDIRECTEDGRAPH_H
#include <map>
#include <string>
#include <unordered_set>
#include <vector>
#include <oneapi/tbb/concurrent_set.h>
#include <oneapi/tbb/concurrent_unordered_set.h>

#include "Vertex.h"

using namespace std;

class UndirectedGraph {
    int numberOfVertices;
    int numberOfEdges;

public:
    map<Vertex, unordered_set<Vertex>> _adjacencyList;
    UndirectedGraph();
    explicit UndirectedGraph(unordered_map<string, vector<string>> inputGraph);
    explicit UndirectedGraph(unordered_map<Vertex, vector<Vertex>> inputGraph);

    static UndirectedGraph createGraph(map<string, vector<string>>);
    void addEdge(const Vertex &u, const Vertex &v);
    void addVertex(Vertex u);
    int getNumberOfVertices();
    int getNumberOfEdges();
    int computeNumberOfVertices();
    int computeNumberOfEdges();
    void print();
    unordered_set<Vertex>& neighbor(Vertex);
    tbb::concurrent_unordered_set<Vertex>& neighbor_concurrent(Vertex);
    int degreeOf(Vertex&);
    long getLeftSize();
    long getRightSize();
    void readInBiEdgeList(const string &);


    virtual ~UndirectedGraph();
};



#endif //UNDIRECTEDGRAPH_H
