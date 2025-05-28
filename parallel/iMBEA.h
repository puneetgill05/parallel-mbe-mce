//
// Created by puneet on 02/05/25.
//

#ifndef IMBEA_H
#define IMBEA_H
#include <string>
#include <unordered_set>

#include "utils.h"
#include "Vertex.h"
#include "BipartiteGraph.h"

using namespace std;

class iMBEA {
    BipartiteGraph _graph;
    int _numOfMaximalBicliques;

    // void enumerate(UndirectedGraph& graph, unordered_set<Vertex>& L, unordered_set<Vertex>& R,
        // unordered_set<Vertex>& P, unordered_set<Vertex>& Q, bool imbea, map<vector<Vertex>, vector<Vertex>>& bicliques);

public:
    iMBEA();

    explicit iMBEA(BipartiteGraph&);
    void run();
    void runPar();
    int getNumOfMaximalBicliques();
    virtual ~iMBEA();
};

#endif //IMBEA_H
