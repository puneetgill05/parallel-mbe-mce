//
// Created by puneet on 17/05/25.
//

#ifndef CLIQUEENUM_H
#define CLIQUEENUM_H
#include <set>
#include <tbb/tbb.h>

#include "UndirectedGraph.h"


class CliqueEnum {
    UndirectedGraph _graph;

    Vertex getPivot(set<Vertex> cand, set<Vertex> cand_union_fini);
    Vertex getParPivot(tbb::concurrent_set<Vertex> cand, tbb::concurrent_set<Vertex> cand_union_fini);
    void TTT(set<Vertex> K, set<Vertex> cand, set<Vertex> fini, set<vector<Vertex>>& cliques, int& numCliques);

    tbb::concurrent_set<Vertex> parTTT(tbb::concurrent_set<Vertex> K, tbb::concurrent_set<Vertex> cand,
        tbb::concurrent_set<Vertex> fini, tbb::concurrent_vector<tbb::concurrent_set<Vertex>>& cliques, int& numCliques);

public:
    explicit CliqueEnum(UndirectedGraph &graph);
    void runTTT(set<Vertex> K, set<Vertex> cand, set<Vertex> fini);

    void runParTTT(tbb::concurrent_set<Vertex> K, tbb::concurrent_set<Vertex> cand,
    tbb::concurrent_set<Vertex> fini, int nthreads);

};



#endif //CLIQUEENUM_H
