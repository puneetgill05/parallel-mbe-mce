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

    Vertex getPivot(unordered_set<Vertex> &cand, unordered_set<Vertex>& cand_union_fini);
    Vertex getParPivot(tbb::concurrent_unordered_set<Vertex> cand, tbb::concurrent_unordered_set<Vertex> cand_union_fini);
    void TTT(list<Vertex> K, unordered_set<Vertex> cand, unordered_set<Vertex> fini, vector<vector<Vertex>>&
    cliques, int& numCliques, int& calls);

    void TTT_loop(list<Vertex> K, unordered_set<Vertex> cand, unordered_set<Vertex> fini);
    void TTT_loop2(list<Vertex> K, unordered_set<Vertex> cand, unordered_set<Vertex> fini);

    tbb::concurrent_unordered_set<Vertex> parTTT(tbb::concurrent_unordered_set<Vertex> K, tbb::concurrent_unordered_set<Vertex> cand,
        tbb::concurrent_unordered_set<Vertex> fini, tbb::concurrent_vector<tbb::concurrent_set<Vertex>>& cliques, int& numCliques);

public:
    explicit CliqueEnum(UndirectedGraph &graph);
    void runTTT(list<Vertex> K, unordered_set<Vertex> cand, unordered_set<Vertex> fini);

    void runParTTT(tbb::concurrent_unordered_set<Vertex> K, tbb::concurrent_unordered_set<Vertex> cand,
    tbb::concurrent_unordered_set<Vertex> fini, int nthreads);


    vector<vector<tuple<Vertex, Vertex>>> find_bicliquesbp2(const map<tuple<Vertex, Vertex>,
        tuple<Vertex, Vertex, int>>& em, const unordered_map<Vertex, vector<Vertex>>& up, const unordered_map<Vertex, vector<Vertex>>& pu);

};



#endif //CLIQUEENUM_H
