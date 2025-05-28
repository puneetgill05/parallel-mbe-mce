//
// Created by puneet on 03/05/25.
//

#ifndef PARLMBC_H
#define PARLMBC_H
#include <set>

#include "UndirectedGraph.h"
#include "utils.h"
#include <tbb/tbb.h>

#include "BipartiteGraph.h"

template<typename K, typename V>
using ConcurrentMap = tbb::concurrent_hash_map<K,V>;

class ParLmbc {
    int _min_size_threshold;
    string _outfile;
    // UndirectedGraph _ugraph;
    BipartiteGraph _graph;
    std::atomic<long> _numberOfMaximalBicliques = 0;
    void parlmbc(tbb::concurrent_unordered_set<Vertex>&, tbb::concurrent_unordered_set<Vertex>&,
    tbb::concurrent_unordered_set<Vertex>&, int,
    tbb::concurrent_map<vector<Vertex>, vector<Vertex>>&);
    void MineLMBCSeq(unordered_set<Vertex>&, unordered_set<Vertex>&, unordered_set<Vertex>&, int, map<vector<Vertex>, vector<Vertex>>&);


public:
    ParLmbc();
    // ParLmbc(UndirectedGraph &);
    // ParLmbc(UndirectedGraph &, int);
    ParLmbc(BipartiteGraph &, int);
    ParLmbc(BipartiteGraph &);
    void run(string);
    void runSeq(string);
    string runParMBE(string, int);
    void runParMBESeq(string);
    long getCount();
    virtual ~ParLmbc();

};



#endif //PARLMBC_H
