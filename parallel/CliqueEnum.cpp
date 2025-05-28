//
// Created by puneet on 17/05/25.
//

#include "CliqueEnum.h"

#include "utils.h"

CliqueEnum::CliqueEnum(UndirectedGraph& graph) {
    _graph = graph;
}

Vertex CliqueEnum::getPivot(set<Vertex> cand, set<Vertex> cand_union_fini) {
    int max_size = -1;
    Vertex max_u("");
    for (auto u : cand_union_fini) {
        auto intersect = utils::setIntersect<set<Vertex>>(_graph.neighbor(u), cand);
        int intersect_size = static_cast<int>(intersect.size());
            if (intersect_size > max_size) {
                max_size = intersect_size;
                max_u = u;
            }
    }
    return max_u;
}

Vertex CliqueEnum::getParPivot(tbb::concurrent_set<Vertex> cand, tbb::concurrent_set<Vertex> cand_union_fini) {
    tbb::concurrent_map<Vertex, int> t;
    parallel_for_each(cand_union_fini.begin(), cand_union_fini.end(), [&](Vertex w) {
        tbb::concurrent_set w_neighbours(_graph.neighbor(w).begin(), _graph.neighbor(w).end());
        tbb::concurrent_set<Vertex> cand_intersect_w_neighbours = utils::setIntersectParallel<Vertex>(w_neighbours, cand);

        t.insert({w, cand_intersect_w_neighbours.size()});
    });

    int max_size = -1;
    Vertex max_w("");

    parallel_for_each (cand_union_fini.begin(), cand_union_fini.end(), [&](Vertex w) {
        auto it = t.find(w);
        if (it != t.end()) {
             if (it->second > max_size) {
                 max_size = it->second;
                 max_w = w;
             }
        }
    });

    return max_w;
}


void CliqueEnum::TTT(set<Vertex> K, set<Vertex> cand, set<Vertex> fini, set<vector<Vertex>>& cliques, int& numCliques) {
    if (cand.empty() && fini.empty()) {
        // cout << "[";
        // for (auto const& v : K) {
        //     cout << " " << v.getId();
        // }
        // cout << " ]" << endl;
        if (!K.empty()) {
            // vector K_vector(K.begin(), K.end());
            // cliques.insert(K_vector);
            numCliques += 1;

        }
        if (numCliques % 1000 == 0) {
            cout << "Number of maximal cliques so far: " << numCliques << endl;
        }
        // return K;
    }

    auto cand_union_fini = utils::setUnion<set<Vertex>>(cand, fini);
    // set<Vertex> cand_union_fini;
    // for (const auto& [key, value] : _graph._adjacencyList) {
        // cand_union_fini.insert(key);
    // }
    Vertex pivot = getPivot(cand, cand_union_fini);

    set pivot_neighbours(_graph.neighbor(pivot).begin(), _graph.neighbor(pivot).end());
    auto ext = utils::setDifference<set<Vertex>>(cand, pivot_neighbours);

    for (auto const& q : ext) {
        if (fini.contains(q)) continue;

        set Kq(K.begin(), K.end());
        Kq.insert(q);

        set q_neighbours(_graph.neighbor(q).begin(), _graph.neighbor(q).end());
        vector<Vertex> cand_q;
        std::set_intersection(cand.begin(), cand.end(), q_neighbours.begin(), q_neighbours.end(),
        std::back_inserter(cand_q));
        vector<Vertex> fini_q;
        fini_q = utils::setIntersect<Vertex>(fini, q_neighbours);
        // std::set_intersection(fini.begin(), fini.end(), q_neighbours.begin(), q_neighbours.end(),
        // std::back_inserter(fini_q));

        set cand_q_set(cand_q.begin(), cand_q.end());
        set fini_q_set(fini_q.begin(), fini_q.end());

        cand.erase(q);
        fini.insert(q);

        TTT(Kq, cand_q_set, fini_q_set, cliques, numCliques);
    }
}

/**
 *
 * @param K A clique to extend
 * @param cand Set of vertices that can be used to extend K
 * @param fini Set of vertices that have been used to extend K
 * @return Set of all maximal cliques of G containing K and vertices from cand but not containing any vertex from fini
 */
void CliqueEnum::runTTT(set<Vertex> K, set<Vertex> cand, set<Vertex> fini) {
    set<vector<Vertex>> cliques;
    int numCliques = 0;
   TTT(K, cand, fini, cliques, numCliques);
    cout << "Number of maximal cliques: " << numCliques << endl;

}


tbb::concurrent_set<Vertex> CliqueEnum::parTTT(tbb::concurrent_set<Vertex> K, tbb::concurrent_set<Vertex> cand,
    tbb::concurrent_set<Vertex> fini, tbb::concurrent_vector<tbb::concurrent_set<Vertex>>& cliques, int &numCliques) {
    if (cand.empty() && fini.empty()) {
        if (!K.empty()) {
            // set K_set(K.begin(), K.end());
            // cliques.push_back(K);
            numCliques += 1;
        }
        return K;
    }

    auto cand_union_fini = utils::setUnionParallel<Vertex>(cand, fini);

    // set cand_union_fini_set(cand_union_fini.begin(), cand_union_fini.end());
    Vertex pivot = getParPivot(cand, cand_union_fini);

    tbb::concurrent_set pivot_neighbours(_graph.neighbor(pivot).begin(), _graph.neighbor(pivot).end());
    // set pivot_neighbours_set(pivot_neighbours.begin(), pivot_neighbours.end());

    auto ext_conc_set = utils::setDifferenceParallel<Vertex>(cand, pivot_neighbours);
    vector ext(ext_conc_set.begin(), ext_conc_set.end());
    // set ext_set(ext.begin(), ext.end());

    parallel_for(tbb::blocked_range<int>(0, ext.size()), [&](tbb::blocked_range<int> r) {
        for(int i = r.begin(); i < r.end(); i++) {
            auto q = ext[i];
            if (fini.contains(q)) continue;

            tbb::concurrent_set Kq(K.begin(), K.end());
            Kq.insert(q);
            // set Kq_set(Kq.begin(), Kq.end());

            tbb::concurrent_set q_neighbours(_graph.neighbor(q).begin(), _graph.neighbor(q).end());
            tbb::concurrent_set<Vertex> ext_prefix;
            for (int j = 0; j < i; j++) {
                ext_prefix.insert(ext[j]);
            }
            tbb::concurrent_set<Vertex> cand_minus_ext = utils::setDifferenceParallel(cand, ext_prefix);
            // set cand_minus_ext_set(cand_minus_ext.begin(), cand_minus_ext.end());
            tbb::concurrent_set<Vertex> fini_union_ext = utils::setUnionParallel(fini, ext_prefix);
            // set fini_union_ext_set(fini_union_ext.begin(), fini_union_ext.end());
            tbb::concurrent_set<Vertex> cand_q = utils::setIntersectParallel(cand_minus_ext, q_neighbours);
            // set cand_q_set(cand_q.begin(), cand_q.end());
            tbb::concurrent_set<Vertex> fini_q = utils::setIntersectParallel(fini_union_ext, q_neighbours);
            // set fini_q_set(fini_q.begin(), fini_q.end());

            if (Kq.size() < 1000) {
                set Kq_set(Kq.begin(), Kq.end());
                set cand_q_set(cand_q.begin(), cand_q.end());
                set fini_q_set(fini_q.begin(), fini_q.end());
                set<vector<Vertex>> cliques_set;
                TTT(Kq_set, cand_q_set, fini_q_set, cliques_set, numCliques);
                for (auto c : cliques_set) {
                    tbb::concurrent_set c_conc_set(c.begin(), c.end());
                    cliques.push_back(c_conc_set);
                }
            } else {
                parTTT(Kq, cand_q, fini_q, cliques, numCliques);
            }
        }
    });
    return {};
}


void CliqueEnum::runParTTT(tbb::concurrent_set<Vertex> K, tbb::concurrent_set<Vertex> cand,
tbb::concurrent_set<Vertex> fini, int nthreads) {
    tbb::global_control c(tbb::global_control::max_allowed_parallelism, nthreads);
    tbb::concurrent_vector<tbb::concurrent_set<Vertex>> cliques;
    int numCliques = 0;
    parTTT(K, cand, fini, cliques, numCliques);

    cout << "Number of maximal cliques: " << numCliques << endl;
    // for (auto const& c : cliques) {
    //     cout << "[";
    //     for (auto const& v : c) {
    //         cout << " " << v.getId();
    //     }
    //     cout << " ]" << endl;
    // }

}

