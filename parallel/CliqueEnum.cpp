//
// Created by puneet on 17/05/25.
//

#include "CliqueEnum.h"

#include <stack>

#include "abseil-cpp/absl/container/flat_hash_set.h"

#include "utils.h"

using Edge = std::tuple<Vertex, Vertex>;
struct EdgeHash {
    std::size_t operator()(const std::tuple<Vertex, Vertex>& t) const {
        auto h1 = std::hash<Vertex>()(std::get<0>(t));
        auto h2 = std::hash<Vertex>()(std::get<1>(t));
        return h1 ^ (h2 << 1); // Combine hashes
    }
};


CliqueEnum::CliqueEnum(UndirectedGraph& graph) {
    _graph = graph;
}

Vertex CliqueEnum::getPivot(unordered_set<Vertex> &cand, unordered_set<Vertex>& cand_union_fini) {
    int max_size = -1;
    Vertex max_u("");
    for (auto u : cand_union_fini) {
        auto intersect = utils::setIntersect<Vertex>(_graph.neighbor(u), cand);
        int intersect_size = static_cast<int>(intersect.size());
            if (intersect_size > max_size) {
                max_size = intersect_size;
                max_u = u;
            }
    }
    return max_u;
}

Vertex CliqueEnum::getParPivot(tbb::concurrent_unordered_set<Vertex> cand, tbb::concurrent_unordered_set<Vertex> cand_union_fini) {
    tbb::concurrent_map<Vertex, int> t;
    tbb::parallel_for_each(cand_union_fini.begin(), cand_union_fini.end(), [&](Vertex w) {
        tbb::concurrent_unordered_set w_neighbours(_graph.neighbor(w).begin(), _graph.neighbor(w).end());
        tbb::concurrent_unordered_set<Vertex> cand_intersect_w_neighbours = utils::setIntersectParallel<Vertex>(w_neighbours, cand);

        t.insert({w, cand_intersect_w_neighbours.size()});
    });

    int max_size = -1;
    Vertex max_w("");

    tbb::parallel_for_each (cand_union_fini.begin(), cand_union_fini.end(), [&](Vertex w) {
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


void CliqueEnum::TTT(list<Vertex> K, unordered_set<Vertex> cand, unordered_set<Vertex> fini,
vector<vector<Vertex>>& cliques, int&
numCliques, int& calls) {
    if (cand.empty() && fini.empty()) {
    // if (cand.empty()) {
        // cout << "[";
        // for (auto const& v : K) {
        //     cout << " " << v.getId();
        // }
        // cout << " ]" << endl;
        if (!K.empty()) {
            // vector K_vector(K.begin(), K.end());
            // cliques.push_back(K_vector);
            numCliques += 1;

        }
        if (numCliques % 1000 == 0) {
            cout << "Number of maximal cliques so far: " << numCliques << endl;
        }
        // return K;
        // return;
    } else {
        if (cand.empty()) return;
        calls += 1;

        auto cand_union_fini = utils::setUnion<Vertex>(cand, fini);

        Vertex pivot = getPivot(cand, cand_union_fini);

        unordered_set pivot_neighbours = _graph.neighbor(pivot);
        auto ext = utils::setDifference<Vertex>(cand, pivot_neighbours);

        for (auto const& q : ext) {
            if (fini.contains(q)) continue;

            list Kq(K.begin(), K.end());
            Kq.push_back(q);

            unordered_set q_neighbours(_graph.neighbor(q).begin(), _graph.neighbor(q).end());

            unordered_set<Vertex> cand_q = utils::setIntersect<Vertex>(cand, q_neighbours);
            unordered_set<Vertex> fini_q = utils::setIntersect<Vertex>(fini, q_neighbours);

            cand.erase(q);
            fini.insert(q);

            TTT(Kq, cand_q, fini_q, cliques, numCliques, calls);
        }
    }
}


void CliqueEnum::TTT_loop2(list<Vertex> K, unordered_set<Vertex> cand, unordered_set<Vertex> fini) {
    int numCliques = 0;
    int calls = 0;
    stack<tuple<list<Vertex>, unordered_set<Vertex>, unordered_set<Vertex>, unordered_set<Vertex>>> s;

    auto cand_union_fini = utils::setUnion<Vertex>(cand, fini);
    Vertex pivot = getPivot(cand, cand_union_fini);
    unordered_set pivot_neighbours = _graph.neighbor(pivot);
    auto ext = utils::setDifference<Vertex>(cand, pivot_neighbours);
    s.emplace(K, cand, fini, ext);

    vector<list<Vertex>> cliques_seen;

    while (true) {
        if (s.empty()) break;

        if (!ext.empty()) {
            calls += 1;
            if (calls % 100000 == 0) {
                cout << "Number of calls: " << calls << endl;
            }

            auto q = *ext.begin();
            ext.erase(q);
            if (fini.contains(q)) continue;

            K.push_back(q);
            cand.erase(q);
            fini.insert(q);

            unordered_set q_neighbours(_graph.neighbor(q).begin(), _graph.neighbor(q).end());
            unordered_set<Vertex> cand_q = utils::setIntersect<Vertex>(cand, q_neighbours);
            unordered_set<Vertex> fini_q = utils::setIntersect<Vertex>(fini, q_neighbours);

            s.emplace(K, cand, fini, ext);

            // update
            cand = cand_q;
            fini = fini_q;
            pivot = getPivot(cand, cand);
            pivot_neighbours = _graph.neighbor(pivot);
            ext = utils::setDifference<Vertex>(cand, pivot_neighbours);
        } else {
            if (cand.empty() && fini.empty() && !K.empty()) {
                if (std::find(cliques_seen.begin(), cliques_seen.end(), K) == cliques_seen.end()) {
                    cliques_seen.push_back(K);
                    // cout << "clique: ";
                    // for (const auto& k : K) {
                    //     cout << k.getId() << " ";
                    // }
                    // cout << endl;
                    numCliques += 1;
                    if (numCliques % 1000 == 0) {
                        cout << "Number of maximal cliques so far: " << numCliques << endl;
                    }
                } else {
                    cout << "clique already seen" << endl;
                }
            }
            while (ext.empty() || cand.empty()) {
                tuple<list<Vertex>, unordered_set<Vertex>, unordered_set<Vertex>, unordered_set<Vertex>>
                stack_top = s.top();
                s.pop();
                // K.clear();
                K = get<0>(stack_top);
                cand = get<1>(stack_top);
                fini = get<2>(stack_top);
                ext = get<3>(stack_top);
                for (auto f : fini) {
                    if (find(K.begin(), K.end(), f) != K.end()) {
                        K.remove(f);
                    }
                    if (find(ext.begin(), ext.end(), f) != ext.end()) {
                        ext.erase(f);
                    }
                }
            }
        }

    }
    cout << "Number of maximal cliques: " << numCliques << endl;

}


static unordered_set<Edge, EdgeHash> setIntersect(const unordered_set<Edge, EdgeHash>& S1, const unordered_set<Edge, EdgeHash>& S2) {
    const unordered_set<Edge, EdgeHash>& small = (S1.size() < S2.size()) ? S1 : S2;
    const unordered_set<Edge, EdgeHash>& large = (S1.size() < S2.size()) ? S2 : S1;

    unordered_set<Edge, EdgeHash> result;
    result.reserve(std::min(S1.size(), S2.size()));  // Reserve space

    for (const auto& x : small) {
        if (large.contains(x)) {
            result.insert(x);
        }
    }
    return result;
}

static unordered_set<Edge, EdgeHash> setDifference(unordered_set<Edge, EdgeHash> S1, unordered_set<Edge, EdgeHash> S2) {
    unordered_set<Edge, EdgeHash> Z;
    for (const auto& s1 : S1) {
        if (!S2.contains(s1)) {
            Z.insert(s1);
        }
    }
    return Z;
}


Edge getPivotEdge(unordered_set<Edge, EdgeHash> &cand, unordered_set<Edge, EdgeHash>& subg, unordered_map<Edge,
unordered_set<Edge, EdgeHash>, EdgeHash> adj) {
    int max_size = -1;
    Edge max_u = make_tuple(Vertex(""), Vertex(""));
    for (auto u : subg) {
        auto intersect = setIntersect(adj[u], cand);
        int intersect_size = static_cast<int>(intersect.size());
        if (intersect_size > max_size) {
            max_size = intersect_size;
            max_u = u;
        }
    }
    return max_u;
}

vector<vector<Edge>> CliqueEnum::find_bicliquesbp2(
    const map<Edge, tuple<Vertex, Vertex, int>>& em,
    const unordered_map<Vertex, vector<Vertex>>& up,
    const unordered_map<Vertex, vector<Vertex>>& pu
) {

    if (up.empty()) return {};

    set<Edge> edgeset = utils::getedgeset(em, up);
    unordered_set<Edge, EdgeHash> edges(edgeset.begin(), edgeset.end());
    vector<vector<Edge>> cliques;

    unordered_map<Edge, unordered_set<Edge, EdgeHash>, EdgeHash> adj;
    for (const auto& u : edges) {
        auto nbrs = utils::getNeighbouringEdges(u, em, up, pu);
        unordered_set<Edge, EdgeHash> nbrs_unordered(nbrs.begin(), nbrs.end());
        adj[u] = nbrs_unordered;
        adj[u].erase(u);
    }

    std::vector<Edge> Q;
    Edge none = make_tuple(Vertex(""), Vertex(""));
    Q.push_back(none);

    std::unordered_set<Edge, EdgeHash> cand(edgeset.begin(), edgeset.end());
    std::unordered_set<Edge, EdgeHash> subg = cand;
    std::vector<std::tuple<
    std::unordered_set<Edge, EdgeHash>,
    std::unordered_set<Edge, EdgeHash>,
    std::unordered_set<Edge, EdgeHash>>> stack;

    // Edge u = *ranges::max_element(subg,
    //     [&](const Edge& a, const Edge& b) {
    //         return ranges::count_if(adj[a], [&](const Edge& x) { return cand.count(x); }) <
    //             ranges::count_if(adj[b], [&](const Edge& x) { return cand.count(x); });
    // });
    Edge u = getPivotEdge(cand, subg, adj);

    auto ext_u = setDifference(cand, adj[u]);
    int calls = 0;

    while (true) {
        calls++;
        if (!ext_u.empty()) {
            auto it = ext_u.begin();
            Edge q = *it;
            ext_u.erase(it);
            cand.erase(q);
            Q.back() = q;

            auto adj_q = adj[q];
            unordered_set<Edge, EdgeHash> subg_q;
            subg_q = setIntersect(subg, adj_q);

            if (subg_q.empty()) {
                cliques.push_back(Q);
            } else {
                unordered_set<Edge, EdgeHash> cand_q;
                cand_q = setIntersect(cand, adj_q);

                if (!cand_q.empty()) {
                    stack.emplace_back(subg, cand, ext_u);
                    Q.push_back(none);
                    subg = subg_q;
                    cand = cand_q;

                    u = *ranges::max_element(subg,
                        [&](const Edge& a, const Edge& b) {
                            return ranges::count_if(adj[a], [&](const Edge& x) { return cand.count(x); }) <
                                ranges::count_if(adj[b], [&](const Edge& x) { return cand.count(x); });
                    });
                    // u = getPivotEdge(cand, subg, adj);

                    ext_u.clear();
                    ext_u = setDifference(cand, adj[u]);
                }
            }
        } else {
            Q.pop_back();
            if (stack.empty())
                break;
            std::tie(subg, cand, ext_u) = stack.back();
            stack.pop_back();
        }
    }
    cout << "# calls: " << calls << endl;
    return cliques;
}





void CliqueEnum::TTT_loop(list<Vertex> K, unordered_set<Vertex> cand, unordered_set<Vertex> fini) {
    int numCliques = 0;
    int calls = 0;
    fini = cand;
    stack<tuple<list<Vertex>, unordered_set<Vertex>, unordered_set<Vertex>, unordered_set<Vertex>>> s;

    auto cand_union_fini = utils::setUnion<Vertex>(cand, fini);
    Vertex pivot = getPivot(cand, fini);
    unordered_set pivot_neighbours = _graph.neighbor(pivot);
    auto ext = utils::setDifference<Vertex>(cand, pivot_neighbours);


    // s.emplace(K, cand, fini, ext);

    while (true) {
        // if (!s.empty()) {
        //     tuple<unordered_set<Vertex>, unordered_set<Vertex>, unordered_set<Vertex>> stack_top = s.top();
        //     s.pop();
        //     K = get<0>(stack_top);
        //     cand = get<1>(stack_top);
        //     fini = get<2>(stack_top);
        // }
        //
        // if (fini.empty() && cand.empty()) {
        //     numCliques += 1;
        //     if (numCliques % 1000 == 0) {
        //         cout << "Number of maximal cliques so far: " << numCliques << endl;
        //     }
        // }

        calls += 1;
        // if (calls % 10000 == 0) {
        //     cout << "Number of calls: " << calls << endl;
        // }
        if (!ext.empty()) {
            auto q = *ext.begin();
            // if (!fini.contains(q)) {
            ext.erase(q);
            cand.erase(q);
            K.push_back(q);

            unordered_set q_neighbours(_graph.neighbor(q).begin(), _graph.neighbor(q).end());
            unordered_set<Vertex> fini_q = utils::setIntersect<Vertex>(fini, q_neighbours);
            if (fini_q.empty()) {
                numCliques += 1;
                cout << "Clique found: ";
                for (auto k : K) {
                    cout << k.getId() << " ";
                }
                cout << endl;
                if (numCliques % 1000 == 0) {
                    cout << "# cliques: " << numCliques << endl;
                }
            } else {
                unordered_set<Vertex> cand_q = utils::setIntersect<Vertex>(cand, q_neighbours);

                if (!cand_q.empty()) {
                    s.emplace(K, cand, fini, ext);
                    cand = cand_q;
                    fini = fini_q;
                    pivot = getPivot(cand, fini);
                    pivot_neighbours = _graph.neighbor(pivot);
                    ext = utils::setDifference<Vertex>(cand, pivot_neighbours);
                }
            }
        } else {
            if (!K.empty()) {
                K.pop_back();
            }
            if (!s.empty()) {
                tuple<list<Vertex>, unordered_set<Vertex>, unordered_set<Vertex>, unordered_set<Vertex>>
                stack_top = s.top();
                s.pop();
                // K = get<0>(stack_top);
                cand = get<1>(stack_top);
                fini = get<2>(stack_top);
                ext = get<3>(stack_top);
            }
        }
        // }
    }
}


/**
 *
 * @param K A clique to extend
 * @param cand Set of vertices that can be used to extend K
 * @param fini Set of vertices that have been used to extend K
 * @return Set of all maximal cliques of G containing K and vertices from cand but not containing any vertex from fini
 */
void CliqueEnum::runTTT(list<Vertex> K, unordered_set<Vertex> cand, unordered_set<Vertex> fini) {
    vector<vector<Vertex>> cliques;
    int numCliques = 0;
    int calls = 0;
    // TTT(K, cand, fini, cliques, numCliques, calls);
    TTT_loop2(K, cand, fini);
    // TTT_loop(K, cand, fini);
    cout << "Number of maximal cliques: " << numCliques << endl;
    // cout << "Number of calls: " << calls << endl;

}


tbb::concurrent_unordered_set<Vertex> CliqueEnum::parTTT(tbb::concurrent_unordered_set<Vertex> K, tbb::concurrent_unordered_set<Vertex> cand,
    tbb::concurrent_unordered_set<Vertex> fini, tbb::concurrent_vector<tbb::concurrent_set<Vertex>>& cliques, int &numCliques) {
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

    tbb::concurrent_unordered_set pivot_neighbours(_graph.neighbor(pivot).begin(), _graph.neighbor(pivot).end());
    // set pivot_neighbours_set(pivot_neighbours.begin(), pivot_neighbours.end());

    auto ext_conc_set = utils::setDifferenceParallel<Vertex>(cand, pivot_neighbours);
    vector ext(ext_conc_set.begin(), ext_conc_set.end());
    // set ext_set(ext.begin(), ext.end());

    parallel_for(tbb::blocked_range<int>(0, ext.size()), [&](tbb::blocked_range<int> r) {
        for(int i = r.begin(); i < r.end(); i++) {
            auto q = ext[i];
            if (fini.contains(q)) continue;

            tbb::concurrent_unordered_set Kq(K.begin(), K.end());
            Kq.insert(q);
            // set Kq_set(Kq.begin(), Kq.end());

            tbb::concurrent_unordered_set q_neighbours(_graph.neighbor(q).begin(), _graph.neighbor(q).end());
            tbb::concurrent_unordered_set<Vertex> ext_prefix;
            for (int j = 0; j < i; j++) {
                ext_prefix.insert(ext[j]);
            }
            tbb::concurrent_unordered_set<Vertex> cand_minus_ext = utils::setDifferenceParallel(cand, ext_prefix);
            // set cand_minus_ext_set(cand_minus_ext.begin(), cand_minus_ext.end());
            tbb::concurrent_unordered_set<Vertex> fini_union_ext = utils::setUnionParallel(fini, ext_prefix);
            // set fini_union_ext_set(fini_union_ext.begin(), fini_union_ext.end());
            tbb::concurrent_unordered_set<Vertex> cand_q = utils::setIntersectParallel(cand_minus_ext, q_neighbours);
            // set cand_q_set(cand_q.begin(), cand_q.end());
            tbb::concurrent_unordered_set<Vertex> fini_q = utils::setIntersectParallel(fini_union_ext, q_neighbours);
            // set fini_q_set(fini_q.begin(), fini_q.end());

            if (Kq.size() < 1000) {
                list Kq_list(Kq.begin(), Kq.end());
                unordered_set cand_q_set(cand_q.begin(), cand_q.end());
                unordered_set fini_q_set(fini_q.begin(), fini_q.end());
                vector<vector<Vertex>> cliques_set;
                int c = 0;
                TTT(Kq_list, cand_q_set, fini_q_set, cliques_set, numCliques, c);
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


void CliqueEnum::runParTTT(tbb::concurrent_unordered_set<Vertex> K, tbb::concurrent_unordered_set<Vertex> cand,
tbb::concurrent_unordered_set<Vertex> fini, int nthreads) {
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

