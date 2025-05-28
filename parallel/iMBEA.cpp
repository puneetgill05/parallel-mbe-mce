//
// Created by puneet on 02/05/25.
//

#include "iMBEA.h"

#include <oneapi/tbb/concurrent_set.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_for_each.h>

#include "utils.h"
#include "Vertex.h"

iMBEA::iMBEA() {
    _numOfMaximalBicliques = 0;
}

iMBEA::iMBEA(BipartiteGraph &g) {
    _graph = g;
    _numOfMaximalBicliques = 0;
}

vector<Vertex> sortByCommonNeighbourhoodSize(vector<Vertex>& P) {
    return P;
}

tbb::concurrent_vector<Vertex> sortByCommonNeighbourhoodSize(tbb::concurrent_vector<Vertex>& P) {
    return P;
}


/** Description: Get neighbours of x in L
 **/
unordered_set<Vertex> getNeighboursOf(BipartiteGraph graph, Vertex &x, unordered_set<Vertex>& L) {
    unordered_set<Vertex> Lprime;
    for (const auto& l : L) {
        if (graph._adjacencyList[x].contains(l)) {
            Lprime.insert(l);
        }
    }
    return Lprime;
}

tbb::concurrent_set<Vertex> getNeighboursOf(BipartiteGraph graph, Vertex &x, tbb::concurrent_set<Vertex>& L) {
    tbb::concurrent_set<Vertex> Lprime;
    tbb::parallel_for_each(L.begin(), L.end(), [&](Vertex &l) {
        if (graph._adjacencyList[x].contains(l)) {
            Lprime.insert(l);
        }
    });
    return Lprime;
}

unordered_set<Vertex> difference(unordered_set<Vertex>& X, unordered_set<Vertex>& Y) {
    unordered_set<Vertex> Z;
    for (const Vertex& x : X) {
        if (!Y.contains(x)) {
            Z.insert(x);
        }
    }
    return Z;
}

tbb::concurrent_set<Vertex> difference(tbb::concurrent_set<Vertex>& X, tbb::concurrent_set<Vertex>& Y) {
    tbb::concurrent_set<Vertex> Z;
    tbb::parallel_for_each(X.begin(), X.end(), [&](Vertex &x) {
        if (!Y.contains(x)) {
            Z.insert(x);
        }
    });
    return Z;
}

unordered_set<Vertex> setUnion(unordered_set<Vertex>& X, unordered_set<Vertex>& Y) {
    unordered_set<Vertex> Z;
    for (const auto& x : X) {
        Z.insert(x);
    }
    for (const auto& y : Y) {
        Z.insert(y);
    }
    return Z;
}

tbb::concurrent_set<Vertex> setUnion(tbb::concurrent_set<Vertex>& X, tbb::concurrent_set<Vertex>& Y) {
    tbb::concurrent_set<Vertex> Z;
    tbb::parallel_for_each(X.begin(), X.end(), [&](const Vertex &x) {
        Z.insert(x);
    });
    parallel_for_each(Y.begin(), Y.end(), [&](const Vertex& y) {
        Z.insert(y);
    });
    return Z;
}

vector<Vertex> notInSet(vector<Vertex>& X, const unordered_set<Vertex>& Y) {
    vector<Vertex> Z;
    for (const Vertex& x : X) {
        if (!Y.contains(x)) {
            Z.push_back(x);
        }
    }
    return Z;
}

vector<Vertex> notInSet(vector<Vertex>& X, const tbb::concurrent_set<Vertex>& Y) {
    vector<Vertex> Z;
    for (const auto& x : X) {
        if (!Y.contains(x)) {
            Z.push_back(x);
        }
    }
    return Z;
}


void enumerate(const BipartiteGraph &graph, unordered_set<Vertex> &L, unordered_set<Vertex> &R, vector<Vertex> &P,
               unordered_set<Vertex> &Q, bool imbea, map<vector<Vertex>, vector<Vertex> > &bicliques) {

    while (!P.empty()) {
        P = sortByCommonNeighbourhoodSize(P);
        Vertex x = P.back();
        P.pop_back();
        unordered_set<Vertex> Rprime = R;
        Rprime.insert(x);
        unordered_set<Vertex> Lprime = getNeighboursOf(graph, x, L);
        unordered_set<Vertex> Lstar;
        unordered_set<Vertex> C;

        if (imbea) {
            Lstar = difference(L, Lprime);
            C.insert(x);
        }

        vector<Vertex> Pprime;
        unordered_set<Vertex> Qprime;

        bool isMaximalBiclique = true;

        // for all the vertices considere to be added to R but then rejected
        for (auto v : Q) {
            unordered_set<Vertex> Nv = getNeighboursOf(graph, v, Lprime);
            if (Nv.size() == Lprime.size()) {
                isMaximalBiclique = false;
                break;
            } else if (!Nv.empty()) {
                Qprime.insert(v);
            }
        }

        // At this point Q is updated
        if (isMaximalBiclique) {
            // check for other candidate vertices in P
            for (auto v : P) {
                if (v == x) continue;

                // Get the neighbours of v in L'
                unordered_set<Vertex> Nv = getNeighboursOf(graph, v, Lprime);

                // if this happens, we know that all neighbours of v are in the L'
                // and we can add it to R, so R is updated to R'
                if (Nv.size() == Lprime.size()) {
                    Rprime.insert(v);

                    if (imbea) {
                        unordered_set<Vertex> S = getNeighboursOf(graph, v, Lstar);
                        if (S.empty())
                            C.insert(v);
                    }
                } else if (!Nv.empty()) {
                    Pprime.push_back(v);
                }
            }

            vector LprimeVector(Lprime.begin(), Lprime.end());
            vector RprimeVector(Rprime.begin(), Rprime.end());
            bicliques[LprimeVector] = RprimeVector;

            if (!Pprime.empty()) {
                enumerate(graph, Lprime, Rprime, Pprime, Qprime, imbea, bicliques);
            }
        }

        if (imbea) {
            Q = setUnion(Q, C);
            P = notInSet(P, C);
        }
    }
}

void iMBEA::run() {
    vector<Vertex> P;
    unordered_set<Vertex> Q;
    unordered_set<Vertex> L;
    unordered_set<Vertex> R;
    map<vector<Vertex>, vector<Vertex>> bicliques;

    for (const auto& [u, vs]: _graph._adjacencyList) {
        if (utils::startsWith(u.getId(), "U") || utils::startsWith(u.getId(), "u")) {
            L.insert(u);
        } else if (utils::startsWith(u.getId(), "P") || utils::startsWith(u.getId(), "p")) {
            P.push_back(u);
        }
    }
    enumerate(_graph, L, R, P, Q, true, bicliques);
    // utils::printBicliques(bicliques);
    cout << "Number of maximal bicliques: " << bicliques.size() << endl;
}



void enumeratePar(const BipartiteGraph& graph, tbb::concurrent_set<Vertex>& L, tbb::concurrent_set<Vertex>& R,
vector<Vertex>& P, tbb::concurrent_set<Vertex>& Q, bool imbea, tbb::concurrent_map<vector<Vertex>, vector<Vertex>>&
bicliques) {

    while (!P.empty()) {
        P = sortByCommonNeighbourhoodSize(P);
        Vertex x = P.back();
        P.pop_back();
        tbb::concurrent_set<Vertex> Rprime = R;
        Rprime.insert(x);
        tbb::concurrent_set<Vertex> Lprime = getNeighboursOf(graph, x, L);
        tbb::concurrent_set<Vertex> Lstar;
        tbb::concurrent_set<Vertex> C;

        if (imbea) {
            Lstar = difference(L, Lprime);
            C.insert(x);
        }

        vector<Vertex> Pprime;
        tbb::concurrent_set<Vertex> Qprime;

        bool isMaximalBiclique = true;

        // for all the vertices considere to be added to R but then rejected
        for (Vertex v : Q) {
            tbb::concurrent_set<Vertex> Nv = getNeighboursOf(graph, v, Lprime);
            if (Nv.size() == Lprime.size()) {
                isMaximalBiclique = false;
                break;
            } else if (!Nv.empty()) {
                Qprime.insert(v);
            }
        }

        // At this point Q is updated
        if (isMaximalBiclique) {
            // check for other candidate vertices in P
            for (auto v : P) {
                if (v == x) continue;

                // Get the neighbours of v in L'
                tbb::concurrent_set<Vertex> Nv = getNeighboursOf(graph, v, Lprime);

                // if this happens, we know that all neighbours of v are in the L'
                // and we can add it to R, so R is updated to R'
                if (Nv.size() == Lprime.size()) {
                    Rprime.insert(v);

                    if (imbea) {
                        tbb::concurrent_set<Vertex> S = getNeighboursOf(graph, v, Lstar);
                        if (S.empty())
                            C.insert(v);
                    }
                } else if (!Nv.empty()) {
                    Pprime.push_back(v);
                }
            }

            vector LprimeVector(Lprime.begin(), Lprime.end());
            vector RprimeVector(Rprime.begin(), Rprime.end());
            bicliques[LprimeVector] = RprimeVector;

            if (!Pprime.empty()) {
                enumeratePar(graph, Lprime, Rprime, Pprime, Qprime, imbea, bicliques);
            }
        }

        if (imbea) {
            Q = setUnion(Q, C);
            P = notInSet(P, C);
        }
    }
}

void iMBEA::runPar() {
    vector<Vertex> P;
    tbb::concurrent_set<Vertex> Q;
    tbb::concurrent_set<Vertex> L;
    tbb::concurrent_set<Vertex> R;
    tbb::concurrent_map<vector<Vertex>, vector<Vertex>> bicliques;

    for (const auto& [u, vs]: _graph._adjacencyList) {
        if (utils::startsWith(u.getId(), "U") || utils::startsWith(u.getId(), "u")) {
            L.insert(u);
        } else if (utils::startsWith(u.getId(), "P") || utils::startsWith(u.getId(), "p")) {
            P.push_back(u);
        }
    }
    enumeratePar(_graph, L, R, P, Q, true, bicliques);
    // utils::printBicliques(bicliques);
}

iMBEA::~iMBEA() = default;



