//
// Created by puneet on 03/05/25.
//

#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <concepts>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <oneapi/tbb/concurrent_set.h>
#include <tbb/concurrent_unordered_set.h>
#include <tbb/concurrent_map.h>
#include <tbb/parallel_for.h>


#include "Vertex.h"
#include "abseil-cpp/absl/container/flat_hash_set.h"

template <typename C>
concept Container = requires(C c) {
    typename C::value_type;
    c.begin();
    c.end();
};



class utils {
    static int size;
    using Edge = std::tuple<Vertex, Vertex>;
    struct EdgeHash {
        std::size_t operator()(const std::tuple<Vertex, Vertex>& t) const {
            auto h1 = std::hash<Vertex>()(std::get<0>(t));
            auto h2 = std::hash<Vertex>()(std::get<1>(t));
            return h1 ^ (h2 << 1); // Combine hashes
        }
    };
public:

    static void unordered_intersect(tbb::concurrent_unordered_set<Vertex> &A,
        unordered_set<Vertex> &B, tbb::concurrent_unordered_set<Vertex> *result) {
        if (static_cast<int>(A.size()) > static_cast<int>(B.size())) {
            for (const Vertex& i : B) {
                if (A.contains(i)){
                    result->insert(i);
                }
            }
        } else {
            for (const Vertex& i : A) {
                if (B.contains(i)){
                    result->insert(i);
                }
            }
        }
    }

    static void unordered_intersect(unordered_set<Vertex> &A,
    tbb::concurrent_unordered_set<Vertex> &B, tbb::concurrent_unordered_set<Vertex> *result) {
        if (static_cast<int>(A.size()) > static_cast<int>(B.size())) {
            for (const Vertex& i : B) {
                if (A.contains(i)){
                    result->insert(i);
                }
            }
        } else {
            for (const Vertex& i : A) {
                if (B.contains(i)){
                    result->insert(i);
                }
            }
        }
    }


    static void unordered_intersect(const unordered_set<Vertex> &A, const unordered_set<Vertex> &B, unordered_set<Vertex> *result) {
        if (static_cast<int>(A.size()) > static_cast<int>(B.size())) {
            for (const Vertex& i : B) {
                if (A.contains(i)){
                    result->insert(i);
                }
            }
        } else {
            for (const Vertex& i : A) {
                if (B.contains(i)){
                    result->insert(i);
                }
            }
        }
        //return *result;
    }


    static string printBiclique(vector<Vertex> &L, vector<Vertex> &R) {
        std::stringstream ss;
        ss << "[ ";
        for (auto const& l : L) {
            ss << l.getId() << " ";
        }
        ss << "] : [ ";
        for (auto const& r : R) {
            ss << r.getId() << " " ;
        }
        ss << "]" << endl;
        cout << ss.str();
        return ss.str();
    }

    static string printBicliques(tbb::concurrent_map<vector<Vertex>, vector<Vertex>> bicliques) {
        tbb::concurrent_map<vector<Vertex>, vector<Vertex>>::iterator it = bicliques.begin();
        std::stringstream ss;

        while (it != bicliques.end()) {
            vector<Vertex> L = it->first;
            vector<Vertex> R = it->second;
            string biclique_str = printBiclique(L, R);
            ss << biclique_str;
            it++;
        }
        return ss.str();
    }

    static void printBicliques(map<vector<Vertex>, vector<Vertex>> bicliques) {
        auto it = bicliques.begin();
        while (it != bicliques.end()) {
            vector<Vertex> L = it->first;
            vector<Vertex> R = it->second;
            printBiclique(L, R);
            it++;
        }
    }

    static bool startsWith(const std::string& str, const std::string& prefix) {
        return str.size() >= prefix.size() &&
               str.compare(0, prefix.size(), prefix) == 0;
    }


/** setDifference(S1, S2): S1 - S2
**/
    template<typename R, typename C1, typename C2>
    static R setDifference(C1& S1, C2& S2) {
        R Z;
        for (const auto& s1 : S1) {
            if (!S2.contains(s1)) {
                Z.insert(s1);
            }
        }
        return Z;
    }

    /** setDifference(S1, S2): S1 - S2
**/
    template<typename C>
    static C setDifference(C S1, C S2) {
        C Z;
        for (const auto& s1 : S1) {
            if (!S2.contains(s1)) {
                Z.insert(s1);
            }
        }
        return Z;
    }

    template<typename T>
    static unordered_set<T> setDifference(unordered_set<T>& S1, unordered_set<T>& S2) {
        unordered_set<T> Z;
        for (auto s1 : S1) {
            if (!S2.contains(s1)) {
                Z.insert(s1);
            }
        }
        return Z;
    }

    /** setDifference(S1, S2): S1 - S2
**/
    template <typename T>
    static tbb::concurrent_unordered_set<T> setDifferenceParallel(tbb::concurrent_unordered_set<T> S1, tbb::concurrent_unordered_set<T> S2) {
        tbb::concurrent_unordered_set<T> Z;
        // parallel_for_each(S1.begin(), S1.end(), [&](Vertex s1){
        for (auto s1 : S1) {
            if (!S2.contains(s1)) {
                Z.insert(s1);
            }
        }
        // });
        return Z;
    }

    /** setRemove(S, x): S - {x}
**/
    template<typename C, typename T>
    static C setRemove(C S, T x) {
        C Z;
        for (const auto s : S) {
            if (s != x) {
                Z.push_back(s);
            }
        }
        return Z;
    }

    // template<typename T>
    // static unordered_set<T> setRemove(unordered_set<T> S, T x) {
    //         unordered_set<T> Z;
    //         for (const auto s : S) {
    //             if (s != x) {
    //                 Z.insert(s);
    //             }
    //         }
    //         return Z;
    //     }

    /** setUnion(S1, S2): S1 + S2
    **/
    // template<typename R, typename C1, typename C2>
    // static R setUnion(C1 S1, C2 S2) {
    //     R Z;
    //     for (const auto s1 : S1) {
    //             Z.insert(s1);
    //     }
    //     for (const auto s2 : S2) {
    //         Z.insert(s2);
    //     }
    //     return Z;
    // }

    /** setUnion(S1, S2): S1 + S2
**/
    template<typename T>
    static unordered_set<T> setUnion(unordered_set<T>& S1, unordered_set<T>& S2) {
        unordered_set<T> Z;
        for (const auto s1 : S1) {
            Z.insert(s1);
        }
        for (const auto s2 : S2) {
            Z.insert(s2);
        }
        return Z;
    }

/** setUnion(S1, S2): S1 + S2
**/
    template<typename T>
    static tbb::concurrent_unordered_set<T> setUnionParallel(tbb::concurrent_unordered_set<T> S1, tbb::concurrent_unordered_set<T> S2) {
        tbb::concurrent_unordered_set<T> Z;
        // parallel_for_each(S1.begin(), S1.end(), [&](Vertex s1){
        for (auto s1 : S1) {
            Z.insert(s1);
        }
        // });
        // parallel_for_each(S2.begin(), S2.end(), [&](Vertex s2){
        for (auto s2 : S2) {
            Z.insert(s2);
        }
        // });
        return Z;
    }

/** setIntersect(S1, S2): S1 & S2
**/
    template<typename C>
    static C setIntersect(C& S1, C& S2) {
        const C& small = (S1.size() < S2.size()) ? S1 : S2;
        const C& large = (S1.size() < S2.size()) ? S2 : S1;

        C result;
        result.reserve(std::min(S1.size(), S2.size()));  // Reserve space

        for (auto x : small) {
            if (large.find(x) != large.end()) {
                result.insert(x);
            }
        }
        return result;
    }

    template<typename T>
    static tbb::concurrent_unordered_set<T> setIntersectParallel(tbb::concurrent_unordered_set<T>& S1, tbb::concurrent_unordered_set<T>& S2) {
        tbb::concurrent_unordered_set<T> Z;
        if (S1.size() < S2.size()) {
            // parallel_for_each(S1.begin(), S1.end(), [&](Vertex s1){
            for (auto s1 : S1) {
                if (S2.contains(s1)) {
                    Z.insert(s1);
                }
            }
            // });
        } else {
            // parallel_for_each(S2.begin(), S2.end(), [&](Vertex s2){
            for (auto s2 : S2) {
                if (S1.contains(s2)) {
                    Z.insert(s2);
                }
            }
            // });
        }
        return Z;
    }

    template<typename T>
    static unordered_set<T> setIntersect(unordered_set<T>& S1, unordered_set<T>& S2) {
        const std::unordered_set<T>& small = (S1.size() < S2.size()) ? S1 : S2;
        const std::unordered_set<T>& large = (S1.size() < S2.size()) ? S2 : S1;

        unordered_set<T> result;
        result.reserve(std::min(S1.size(), S2.size()));  // Reserve space

        for (T x : small) {
            if (large.find(x) != large.end()) {
                result.insert(x);
            }
        }
        return result;
    }

    template<typename T>
    static absl::flat_hash_set<T> setIntersect(absl::flat_hash_set<T>& S1, absl::flat_hash_set<T>& S2) {
        const absl::flat_hash_set<T>& small = (S1.size() < S2.size()) ? S1 : S2;
        const absl::flat_hash_set<T>& large = (S1.size() < S2.size()) ? S2 : S1;

        absl::flat_hash_set<T> result;
        result.reserve(std::min(S1.size(), S2.size()));  // Reserve space

        for (T x : small) {
            if (large.find(x) != large.end()) {
                result.insert(x);
            }
        }
        return result;
    }


    // template<typename T>
    // static unordered_set<T> setIntersect(tbb::concurrent_unordered_set<T>& S1, unordered_set<T>& S2) {
    //     const std::unordered_set<T>& small = (S1.size() < S2.size()) ? S1 : S2;
    //     const std::unordered_set<T>& large = (S1.size() < S2.size()) ? S2 : S1;
    //
    //     unordered_set<T> result;
    //     for (const T& x : small) {
    //         if (large.contains(x)) {
    //             result.insert(x);
    //         }
    //     }
    //     return result;
    // }



    /** setAdd(S1, x): S1 + {x}
**/
    template<typename C, typename T>
    static C setAdd(C& S, T x) {
        C Z;
        for (auto s : S) {
            Z.insert(s);
        }
        Z.insert(x);
        return Z;
    }

    /** isSubsetEq(S1, S2): Is S1 a subseteq of S3
**/
    // template<typename C1, typename C2>
    static bool isSubsetEq(unordered_set<Vertex>& S1, unordered_set<Vertex>& S2) {
        for (auto s : S1) {
            if (!S2.contains(s)) {
                return false;
            }
        }
        return true;
    }

    /** isSubsetEq(S1, S2): Is S1 a subseteq of S3
**/
    template<typename T>
    static bool isSubsetEq(T& S1, T& S2) {
        for (const auto& s : S1) {
            if (!S2.contains(s)) {
                return false;
            }
        }
        return true;
    }

    static void write_to_file(const string &filename, const string &content) {
        ofstream file(filename);  // Open file for writing
        if (file.is_open()) {
            file << content;                  // Write string to file
            file.close();                  // Close the file
        } else {
            std::cerr << "Unable to open file.\n";
        }
    }



    static unordered_map<Vertex, vector<Vertex>> readup(const string &upfilename) {
        ifstream infile(upfilename);
        unordered_map<Vertex, vector<Vertex>> up;


        string line;
        while (std::getline(infile, line)) {
            istringstream linestream(line);
            string u, values_str;

            if (std::getline(linestream, u, ':')) {
                Vertex u_vertex(u);
                std::getline(linestream, values_str);

                // Clean leading/trailing whitespace
                values_str.erase(0, values_str.find_first_not_of(" \t"));
                values_str.erase(values_str.find_last_not_of(" \t") + 1);

                // Expect value_part like: [item1,item2,...]
                if (values_str.front() == '[' && values_str.back() == ']') {
                    values_str = values_str.substr(1, values_str.size() - 2);  // strip brackets

                    std::vector<Vertex> permissions;
                    std::stringstream ss(values_str);
                    std::string item;

                    while (std::getline(ss, item, ',')) {
                        item.erase(0, item.find_first_not_of(" \t"));
                        item.erase(item.find_last_not_of(" \t") + 1);
                        Vertex item_vertex(item);
                        permissions.push_back(item_vertex);
                    }
                    up[u_vertex] = permissions;
                }
            }
        }
        infile.close();
        return up;
    }

    static bool hasbeenremoved(tuple<Vertex, Vertex> e, map<tuple<Vertex, Vertex>, tuple<Vertex, Vertex, int>> em) {
        for (auto it = em.begin(); it != em.end(); ++it) {
            if (it->first == e) {
                return true;
            }
        }
        return false;
    }


    static set<tuple<Vertex, Vertex>> getNeighbouringEdges(tuple<Vertex, Vertex> e, map<tuple<Vertex, Vertex>,
    tuple<Vertex, Vertex, int>> em, unordered_map<Vertex, vector<Vertex>> up,
        unordered_map<Vertex, vector<Vertex>> pu) {
        set<tuple<Vertex, Vertex>> neighs;
        Vertex u = get<0>(e);
        Vertex p = get<1>(e);

        set<Vertex> uprimes;
        for (const auto& u_prime : pu[p]) {
            if (u_prime == u) continue;
            uprimes.insert(u_prime);
            tuple f(u_prime, p);

            if (!hasbeenremoved(f, em)) {
                neighs.insert(f);
            }
        }

        set<Vertex> pprimes;
        for (const auto& p_prime : up[u]) {
            if (p_prime == p) continue;
            pprimes.insert(p_prime);
            tuple f(u, p_prime);

            if (!hasbeenremoved(f, em)) {
                neighs.insert(f);
            }
        }

        for (const auto& u_prime : uprimes) {
            for (const auto& p_prime : pprimes) {
                vector u_prime_neighs = up[u_prime];
                bool p_prime_is_neigh = std::find(u_prime_neighs.begin(), u_prime_neighs.end(), p_prime) != u_prime_neighs.end();
                if (!p_prime_is_neigh ) continue;
                tuple f(u_prime, p_prime);

                if (!hasbeenremoved(f, em) && isNeighbour(e, f, up)) {
                    neighs.insert(f);
                }
            }
        }
        return neighs;
    }

    static bool isNeighbour(tuple<Vertex, Vertex> e, tuple<Vertex, Vertex> f, unordered_map<Vertex, vector<Vertex>> up) {
        Vertex u = get<0>(e);
        Vertex p = get<1>(e);
        Vertex u_prime = get<0>(f);
        Vertex p_prime = get<1>(f);

        bool p_neighbour_with_u_prime = std::find(up.at(u_prime).begin(), up.at(u_prime).end(), p) != up.at(u_prime).end();
        bool p_prime_neighbour_with_u = std::find(up.at(u).begin(), up.at(u).end(), p_prime) != up.at(u).end();
        return p_neighbour_with_u_prime && p_prime_neighbour_with_u;
    }

    static unordered_map<Vertex, vector<Vertex>> uptopu(unordered_map<Vertex, vector<Vertex>> up) {
        unordered_map<Vertex, vector<Vertex>> pu;
        for (auto & it : up) {
            Vertex u = it.first;
            vector<Vertex> permissions = it.second;
            for (auto& p : permissions) {
                pu[p].push_back(u);
            }
        }
        return pu;
    }


    static set<tuple<Vertex, Vertex>> getedgeset(map<tuple<Vertex, Vertex>, tuple<Vertex, Vertex, int>> em,
    unordered_map<Vertex, vector<Vertex>> up) {
        // set of all edges
        set<tuple<Vertex, Vertex>> edgeset{};
        for (auto it = up.begin(); it != up.end(); ++it) {
            for (auto const p : it->second) {
                tuple e(it->first, p);

                if (hasbeenremoved(e, em)) {
                    continue;
                }
                edgeset.insert(e);
            }
        }
        return edgeset;
    }


    static string strip(string s) {
        size_t start = s.find_first_not_of(" \t\n\r");
        size_t end = s.find_last_not_of(" \t\n\r");
        return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
    }

    static string remove_quotes(string s) {
        if (!s.empty() && s.front() == '\'' && s.back() == '\'') {
            return s.substr(1, s.size() - 2);
        }
        return s;
    }

    static string remove_comma(string s) {
        std::erase(s, ',');
        return s;
    }

    static map<Edge, tuple<Vertex, Vertex, int>> readem(string emfile) {
        std::ifstream infile(emfile);
        std::string line;

        map<Edge, tuple<Vertex, Vertex, int>> ret;

        while (std::getline(infile, line)) {
            size_t colon_pos = line.find(':');
            if (colon_pos == std::string::npos) continue;

            std::string left = line.substr(0, colon_pos);
            std::string right = line.substr(colon_pos + 1);

            // Remove outer parentheses
            left = left.substr(1, left.size() - 2);
            right = right.substr(1, right.size() - 2);

            // Parse left side: ('u61', 'p163')
            std::stringstream lss(left);
            std::string token1, token2;
            std::getline(lss, token1, ',');
            std::getline(lss, token2, ',');

            token1 = remove_quotes(strip(token1));
            token2 = remove_quotes(strip(token2));
            std::tuple key = std::make_tuple(Vertex(token1), Vertex(token2));

            // Parse right side: ('u61', 'p162', 710)
            std::stringstream rss(right);
            std::string t1, t2, t3;
            std::getline(rss, t1, ',');
            std::getline(rss, t2, ',');
            std::getline(rss, t3);

            t1 = remove_quotes(strip(t1));
            t2 = remove_quotes(strip(t2));
            int weight = std::stoi(strip(t3));
            Vertex vertex_t1(t1);
            Vertex vertex_t2(t2);
            std::tuple value = std::make_tuple(vertex_t1, vertex_t2, weight);

            ret.insert({key, value});
        }
        return ret;
    }




};

#endif //UTILS_H
