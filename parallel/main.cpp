#include <iostream>
#include <map>
#include <omp.h>
#include <bits/fs_fwd.h>
#include <bits/fs_path.h>
#include <tbb/parallel_for.h>
#include <tbb/version.h>

#include "CliqueEnum.h"
#include "GMBE.h"
#include "iMBEA.h"
#include "ParLmbc.h"
#include "UndirectedGraph.h"

using namespace std;

// TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
int main_biclique(int argc, char** argv) {
    BipartiteGraph graph;
    graph.readInBiEdgeList(argv[1]);
    // graph.print();
    std::cout << "# vertices: " << graph.getNumberOfVertices() << std::endl;
    std::cout << "# edges: " << graph.getNumberOfEdges() << std::endl;

    ParLmbc parLmbc(graph);

    auto start = std::chrono::high_resolution_clock::now();

    cout << " parmbe" << endl;
    string bilciques_str = parLmbc.runParMBE("", 12);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << endl;
    utils::write_to_file("bicliques.txt", bilciques_str);


    // cout << " parmbe seq" << endl;
    // parLmbc.runParMBESeq("");


    // start = std::chrono::high_resolution_clock::now();
    // cout << "IMBEA" << endl;
    // iMBEA imbea(graph);
    // imbea.run();
    // end = std::chrono::high_resolution_clock::now();
    // elapsed = end - start;
    // std::cout << "Elapsed time: " << elapsed.count() << " seconds" << endl;

    return 0;
}

int main(int argc, char** argv) {
    std::filesystem::path cwd = std::filesystem::current_path();
    string cwd_str = cwd.string();
    cout << "File path: " << cwd_str << endl;
    map<Vertex, vector<Vertex>> up = utils::readup(cwd_str + "/../small_02-mapped.txt");
    map<Vertex, vector<Vertex>> up_undirected = utils::readup(cwd_str + "/../small_02-undirected.txt");
    map<tuple<Vertex, Vertex>, tuple<Vertex, Vertex, int>> em = utils::readem
    ( cwd_str + "/../small_02-em.txt");

     set<tuple<Vertex, Vertex>> edgeset = utils::getedgeset(em, up);

    set<tuple<Vertex, Vertex>> allEdges;

    for (auto it : up) {
        auto u = it.first;
        for (auto p : it.second) {
            allEdges.insert(make_tuple(u, p));
        }
    }

    vector<tuple<Vertex, Vertex>> edgesRemoved;
    set_difference(allEdges.begin(), allEdges.end(), edgeset.begin(),
    edgeset.end(), std::back_insert_iterator(edgesRemoved));

    UndirectedGraph graph(up_undirected);

    // graph.readInBiEdgeList(argv[1]);
    // graph.print();
    std::cout << "# vertices: " << graph.getNumberOfVertices() << std::endl;
    std::cout << "# edges: " << graph.getNumberOfEdges() << std::endl;

    CliqueEnum cliqueEnum(graph);

    cout << " CliqueEnum: TTT" << endl;
    set<Vertex> K;
    set<Vertex> cand;
    set<Vertex> fini;

    // for (auto const& v : graph._adjacencyList) {
    for (auto const& e : allEdges) {
        Vertex e_vertex("(" + get<0>(e).getId() + " " + get<1>(e).getId() + ")");
        cand.insert(e_vertex);
        // cand.insert(v.first);
    }
    // for (auto const& v : graph._adjacencyList) {
    for (auto const& e :edgesRemoved) {
        Vertex e_vertex("(" + get<0>(e).getId() + " " + get<1>(e).getId() + ")");
        fini.insert(e_vertex);

        // if (cand.find(v.first) == cand.end()) {
        //     fini.insert(v.first);
        // }
    }

    tbb::concurrent_set<Vertex> K_par;
    tbb::concurrent_set<Vertex> cand_par;
    tbb::concurrent_set<Vertex> fini_par;

    // for (auto const& v : graph._adjacencyList) {
    for (auto const& e : allEdges) {
        Vertex e_vertex("(" + get<0>(e).getId() + " " + get<1>(e).getId() + ")");
        cand_par.insert(e_vertex);
        // cand_par.insert(v.first);
    }

    // for (auto const& v : graph._adjacencyList) {
    for (auto const& e : edgesRemoved) {
        Vertex e_vertex("(" + get<0>(e).getId() + " " + get<1>(e).getId() + ")");
        fini_par.insert(e_vertex);

    }

    auto start = std::chrono::high_resolution_clock::now();

    // cliqueEnum.runParTTT(K_par, cand_par, fini_par, 12);
    cliqueEnum.runTTT(K, cand, fini);


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << endl;

    // GMBE gmbe;
    // gmbe.run();


    return 0;
}