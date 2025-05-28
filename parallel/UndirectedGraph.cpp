//
// Created by puneet on 28/04/25.
//

#include "UndirectedGraph.h"

#include <fstream>
#include <iostream>
#include <set>
#include <utility>
#include <vector>
#include <tuple>
#include <sstream>


UndirectedGraph::UndirectedGraph() {
    numberOfVertices = 0;
    numberOfEdges = 0;
//     map<string, vector<string>> adj = {};
//     createGraph(adj);
//     numberOfVertices = computeNumberOfVertices();
//     numberOfEdges = computeNumberOfEdges();
}

UndirectedGraph::UndirectedGraph(map<string, vector<string>> inputGraph) {
    for (const auto& [u, neighbors] : inputGraph) {
        addVertex(Vertex{u});
        for (const auto& v : neighbors) {
            addVertex(Vertex{v});
            addEdge(u, v);
            // cout << "Edge from " << u << " to " << v << std::endl;
        }
    }
    numberOfVertices = computeNumberOfVertices();
    numberOfEdges = computeNumberOfEdges();
}

UndirectedGraph::UndirectedGraph(map<Vertex, vector<Vertex>> inputGraph) {
    for (const auto& [u, neighbors] : inputGraph) {
        addVertex(u);
        for (const auto& v : neighbors) {
            addVertex(v);
            addEdge(u, v);
            // cout << "Edge from " << u << " to " << v << std::endl;
        }
    }
    numberOfVertices = computeNumberOfVertices();
    numberOfEdges = computeNumberOfEdges();
}

UndirectedGraph UndirectedGraph::createGraph(map<string, vector<string>> inputGraph) {

    // return newGraph;
}

void UndirectedGraph::addVertex(Vertex u) {
    if (_adjacencyList.find(u) == _adjacencyList.end()) {
        _adjacencyList[u] = {};
        numberOfVertices += 1;
    }
}

int UndirectedGraph::getNumberOfVertices() {
    return numberOfVertices;
}

int UndirectedGraph::getNumberOfEdges() {
    return numberOfEdges;
}


int UndirectedGraph::computeNumberOfVertices() {
    return _adjacencyList.size();
}

unordered_set<Vertex>& UndirectedGraph::neighbor(Vertex v) {
    return _adjacencyList[v];
}

tbb::concurrent_unordered_set<Vertex>& UndirectedGraph::neighbor_concurrent(Vertex v) {
    unordered_set<Vertex> neighbours = _adjacencyList[v];
    tbb::concurrent_unordered_set ret(neighbours.begin(), neighbours.end());
    return ret;
}


int UndirectedGraph::degreeOf(Vertex &v) {
    return _adjacencyList[v].size();
}

int UndirectedGraph::computeNumberOfEdges() {
    int numberOfEdges = 0;
    for (const auto& [u, neighbours] : _adjacencyList) {
        numberOfEdges += neighbours.size();
    }
    return numberOfEdges / 2;
}

void UndirectedGraph::addEdge(const Vertex &u, const Vertex &v) {
    Vertex u_vertex = u;
    Vertex v_vertex = v;
    if (_adjacencyList.find(u_vertex) != _adjacencyList.end()) {
        _adjacencyList[u_vertex].insert(v_vertex);
        numberOfEdges += 1;
    }
    if (_adjacencyList.find(v_vertex) != _adjacencyList.end()) {
        _adjacencyList[v_vertex].insert(u_vertex);
    }
}

void UndirectedGraph::print() {
    for (const auto& [u, neighbors] : _adjacencyList) {
        cout << u.getId() << " : [ ";
        for (const auto& v : neighbors) {
            cout << v.getId() << " ";
        }
        cout << "]\n";
    }
}


void UndirectedGraph::readInBiEdgeList(const string &fname) {
    std::ifstream instream(fname.c_str());
    int numVL = 0;
    int numVR = 0;

    if (instream.good() && !instream.eof()) {
        while (true) {
            if (!instream.good() || instream.eof()) {
                break;
            }

            string line;
            std::getline(instream, line);
            stringstream strm(line);

            if (!line.empty() && strm.good() && !strm.eof()) {
                string u, v;
                strm >> u >> v;

                auto u_vertex = Vertex(u);
                auto v_vertex = Vertex(v);
                if(_adjacencyList.find(v_vertex) == _adjacencyList.end()) {
                    numVR++;
                    addVertex(v_vertex);
                }
                if(_adjacencyList.find(u_vertex) == _adjacencyList.end()) {
                    numVL++;
                    addVertex(u_vertex);
                }
                addEdge(u_vertex, v_vertex);
            }
        }
    }
}


UndirectedGraph::~UndirectedGraph() {
    // even if empty
}
