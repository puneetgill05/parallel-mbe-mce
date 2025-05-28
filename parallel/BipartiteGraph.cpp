//
// Created by puneet on 13/05/25.
//

#include "BipartiteGraph.h"

#include <fstream>
#include <iostream>
#include <sstream>

#include "Vertex.h"

BipartiteGraph::BipartiteGraph() {
    numberOfVertices = 0;
    numberOfEdges = 0;
}

BipartiteGraph::BipartiteGraph(map<string, vector<string>> inputGraph) {
    for (const auto& [u, neighbors] : inputGraph) {
        addVertex(Vertex{u});
        for (const auto& v : neighbors) {
            addVertex(Vertex{v});
            addEdge(u, v);
            cout << "Edge from " << u << " to " << v << std::endl;
        }
    }
    numberOfVertices = computeNumberOfVertices();
    numberOfEdges = computeNumberOfEdges();
}


void BipartiteGraph::addVertex(const Vertex &u) {
    if (!_adjacencyList.contains(u)) {
        _adjacencyList[u] = {};
        numberOfVertices += 1;
    }
}

void BipartiteGraph::addEdge(const Vertex &u, const Vertex &v) {
    const Vertex& u_vertex = u;
    const Vertex& v_vertex = v;
    if (_adjacencyList.contains(u_vertex)) {
        _adjacencyList[u_vertex].insert(v_vertex);
        numberOfEdges += 1;
    }
    if (_adjacencyList.contains(v_vertex)) {
        _adjacencyList[v_vertex].insert(u_vertex);
    }
}

int BipartiteGraph::getNumberOfVertices() const {
    return numberOfVertices;
}

int BipartiteGraph::getNumberOfEdges() const {
    return numberOfEdges;
}


int BipartiteGraph::computeNumberOfVertices() const {
    return _adjacencyList.size();
}

unordered_set<Vertex>& BipartiteGraph::neighbor(const Vertex &v) {
    return _adjacencyList[v];
}

tbb::concurrent_unordered_set<Vertex>& BipartiteGraph::neighbor_concurrent(Vertex v) {
    unordered_set<Vertex> neighbours = _adjacencyList[v];
    tbb::concurrent_unordered_set ret(neighbours.begin(), neighbours.end());
    return ret;
}

int BipartiteGraph::degreeOf(const Vertex& v) {
    return _adjacencyList[v].size();
}

int BipartiteGraph::computeNumberOfEdges() {
    int numberOfEdges = 0;
    for (const auto& [u, neighbours] : _adjacencyList) {
        numberOfEdges += neighbours.size();
    }
    return numberOfEdges / 2;
}

void BipartiteGraph::print() {
    for (const auto& [u, neighbors] : _adjacencyList) {
        cout << u.getId() << " : [ ";
        for (const auto& v : neighbors) {
            cout << v.getId() << " ";
        }
        cout << "]\n";
    }
}

void BipartiteGraph::readInBiEdgeList(const string &fname) {
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
                if(!_adjacencyList.contains(v_vertex)) {
                    numVR++;
                    addVertex(v_vertex);
                }
                if(!_adjacencyList.contains(u_vertex)) {
                    numVL++;
                    addVertex(u_vertex);
                }
                addEdge(u_vertex, v_vertex);
            }
        }
    }
}

BipartiteGraph::~BipartiteGraph() {
    // even if empty
}

