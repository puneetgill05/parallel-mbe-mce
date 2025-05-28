//
// Created by puneet on 03/05/25.
//

#include "ParLmbc.h"

#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_unordered_set.h>
#include <tbb/concurrent_vector.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <iostream>
#include <string>
#include <set>
#include <sstream>
#include <utility>


// ParLmbc::ParLmbc(UndirectedGraph &g) {
// 	_ugraph = g;
// 	_numberOfMaximalBicliques = 0;
// 	_min_size_threshold = 1;
// }

ParLmbc::ParLmbc(BipartiteGraph &g) {
    _graph = g;
	_numberOfMaximalBicliques = 0;
	_min_size_threshold = 1;
}

long ParLmbc::getCount(){
    return _numberOfMaximalBicliques;
}

bool inBicliques(map<vector<Vertex>, vector<Vertex>>& bicliques, vector<Vertex>L) {
	for (const auto&[fst, snd] : bicliques) {
		unordered_set x_first(fst.begin(), fst.end());
		unordered_set L_set(L.begin(), L.end());
		if (x_first == L_set) {
			return true;
		}
	}
	return false;
}

bool inBicliques(tbb::concurrent_map<vector<Vertex>, vector<Vertex>>& bicliques, vector<Vertex>L) {
	for (const auto&[fst, snd] : bicliques) {
		unordered_set x_first(fst.begin(), fst.end());
		unordered_set L_set(L.begin(), L.end());
		if (x_first == L_set) {
			return true;
		}
	}
	return false;
}

set<Vertex> getEdgesfromBiclique(unordered_set<Vertex>& L, unordered_set<Vertex>& R) {
	set<Vertex> bicliqueEdges;
	for (auto const& l : L) {
		bicliqueEdges.insert(l);
		for (auto const& r : R) {
			bicliqueEdges.insert(r);
		}
	}
	return bicliqueEdges;
}


struct VertexComparator {
	unordered_set<Vertex> &_mygammaX;
	// UndirectedGraph& _mygraph;
	BipartiteGraph& _mygraph;
	// VertexComparator(unordered_set<Vertex>& mygammaX, UndirectedGraph ug) : _mygammaX(mygammaX),
	VertexComparator(unordered_set<Vertex>& mygammaX, BipartiteGraph& g) : _mygammaX(mygammaX),
	_mygraph(g) {}

	bool operator()(const Vertex& a, const Vertex& b) const {
		unordered_set<Vertex> intersect_a;
		utils::unordered_intersect(_mygammaX, _mygraph.neighbor(a), &intersect_a);
		int size_a = intersect_a.size();

		unordered_set<Vertex> intersect_b;
		utils::unordered_intersect(_mygammaX, _mygraph.neighbor(b), &intersect_b);
		int size_b = intersect_b.size();
		return size_a < size_b;
	}
};


string ParLmbc::runParMBE(string outfile, int nthreads){
    tbb::global_control c(tbb::global_control::max_allowed_parallelism, nthreads);
    _outfile = outfile;
	tbb::concurrent_map<vector<Vertex>, vector<Vertex>> bicliques;

    tbb::parallel_for_each(_graph._adjacencyList.begin(), _graph._adjacencyList.end(), [&](pair<Vertex,
    unordered_set<Vertex> > p){
	// for (auto const& p : _graph._adjacencyList) {
		Vertex v = p.first;
		if (utils::startsWith(v.getId(), "U") || utils::startsWith(v.getId(), "u")) {
			tbb::concurrent_unordered_set<Vertex> X;
			tbb::concurrent_unordered_set<Vertex> tailX;
			tbb::concurrent_unordered_set<Vertex> gammaX;

			// tailX.insert(v);
			X.insert(v);

			tbb::parallel_for_each(p.second.begin(), p.second.end(), [&](Vertex w){
			// for (auto const& w : p.second) {
				gammaX.insert(w);
				for(auto y : _graph.neighbor(w)){
					if(_graph.degreeOf(y) > _graph.degreeOf(v)){
						tailX.insert(y);
					}
					if(_graph.degreeOf(y) == _graph.degreeOf(v)){
						if(y.getId().compare(v.getId()) > 0)
							tailX.insert(y);
					}
				}
				});
			// }
			// cout << "X:";
			// for (auto x : X) {
			// 	cout << " " << x.getId();
			// }
			// cout << endl;
			// cout << "tailX:";
			// for (auto x : tailX) {
			// 	cout << " " << x.getId();
			// }
			// cout << endl;
			// cout << "gammaX:";
			// for (auto x : gammaX) {
			// 	cout << " " << x.getId();
			// }
			// cout << endl;
			parlmbc(X, gammaX, tailX, _min_size_threshold, bicliques);
		}
		});
	// }
	_numberOfMaximalBicliques = bicliques.size();
	string bicliques_str = utils::printBicliques(bicliques);
	cout << "# of maximal bicliques: " << bicliques.size() << endl;
	return bicliques_str;

}

void ParLmbc::runParMBESeq(string outfile) {
	_outfile = outfile;
	// vector<set<Vertex>> bicliqueSeen;
	map<vector<Vertex>, vector<Vertex>> bicliques;

	for (auto const& x : _graph._adjacencyList) {
		Vertex v = x.first;
		if (utils::startsWith(v.getId(), "U") || utils::startsWith(v.getId(), "u")) {
			unordered_set<Vertex> X;
			unordered_set<Vertex> tailX;
			unordered_set<Vertex> gammaX;

			X.insert(v);

			// for all neighbours of v
			for (auto const& w : x.second) {
				// get neighbour w of v
				gammaX.insert(w);

				for(const auto& y : _graph.neighbor(w)){
					if(_graph.degreeOf(y) > _graph.degreeOf(v)){
						tailX.insert(y);
					}
					if(_graph.degreeOf(y) == _graph.degreeOf(v)){
						if(y.getId().compare(v.getId()) > 0)
							tailX.insert(y);
					}
				}
			}
			// cout << "X:";
			// for (auto x : X) {
			// 	cout << " " << x.getId();
			// }
			// cout << endl;
			// cout << "tailX:";
			// for (auto x : tailX) {
			// 	cout << " " << x.getId();
			// }
			// cout << endl;
			// cout << "gammaX:";
			// for (auto x : gammaX) {
			// 	cout << " " << x.getId();
			// }
			// cout << endl;
			MineLMBCSeq(X, gammaX, tailX, _min_size_threshold, bicliques);
		}
	}

	_numberOfMaximalBicliques = bicliques.size();
	// utils::printBicliques(bicliques);
	cout << "# of maximal bicliques: " << bicliques.size() << endl;
}


void ParLmbc::parlmbc(tbb::concurrent_unordered_set<Vertex>& X, tbb::concurrent_unordered_set<Vertex>& gammaX,
tbb::concurrent_unordered_set<Vertex>& tailX, int ms, tbb::concurrent_map<vector<Vertex>, vector<Vertex>>& bicliques) {
	ConcurrentMap<Vertex, tbb::concurrent_unordered_set<Vertex>> M;
	tbb::concurrent_unordered_set<Vertex> verticesToRemove;

	if (X.size() == 1 && ms == 1) {
		auto gamma_x_iterator = gammaX.begin();
		const Vertex& vi = *gamma_x_iterator;
		tbb::concurrent_unordered_set Y(_graph.neighbor(vi).begin(), _graph.neighbor(vi).end());

		// tbb::parallel_for_each(gammaX.begin(), gammaX.end(), [&](Vertex v){
		for (auto const& v : gammaX) {
			tbb::concurrent_unordered_set<Vertex> temp;
			utils::unordered_intersect(_graph.neighbor(v), Y, &temp);
			Y = temp;
			if(Y.empty()) break;
		}
		vector L(Y.begin(), Y.end());
		vector R(gammaX.begin(), gammaX.end());
		if (!inBicliques(bicliques, L)) {
			bicliques[L]= std::move(R);
			// utils::printBiclique(L, R);
		}
	}

	tbb::parallel_for_each(tailX.begin(), tailX.end(), [&](Vertex v){
	// for (auto const& v : tailX) {
		tbb::concurrent_unordered_set<Vertex> intersect;
		utils::unordered_intersect(gammaX, _graph.neighbor(v), &intersect);
		ConcurrentMap<Vertex, tbb::concurrent_unordered_set<Vertex>>::accessor ac;
		M.insert(ac, v);
		ac->second = intersect;
		if(static_cast<int>(intersect.size()) < ms) {
			verticesToRemove.insert(v);
		}
		});
	// }
	for (auto const& v : verticesToRemove) {
		tailX.unsafe_erase(v);
	}

	if(static_cast<int>(X.size()) + static_cast<int>(tailX.size()) < ms)
		return;

	tbb::concurrent_vector sortedTailX(tailX.begin(), tailX.end());
	unordered_set mygammaX(gammaX.begin(), gammaX.end());
	unordered_set mytailX(tailX.begin(), tailX.end());
	// unordered_set mygammaX(gammaX.begin(), gammaX.end());
	sort(sortedTailX.begin(), sortedTailX.end(), VertexComparator(mytailX, _graph));

	parallel_for(tbb::blocked_range<int>(0, sortedTailX.size()), [&](tbb::blocked_range<int> r){
		for(int idx = r.begin(); idx < r.end(); idx++) {
		// for(int idx = 0; idx < sortedTailX.size(); idx++) {
			const Vertex& v = sortedTailX[idx];
			tbb::concurrent_unordered_set<Vertex> newtailX;
			for(int i=idx+1; i < static_cast<int>(sortedTailX.size()); i++){
				const Vertex& w = sortedTailX[i];
				newtailX.insert(w);
			}

			unordered_set set_newTailX(newtailX.begin(), newtailX.end());
			unordered_set set_sortedTailX(sortedTailX.begin(), sortedTailX.end());

			if(static_cast<int>(X.size()) + static_cast<int>(newtailX.size()) + 1 > ms) {
				tbb::concurrent_unordered_set<Vertex> Y;
				ConcurrentMap<Vertex, tbb::concurrent_unordered_set<Vertex>>::accessor ac;
				M.find(ac, v);

				tbb::concurrent_unordered_set<Vertex>::iterator gamma_x_union_v_iterator = (ac->second).begin();
				tbb::concurrent_unordered_set<Vertex>::iterator gamma_x_union_v_end_iterator = (ac->second).end();
				Vertex vertex = *gamma_x_union_v_iterator;
				Y.insert(_graph.neighbor(vertex).begin(), _graph.neighbor(vertex).end());
				gamma_x_union_v_iterator++;

				while(gamma_x_union_v_iterator != gamma_x_union_v_end_iterator){
					tbb::concurrent_unordered_set<Vertex> temp;

					Vertex vi = *gamma_x_union_v_iterator;
					utils::unordered_intersect(_graph.neighbor(vi), Y, &temp);
					Y = temp;
					gamma_x_union_v_iterator++;
					if(Y.empty()) break;
				}

				auto X_union_v = utils::setAdd<tbb::concurrent_unordered_set<Vertex>, Vertex>(X, v);
				auto Y_minus_X_union_v = utils::setDifference<tbb::concurrent_unordered_set<Vertex>, tbb::concurrent_unordered_set<Vertex>,
					tbb::concurrent_unordered_set<Vertex>> (Y, X_union_v);

				bool flag = utils::isSubsetEq<tbb::concurrent_unordered_set<Vertex>>(Y_minus_X_union_v, newtailX);
				if(flag) {
					vector L(Y.begin(), Y.end());
					if(static_cast<int>(Y.size()) >= ms  && !inBicliques(bicliques, L)) {
						vector R((ac->second).begin(), (ac->second).end());
						// utils::printBiclique(L, R);

						bicliques[L]= std::move(R);
					}
					tbb::concurrent_unordered_set<Vertex> updatedtailX_minus_Y;
					tbb::parallel_for_each(newtailX.begin(), newtailX.end(), [&](const Vertex &y){
					// for (auto const& y : newtailX) {
						if(!Y.contains(y)) updatedtailX_minus_Y.insert(y);
					// }
					});
					parlmbc(Y, ac->second, updatedtailX_minus_Y, ms, bicliques);
				}
			}
		}
		});
}


void ParLmbc::MineLMBCSeq(unordered_set<Vertex>& X, unordered_set<Vertex>& gammaX, unordered_set<Vertex>& tailX, int
                          ms, map<vector<Vertex>, vector<Vertex>>&
                          bicliques) {
	unordered_map<Vertex, unordered_set<Vertex>> M;
	unordered_set<Vertex> verticesToRemove;

	if (X.size() == 1 && ms == 1) {
		auto gamma_x_iterator = gammaX.begin();
		const Vertex& vi = *gamma_x_iterator;
		unordered_set<Vertex> Y = _graph.neighbor(vi);

		for (auto const& v : gammaX) {
			unordered_set<Vertex> temp;
			utils::unordered_intersect(_graph.neighbor(v), Y, &temp);
			Y = temp;
			if(Y.empty()) break;
		}
		vector L(Y.begin(), Y.end());
		vector R(gammaX.begin(), gammaX.end());
		if (!inBicliques(bicliques, L)) {
			bicliques[L] = R;
			_numberOfMaximalBicliques.fetch_add(1);
			// utils::printBiclique(L, R);
		}

	}

	for (auto const& v : tailX) {
		unordered_set<Vertex> intersect;
		utils::unordered_intersect(gammaX, _graph.neighbor(v), &intersect);
		M[v] = intersect;
		if(M[v].size() < ms){
			verticesToRemove.insert(v);
		}
	}
	// tailX - v
	for (auto const& v : verticesToRemove) {
		tailX.erase(v);
	}

    if(static_cast<int>(X.size()) + static_cast<int>(tailX.size()) < ms) {
	    return;
    }

	vector sorted_tailX(tailX.begin(), tailX.end());
	std::sort(sorted_tailX.begin(), sorted_tailX.end(), VertexComparator(tailX, _graph));
	unordered_set<Vertex> toRemove;
	vector<Vertex> updatedTailX;

	for (const auto& v : sorted_tailX){
		if (toRemove.contains(v)) continue;
		toRemove.insert(v);

		updatedTailX = utils::setRemove<vector<Vertex>>(sorted_tailX, v);
		unordered_set updatedTailXSet(updatedTailX.begin(), updatedTailX.end());

		int sortedTailXSizeAfterRemoving = updatedTailXSet.size();
		if(static_cast<int>(X.size()) + sortedTailXSizeAfterRemoving + 1 > ms) {
			unordered_set<Vertex> Y;
			auto gamma_x_union_v_iterator = M[v].begin();
			auto gamma_x_union_v_end_iterator = M[v].end();

			Vertex vertex = *gamma_x_union_v_iterator;
			Y = _graph.neighbor(vertex);
			gamma_x_union_v_iterator++;

			while(gamma_x_union_v_iterator != gamma_x_union_v_end_iterator){
				unordered_set<Vertex> temp;
				Vertex vi = *gamma_x_union_v_iterator;
				utils::unordered_intersect(_graph.neighbor(vi), Y, &temp);
				Y = temp;
				gamma_x_union_v_iterator++;
				if (Y.empty()) break;
			}

			vector L(Y.begin(), Y.end());

			auto X_union_v = utils::setAdd<unordered_set<Vertex>, Vertex>(X, v);
			auto Y_minus_X_union_v = utils::setDifference<unordered_set<Vertex>, unordered_set<Vertex>,
			unordered_set<Vertex>> (Y, X_union_v);

			// bool flag = includes(updatedTailX.begin(), updatedTailX.end(), Y_minus_X_union_v.begin(), Y_minus_X_union_v.end());
			bool flag = utils::isSubsetEq(Y_minus_X_union_v, updatedTailXSet);
			if (flag) {
				if(static_cast<int>(Y.size()) >= ms && !inBicliques(bicliques, L)) {
					vector R(M[v].begin(), M[v].end());
					bicliques[L] = R;
					_numberOfMaximalBicliques.fetch_add(1);
					// utils::printBiclique(L, R);
				}
			auto tailX_minus_Y = utils::setDifference<unordered_set<Vertex>>(updatedTailXSet, Y);
			MineLMBCSeq(Y, M[v], tailX_minus_Y, ms, bicliques);
			}
        }
	}
}




ParLmbc::~ParLmbc() {
    // TODO Auto-generated destructor stub
}