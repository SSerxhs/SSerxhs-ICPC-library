#include "bits/stdc++.h"
using namespace std;
namespace StoerWagner
{
	const int N = 602;//点数
	typedef int T;//边权和
	T e[N][N], w[N];
	int ed[N], p[N], f[N];//f 仅输出方案用
	int getf(int u) { return f[u] == u ? u : f[u] = getf(f[u]); }
	template<class TT> pair<T, vector<int>> mincut(int n, const vector<tuple<int, int, TT>> &edges)//[1,n]，返回某一集合
	{
		vector<int> ans; ans.reserve(n);
		int i, j, m;
		T r;
		r = numeric_limits<T>::max();
		for (i = 1; i <= n; i++) memset(e[i] + 1, 0, n * sizeof e[0][0]);
		for (auto [u, v, w] : edges) e[u][v] += w, e[v][u] += w;
		fill_n(ed + 1, n, 0);
		iota(f + 1, f + n + 1, 1);
		for (m = n; m > 1; m--)
		{
			fill_n(w + 1, n, 0);
			for (i = 1; i <= n; i++) ed[i] &= 2;
			for (i = 1; i <= m; i++)
			{
				int x = 0;
				for (j = 1; j <= n; j++) if (!ed[j]) break; x = j;
				for (j++; j <= n; j++) if (!ed[j] * w[j] > w[x]) x = j;
				ed[p[i] = x] = 1;
				for (j = 1; j <= n; j++) w[j] += !ed[j] * e[x][j];
			}
			int s = p[m - 1], t = p[m];
			if (r > w[t])
			{
				r = w[t]; ans.clear();
				for (i = 1; i <= n; i++) if (getf(i) == getf(t)) ans.push_back(i);
			}
			for (i = 1; i <= n; i++) e[i][s] = e[s][i] += e[t][i];
			ed[t] = 2;
			f[getf(s)] = getf(t);
		}
		return {r, ans};
	}
}
int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	int n, m;
	cin >> n >> m;
	vector<tuple<int, int, int>> e(m);
	for (auto &[u, v, w] : e) cin >> u >> v >> w;
	auto [_, v] = StoerWagner::mincut(n, e);
	cout << _ << endl;
	static int ed[602];
	for (int x : v) ed[x] = 1;
	for (auto [u, v, w] : e) _ -= w * (ed[u] ^ ed[v]);
	assert(!_);
}

