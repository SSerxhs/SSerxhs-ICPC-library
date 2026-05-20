template <class T, class W, class F>
struct solver
{
	int n;
	vector<T> f, g;
	vector<int> dep;
	solver(const vector<tuple<int, int, W>> &eg, vector<T> a, const F &merge, int rt = 0, T zero = T{ })
		:n(a.size()), f(a), g(n), dep(n)
	{
		assert(eg.size() == n - 1);
		vector<vector<pair<int, W>>> e(n);
		vector<W> lf(n);
		for (auto [u, v, w] : eg)
		{
			e[u].push_back({v, w});
			e[v].push_back({u, w});
		}
		{
			auto dfs = [&](auto &&dfs, int u)->void {
				for (auto [v, w] : e[u])
				{
					erase(e[v], pair{u, w});
					dep[v] = dep[u] + 1;
					lf[v] = w;
					dfs(dfs, v);
					merge(f[u], f[v], w);
				}
			};
			dfs(dfs, rt);
		}
		{
			auto dfs = [&](auto &&dfs, int u)->void {
				T s = a[u];
				if (u != rt) merge(s, g[u], lf[u]);
				for (auto [v, w] : e[u]) g[v] = s, merge(s, f[v], w);
				s = zero;
				reverse(all(e[u]));
				for (auto [v, w] : e[u]) merge(g[v], s, w), dfs(dfs, v), merge(s, f[v], w);
			};
			dfs(dfs, rt);
		}
	}
};
