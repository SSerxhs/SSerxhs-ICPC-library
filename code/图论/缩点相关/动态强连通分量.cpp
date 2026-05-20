
struct union_set
{
	vector<int> f;
	int n;
	union_set() { }
	union_set(int nn) :n(nn), f(nn + 1)
	{
		iota(all(f), 0);
	}
	int getf(int u) { return f[u] == u ? u : f[u] = getf(f[u]); }
	bool merge(int u, int v)
	{
		u = getf(u); v = getf(v);
		if (u == v) return 0;
		f[u] = v;
		return 1;
	}
	bool connected(int u, int v) { return getf(u) == getf(v); }
};
struct edge
{
	int u, v, t;
};
vector<vector<edge>> solve(int n, const auto &eg)//[0,n)
{
	int m = eg.size(), tp = -1, id = 0, fs = 0;
	vector<vector<edge>> res(m);
	vector e(n, vector<int>());
	vector<int> dfn(n, -1), low(n, -1), st(n), ed(n), blk(n), node;
	union_set s(n - 1);
	function<void(int)> dfs = [&](int u) {
		dfn[u] = low[u] = id++;
		ed[st[++tp] = u] = 1;
		for (int v : e[u]) if (dfn[v] != -1)
		{
			if (ed[v]) cmin(low[u], dfn[v]);
		}
		else dfs(v), cmin(low[u], low[v]);
		if (dfn[u] == low[u])
		{
			do
			{
				ed[st[tp]] = 0;
				blk[st[tp]] = fs;
			} while (st[tp--] != u);
			++fs;
		}
	};
	auto ztef = [&](auto ztef, int l, int r, const vector<edge> &q) {
		if (eg.size() == 0) return;
		if (l + 1 == r)
		{
			if (l < m)
			{
				res[l].insert(res[l].end(), all(q));
				for (auto [u, v, t] : q) s.merge(u, v);
			}
			return;
		}
		int m = (l + r) / 2;
		node.clear();
		for (auto [u, v, t] : q) if (t < m)
		{
			u = s.getf(u);
			v = s.getf(v);
			e[u].push_back(v);
			node.push_back(u);
			node.push_back(v);
		}
		else break;
		for (int u : node) if (dfn[u] == -1) dfs(u);
		vector<vector<edge>> g(2);
		for (auto [u, v, t] : q) g[t < m && blk[s.f[u]] == blk[s.f[v]]].push_back({u, v, t});
		for (int u : node)
		{
			e[u].clear();
			dfn[u] = low[u] = -1;
		}
		id = fs = 0;
		ztef(ztef, l, m, g[1]);
		ztef(ztef, m, r, g[0]);
	};
	ztef(ztef, 0, m + 1, eg);
	return res;
}
int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	cout << fixed << setprecision(15);
	int n, m, i, j;
	cin >> n >> m;
	vector<ull> x(n);
	cin >> x;
	vector<edge> edges(m);
	for (i = 0; i < m; i++)
	{
		auto &[u, v, t] = edges[i];
		cin >> u >> v;
		t = i;
	}
	auto event = solve(n, edges);
	union_set s(n - 1);
	ull ans = 0;
	for (auto e : event)
	{
		for (auto [u, v, t] : e)
		{
			u = s.getf(u);
			v = s.getf(v);
			if (u == v) continue;
			s.f[v] = u;
			(ans += x[u] * x[v]) %= p;
			(x[u] += x[v]) %= p;
		}
		cout << ans << '\n';
	}
}

