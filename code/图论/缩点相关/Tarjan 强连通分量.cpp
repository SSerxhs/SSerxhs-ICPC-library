struct scc
{
	int n, id, tp, cc_n;
	vector<vector<pair<int, int>>> e, cur_e;
	vector<vector<int>> cc_node;
	vector<int> dfn, low, st, ed, blk;
	void dfs(int u)
	{
		dfn[u] = low[u] = id++;
		st[tp++] = u; ed[u] = 1;
		for (auto [v, w] : e[u]) if (dfn[v] == -1)
		{
			dfs(v);
			cmin(low[u], low[v]);
		}
		else if (ed[v]) cmin(low[u], dfn[v]);
		if (low[u] == dfn[u])
		{
			cc_node.push_back({ });
			do
			{
				int v = st[--tp];
				ed[v] = 0;
				blk[v] = cc_n;
				cc_node[cc_n].push_back(v);
			} while (st[tp] != u);
			cc_n++;
		}
	}
	scc(const vector<vector<pair<int, int>>> &e) :n(e.size()), id(0), tp(0), cc_n(0),
		e(e), cur_e(n), dfn(n, -1), low(dfn), st(n), ed(n), blk(n)
	{
		for (int i = 0; i < n; i++) for (auto [v, w] : e[i]) assert(v >= 0 && v < n);
		for (int i = 0; i < n; i++) if (dfn[i] == -1) dfs(i);
		reverse(all(cc_node));
		for (int &x : blk) x = cc_n - x - 1;
		for (int u = 0; u < n; u++)
			for (auto [v, w] : e[u])
				if (blk[u] != blk[v])
					cur_e[blk[u]].push_back({blk[v], w});
		cur_e.resize(cc_n);
	}
};
vector<vector<int>> unique(vector<vector<pair<int, int>>> &e)
{
	int n = e.size(), i;
	vector<vector<int>> g(n);
	for (i = 0; i < n; i++)
	{
		for (auto [v, w] : e[i]) g[i].push_back(v);
		sort(all(g[i]));
		g[i].resize(unique(all(g[i])) - g[i].begin());
	}
	return g;
}

