vector<int> dom_tree(vector<vector<int>> e, int s)
{
	int n = e.size() - 1, i, id = 0;
	vector<vector<int>> buc(n + 1), ie(n + 1);
	vector<int> mn(n + 1), f(n + 1), sdom(n + 1, n + 1), idom(n + 1), dfn(n + 1), nfd(n + 2), pv(n + 1), ed(n + 1), cf(n + 1);
#define cmp(x,y) (sdom[x] < sdom[y] ? x : y)
	auto getf = [&](auto &&getf, int u) ->void {
		if (f[u] == u) return;
		getf(getf, f[u]);
		mn[u] = cmp(mn[u], mn[f[u]]);
		f[u] = f[f[u]];
	};
	for (i = 1; i <= n; i++) mn[i] = f[i] = i;
	{
		auto dfs = [&](auto &&dfs, int u) ->void {
			ed[u] = 1;
			for (int v : e[u]) if (!ed[v]) dfs(dfs, v);
		};
		dfs(dfs, s);
	}
	for (i = 1; i <= n; i++) if (ed[i]) erase_if(e[i], [&](int v) { return !ed[v]; });
	else e[i].clear();
	for (i = 1; i <= n; i++) for (int v : e[i]) ie[v].push_back(i);
	auto dfs = [&](auto &&dfs, int u) ->void {
		nfd[dfn[u] = ++id] = u;
		for (int v : e[u]) if (!dfn[v]) dfs(dfs, v), cf[v] = u;
	};
	dfs(dfs, s); dfn[0] = n + 1;
	for (i = id; i; i--)
	{
		int u = nfd[i], w = 0;
		for (int v : ie[u])
		{
			cmin(sdom[u], dfn[v]);
			if (dfn[v] > dfn[u])
			{
				getf(getf, v);
				w = cmp(w, mn[v]);
			}
		}
		cmin(sdom[u], sdom[w]);
		buc[nfd[sdom[u]]].push_back(u);
		for (int v : buc[u]) getf(getf, v), pv[v] = mn[v];
		for (int v : e[u]) if (cf[v] == u) f[v] = u, mn[v] = cmp(mn[v], mn[u]);
	}
	for (i = 1; i <= n; i++) idom[nfd[i]] = (sdom[pv[nfd[i]]] == sdom[nfd[i]]) ? nfd[sdom[nfd[i]]] : idom[pv[nfd[i]]];
	idom[s] = s;
	return idom;
#undef cmp
}
int main()
{
	int n, m, s;
	cin >> n >> m >> s; ++s;
	vector<vector<int>> e(n + 1);
	for (int i = 1; i <= m; i++)
	{
		int u, v;
		cin >> u >> v; ++u; ++v;
		e[u].push_back(v);
	}
	auto r = dom_tree(e, s);
	for (int i = 1; i <= n; i++) cout << r[i] - 1 << " \n"[i == n];
}

