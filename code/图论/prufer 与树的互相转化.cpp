vector<int> edges_to_prufer(const vector<pair<int, int>> &eg)//[1,n]，定根为 n
{
	int n = eg.size() + 1, i, j, k;
	vector<int> fir(n + 1), nxt(n * 2 + 1), e(n * 2 + 1), rd(n + 1);
	int cnt = 0;
	for (auto [u, v] : eg)
	{
		e[++cnt] = v; nxt[cnt] = fir[u]; fir[u] = cnt; ++rd[v];
		e[++cnt] = u; nxt[cnt] = fir[v]; fir[v] = cnt; ++rd[u];
	}
	for (i = 1; i <= n; i++) if (rd[i] == 1) break;
	int u = i;
	vector<int> r; r.reserve(n - 2);
	for (j = 1; j < n - 1; j++)
	{
		for (k = fir[u], u = rd[u] = 0; k; k = nxt[k]) if (rd[e[k]])
		{
			r.push_back(e[k]);
			if ((--rd[e[k]] == 1) && (e[k] < i)) u = e[k];
		}
		if (!u) { while (rd[i] != 1) ++i; u = i; }
	}
	return r;
}
vector<pair<int, int>> prufer_to_edges(const vector<int> &p)//[1,n]，定根为 n
{
	int n = p.size(), i, j, k;
	int m = n + 3;
	vector<int> cs(m);
	for (i = 0; i < n; i++) ++cs[p[i]];
	i = 0;
	while (cs[++i]);
	int u = i, v;
	vector<pair<int, int>> r;
	r.reserve(n + 1);
	for (j = 0; j < n; j++)
	{
		cs[u] = 1e9;
		r.push_back({u, v = p[j]});
		if ((--cs[v] == 0) && (v < i)) u = v;
		if (v != u) { while (cs[i]) ++i; u = i; }
	}
	r.push_back({u, n + 2});
	return r;
}

