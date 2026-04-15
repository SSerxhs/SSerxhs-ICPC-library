struct edge_bcc
{
	int n, id, tp, bcc_n;
	vector<vector<pair<int, int>>> e, cur_e;
	vector<vector<int>> bcc_node, bcc_edge;
	vector<int> dfn, low, st, blk, ct;
	edge_bcc(const vector<vector<pair<int, int>>> &e, int m) :
		n(e.size()), id(0), tp(0), bcc_n(0), e(e), dfn(n, -1), low(n, -1), st(n), blk(n), ct(m)
	{
		for (int i = 0; i < n; i++) for (auto [v, w] : e[i])
		{
			assert(v >= 0 && v < n);
			assert(w >= 0 && w < m);
		}
		for (int i = 0; i < n; i++) if (dfn[i] == -1) dfs(i, -1);
		cur_e.resize(bcc_n);
		for (int i = 0; i < n; i++) for (auto [v, w] : e[i]) if (ct[w]) cur_e[blk[i]].push_back({blk[v], w});
		else bcc_edge[blk[i]].push_back(w);
		vector<int> flg(m);
		for (auto &v : bcc_edge)
		{
			vector<int> t;
			for (int x : v) if (!exchange(flg[x], 1)) t.push_back(x);
			swap(t, v);
		}
	}
	void dfs(int u, int fw)
	{
		dfn[u] = low[u] = id++;
		st[tp++] = u;
		for (auto [v, w] : e[u]) if (w != fw)
		{
			if (dfn[v] == -1)
			{
				dfs(v, w);
				cmin(low[u], low[v]);
				ct[w] = (dfn[u] < low[v]);
			}
			else cmin(low[u], dfn[v]);
		}
		if (dfn[u] == low[u])
		{
			bcc_node.push_back({ });
			bcc_edge.push_back({ });
			do
			{
				bcc_node[bcc_n].push_back(st[--tp]);
				blk[st[tp]] = bcc_n;
			} while (st[tp] != u);
			++bcc_n;
		}
	}
};
int main()
{
	ios::sync_with_stdio(0); cin.tie(0);
	int n, m, i;
	cin >> n >> m;
	vector<vector<pair<int, int>>> e(n);
	for (i = 0; i < m; i++)
	{
		int u, v;
		cin >> u >> v;
		--u, --v;
		e[u].push_back({v, i});
		e[v].push_back({u, i});
	}
	edge_bcc s(e, m);
	cout << s.bcc_n << '\n';
	for (auto &v : s.bcc_node)
	{
		for (int &x : v) ++x;
		cout << v.size() << ' ' << v << '\n';
	}
}
