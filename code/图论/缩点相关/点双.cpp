struct node_bcc
{
	int n, id, tp, bcc_n;
	vector<vector<pair<int, int>>> e;
	vector<vector<int>> bcc_node, bcc_edge;
	vector<int> dfn, low, st, ed, blk, ct;
	node_bcc(const vector<vector<pair<int, int>>> &e, int m) :
		n(e.size()), id(0), tp(0), bcc_n(0), e(e), dfn(n, -1), low(n, -1), st(m), ed(m), blk(m), ct(n)
	{
		for (int i = 0; i < n; i++) for (auto [v, w] : e[i])
		{
			assert(v >= 0 && v < n);
			assert(w >= 0 && w < m);
		}
		for (int i = 0; i < n; i++) if (dfn[i] == -1) dfs(i, 1);
		bcc_node.resize(bcc_n);
		for (int i = 0; i < n; i++) for (auto [v, w] : e[i]) bcc_node[blk[w]].push_back(i);
		vector<int> flg(n);
		for (auto &v : bcc_node)
		{
			vector<int> t;
			for (int x : v) if (!exchange(flg[x], 1)) t.push_back(x);
			swap(t, v);
			for (int x : v) flg[x] = 0;
		}
		for (int i = 0; i < n; i++) if (e[i].size() == 0)
		{
			bcc_node.push_back({i});
			bcc_edge.push_back({ });
			++bcc_n;
		}
	}
	void dfs(int u, bool rt)
	{
		dfn[u] = low[u] = id++;
		int cnt = 0;
		for (auto [v, w] : e[u]) if (!ed[w])
		{
			st[tp++] = w;
			ed[w] = 1;
			if (dfn[v] == -1)
			{
				dfs(v, 0);
				++cnt;
				cmin(low[u], low[v]);
				if (dfn[u] <= low[v])
				{
					ct[u] = cnt > rt;
					bcc_edge.push_back({ });
					do
					{
						bcc_edge[bcc_n].push_back(st[--tp]);
						blk[st[tp]] = bcc_n;
					} while (st[tp] != w);
					++bcc_n;
				}
			}
			else cmin(low[u], dfn[v]);
		}
	}
};
int main()
{
	int n, m, i;
	cin >> n >> m;
	vector<vector<pair<int, int>>> e(n);
	for (i = 0; i < m; i++)
	{
		int u, v;
		cin >> u >> v;
		e[u].push_back({v, i});
		e[v].push_back({u, i});
	}
	node_bcc bcc(e, m);
}
