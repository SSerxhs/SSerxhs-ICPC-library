
namespace blossom_tree
{
	const int N = 1005;
	vector<int> e[N];
	int lk[N], rt[N], f[N], dfn[N], typ[N], q[N];
	int id, h, t, n;
	int lca(int u, int v)
	{
		++id;
		while (1)
		{
			if (u)
			{
				if (dfn[u] == id) return u;
				dfn[u] = id; u = rt[f[lk[u]]];
			}
			swap(u, v);
		}
	}
	void blm(int u, int v, int a)
	{
		while (rt[u] != a)
		{
			f[u] = v;
			v = lk[u];
			if (typ[v] == 1) typ[q[++t] = v] = 0;
			rt[u] = rt[v] = a;
			u = f[v];
		}
	}
	void aug(int u)
	{
		while (u)
		{
			int v = lk[f[u]];
			lk[lk[u] = f[u]] = u;
			u = v;
		}
	}
	void bfs(int root)
	{
		memset(typ + 1, -1, n * sizeof typ[0]);
		iota(rt + 1, rt + n + 1, 1);
		typ[q[h = t = 1] = root] = 0;
		while (h <= t)
		{
			int u = q[h++];
			for (int v : e[u])
			{
				if (typ[v] == -1)
				{
					typ[v] = 1; f[v] = u;
					if (!lk[v]) return aug(v);
					typ[q[++t] = lk[v]] = 0;
				}
				else if (!typ[v] && rt[u] != rt[v])
				{
					int a = lca(rt[u], rt[v]);
					blm(v, u, a); blm(u, v, a);
				}
			}
		}
	}
	int max_general_match(int N, vector<pair<int, int>> edges)//[1,n]
	{
		n = N; id = 0;
		memset(f + 1, 0, n * sizeof f[0]);
		memset(dfn + 1, 0, n * sizeof dfn[0]);
		memset(lk + 1, 0, n * sizeof lk[0]);
		int i;
		for (i = 1; i <= n; i++) e[i].clear();
		mt19937 rnd(114);
		shuffle(all(edges), rnd);
		for (auto [u, v] : edges)
		{
			e[u].push_back(v), e[v].push_back(u);
			if (!(lk[u] || lk[v])) lk[u] = v, lk[v] = u;
		}
		int r = 0;
		for (i = 1; i <= n; i++) if (!lk[i]) bfs(i);
		for (i = 1; i <= n; i++) r += !!lk[i];
		return r / 2;
	}
}
using blossom_tree::max_general_match, blossom_tree::lk;

