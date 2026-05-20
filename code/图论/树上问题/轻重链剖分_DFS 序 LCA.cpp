namespace HLD
{
	const int N = 5e5 + 2;
	vector<int> e[N];
	int dfn[N], nfd[N], dep[N], f[N], siz[N], hc[N], top[N];
	int id, n;
	void dfs1(int u)
	{
		siz[u] = 1;
		for (int v : e[u]) if (v != f[u])
		{
			dep[v] = dep[f[v] = u] + 1;
			dfs1(v);
			siz[u] += siz[v];
			if (siz[v] > siz[hc[u]]) hc[u] = v;
		}
	}
	void dfs2(int u)
	{
		dfn[u] = ++id;
		nfd[id] = u;
		if (hc[u])
		{
			top[hc[u]] = top[u];
			dfs2(hc[u]);
			for (int v : e[u]) if (v != hc[u] && v != f[u]) dfs2(top[v] = v);
		}
	}
	int lca(int u, int v)
	{
		while (top[u] != top[v])
		{
			if (dep[top[u]] < dep[top[v]]) swap(u, v);
			u = f[top[u]];
		}
		if (dep[u] > dep[v]) swap(u, v);
		return u;
	}
	int dis(int u, int v)
	{
		return dep[u] + dep[v] - (dep[lca(u, v)] << 1);
	}
	void init(int _n)
	{
		n = _n;
		for (int i = 1; i <= n; i++)
		{
			e[i].clear();
			f[i] = hc[i] = 0;
		}
		id = 0;
	}
	void fun(int root)
	{
		dep[root] = 1; dfs1(root); dfs2(top[root] = root);
	}
	vector<pair<int, int>> get_path(int u, int v)//u->v，注意可能出现 [r>l]（表示反过来走）
	{
		//cerr<<"path from "<<u<<" to "<<v<<": ";
		vector<pair<int, int>> v1, v2;
		while (top[u] != top[v])
		{
			if (dep[top[u]] > dep[top[v]]) v1.push_back({dfn[u], dfn[top[u]]}), u = f[top[u]];
			else v2.push_back({dfn[top[v]], dfn[v]}), v = f[top[v]];
		}
		v1.reserve(v1.size() + v2.size() + 1);
		v1.push_back({dfn[u], dfn[v]});
		reverse(v2.begin(), v2.end());
		for (auto v : v2) v1.push_back(v);
		//for (auto [x,y]:v1) cerr<<"["<<x<<','<<y<<"] ";cerr<<endl;
		return v1;
	}
}
using HLD::e, HLD::dfn, HLD::nfd, HLD::dep, HLD::f, HLD::siz, HLD::get_path;
using HLD::init;//5e5
namespace LCA
{
	using HLD::N, HLD::n;
	int st[__lg(N) + 1][N];
	int cmp(const int &x, const int &y) { return dep[x] < dep[y] ? x : y; }
	void fun(int rt)
	{
		HLD::fun(rt);
		assert(f[rt] == 0);
		for (int i = 1; i <= n; i++) st[0][dfn[i] - 1] = f[i];
		for (int j = 0; j < __lg(n); j++)
			for (int i = 1, k = n - (1 << j + 1); i <= k; i++)
				st[j + 1][i] = cmp(st[j][i], st[j][i + (1 << j)]);
	}
	int lca(int u, int v)
	{
		if (u == v) return u;
		u = dfn[u], v = dfn[v];
		if (u > v) swap(u, v);
		int g = __lg(v - u);
		return cmp(st[g][u], st[g][v - (1 << g)]);
	}
	int dis(int u, int v)
	{
		return dep[u] + dep[v] - (dep[lca(u, v)] << 1);
	}
}
using LCA::lca, LCA::fun, LCA::dis;

struct chain
{
	int u, v, w;
	chain(int u = 0, int v = 0) :u(u), v(v) { w = lca(u, v); }
	chain(int u, int v, int w) :u(u), v(v), w(w) { }
};
int deeper(int u, int v) { return dep[u] > dep[v] ? u : v; }
bool have(int u, int v) { return dfn[u] <= dfn[v] && dfn[v] < dfn[u] + siz[u]; }
chain operator&(chain a, chain b)
{
	if (!a.u || !b.u) return { };
	if (dep[a.w] > dep[b.w]) swap(a, b);
	auto [u1, v1, w1] = a;
	auto [u2, v2, w2] = b;
	if (have(w1, w2) && (have(w2, u1) || have(w2, v1)))
	{
		if (w1 == w2)
			return {deeper(lca(u1, v1), lca(u1, v2)),
			deeper(lca(u2, v1), lca(u2, v2))};
		int w = lca(u1, v1);
		if (w == w2) return {w2, lca(u2, v2), w2};
		if (w == w1) return {w2, deeper(lca(u2, v1), lca(u2, v2)), w2};
		return {w2, w, w2};
	}
	return { };
}

