template<class typC> struct bit
{
	vector<typC> a;
	int n;
	bit() { }
	bit(int nn) :n(nn), a(nn + 1) { }
	template<class T> bit(int nn, T *b) : n(nn), a(nn + 1)
	{
		for (int i = 1; i <= n; i++) a[i] = b[i - 1];
		for (int i = 1; i <= n; i++) if (i + (i & -i) <= n) a[i + (i & -i)] += a[i];
	}
	void add(int x, typC y)
	{
		//cerr<<"add "<<x<<" by "<<y<<endl;
		++x;
		x = clamp(x, 1, n + 1);
		if (x > n) return;
		assert(1 <= x && x <= n);
		a[x] += y;
		while ((x += x & -x) <= n) a[x] += y;
	}
	typC sum(int x)
	{
		//cerr<<"sum "<<x;
		++x;
		x = clamp(x, 0, n);
		assert(0 <= x && x <= n);
		typC r = a[x];
		while (x ^= x & -x) r += a[x];
		//cerr<<"= "<<r<<endl;
		return r;
	}
	typC sum(int x, int y)
	{
		return sum(y) - sum(x - 1);
	}
	int lower_bound(typC x)
	{
		if (n == 0) return 0;
		int i = __lg(n), j = 0;
		for (; i >= 0; i--) if ((1 << i | j) <= n && a[1 << i | j] < x) j |= 1 << i, x -= a[j];
		return j + 1;
	}
};
namespace DFS
{
	typedef long long ll;
	const int N = 1e5 + 5, M = 18;
	ll a[N];
	int st[M][N * 2], lg[N * 2];
	int dep[N], dfn[N], siz[N], f[N], szp[N], szn[N];
	vector<int> e[N], c[N], rg[N];
	bool ed[N];
	int n, ksiz, rt, mn, id;
	int lca(int u, int v)
	{
		u = dfn[u]; v = dfn[v];
		if (u > v) swap(u, v);
		int z = lg[v - u + 1];
		return dep[st[z][u]] < dep[st[z][v - (1 << z) + 1]] ? st[z][u] : st[z][v - (1 << z) + 1];
	}
	int dis(int u, int v)
	{
		return dep[u] + dep[v] - dep[lca(u, v)] * 2;
	}
	void findroot(int u)
	{
		ed[u] = siz[u] = 1;
		int mx = 0;
		for (int v : e[u]) if (!ed[v])
		{
			findroot(v);
			siz[u] += siz[v];
			mx = max(mx, siz[v]);
		}
		mx = max(mx, ksiz - siz[u]);
		ed[u] = 0;
		if (mn > mx) mn = mx, rt = u;
	}
	int dfs(int u)
	{
		mn = 1e9;
		findroot(u);
		u = rt;
		ed[u] = 1;
		for (int v : e[u]) if (!ed[v] && siz[v] > siz[u]) siz[v] = ksiz - siz[u];
		for (int v : e[u]) if (!ed[v])
		{
			ksiz = siz[v];
			c[u].push_back(dfs(v));
			f[c[u].back()] = u;
		}
		return u;
	}
	void pre_dfs(int u)
	{
		st[0][dfn[u] = ++id] = u;
		ed[u] = 1;
		for (int v : e[u]) if (!ed[v])
		{
			dep[v] = dep[u] + 1;
			pre_dfs(v);
			st[0][++id] = u;
		}
		ed[u] = 0;
	}
	void init(int _n)
	{
		n = _n; id = 0;
		int i;
		for (int i = 1; i <= n; i++)
		{
			e[i].clear();
			c[i].clear(); rg[i].clear();
			a[i] = f[i] = siz[i] = ed[i] = 0;
		}
	}
	void new_dfs(int u)
	{
		siz[u] = 1;
		for (int v : c[u]) new_dfs(v), siz[u] += siz[v];
		vector<int> &q = rg[u];
		q = {u};
		int ql = 0;
		while (ql < q.size())
		{
			int x = q[ql++];
			for (int v : c[x]) q.push_back(v);
		}
	}
	void fun()
	{
		pre_dfs(1);
		int i, j;
		for (i = 2; i <= id; i++) lg[i] = lg[i >> 1] + 1;
		for (j = 0; j < lg[id]; j++)
		{
			int R = id - (2 << j) + 1;
			for (i = 1; i <= R; i++) st[j + 1][i] = dep[st[j][i]] < dep[st[j][i + (1 << j)]] ? st[j][i] : st[j][i + (1 << j)];
		}
		ksiz = n;
		rt = dfs(1);
		new_dfs(rt);
	}
	vector<int> get(int u)
	{
		vector<int> st = {u};
		while (u = f[u]) st.push_back(u);
		return st;
	}
}
using DFS::init, DFS::fun, DFS::e, DFS::dis, DFS::rg, DFS::get;
